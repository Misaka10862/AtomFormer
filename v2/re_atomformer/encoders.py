"""Attachment-aware graph and gated geometry encoders with a shared memory interface."""
import torch
from torch import nn
from condmolgen.geometry import GeoEncoderConfig, GeoformerEncoder
from condmolgen.model import AtomTypeEncoder


def masked_mean(x, mask):
    if bool((mask.sum(1)==0).any()):
        raise ValueError('Empty molecular graph')
    return x.masked_fill(~mask.unsqueeze(-1),0).sum(1)/mask.sum(1,keepdim=True)


class ScaffoldSmilesEncoder(nn.Module):
    """Independent character embeddings; the input contains scaffold tokens only."""
    def __init__(self, config):
        super().__init__()
        self.pad = config.decoder_pad_token_id
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=self.pad)
        self.position = nn.Embedding(config.scaffold_block_size, config.embedding_dim)
        self.norm = nn.LayerNorm(config.embedding_dim)
        layer = nn.TransformerEncoderLayer(config.embedding_dim, config.encoder_heads,
            config.encoder_ffn_dim, config.dropout, activation='gelu', batch_first=True,
            norm_first=config.encoder_norm_first)
        layer.self_attn.dropout = config.atom_attention_dropout
        self.layers = nn.TransformerEncoder(layer, config.encoder_layers)

    def forward(self, batch, geometry_enabled=True):
        tokens = batch['scaffold_input_ids']
        if tokens.shape[1] > self.position.num_embeddings:
            raise ValueError('Scaffold encoder length exceeded; input must not be silently dropped')
        padding = tokens.eq(self.pad)
        if bool(padding.all(1).any()):
            raise ValueError('Empty scaffold token sequence')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        memory = self.layers(self.norm(self.embedding(tokens) + self.position(positions)),
                             src_key_padding_mask=padding)
        return memory, padding


class PrefixOnlyCondition(nn.Module):
    """Parameter-free placeholder; decoder blocks contain no cross-attention."""
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, batch, geometry_enabled=True):
        return (batch['pos'].new_zeros((len(batch['z']), 1, self.width)),
                batch['mask'].new_zeros((len(batch['z']), 1)))


class GraphEncoder(nn.Module):
    def __init__(self,width=256,layers=3,dropout=.1):
        super().__init__()
        self.atom=nn.Embedding(128,width,padding_idx=0)
        self.charge=nn.Embedding(11,width)
        self.aromatic=nn.Embedding(2,width)
        self.anchor=nn.Embedding(2,width)
        self.bond=nn.Embedding(5,width,padding_idx=0)
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(width*2,width*2),nn.GELU(),nn.Linear(width*2,width),nn.Dropout(dropout)) for _ in range(layers)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])

    def forward(self,batch):
        mask=batch['mask']; bonds=batch['bonds']
        if bool((batch['charges'].abs()>5).any()):
            raise ValueError('Unsupported atom formal charge')
        h=self.atom(batch['z'])+self.charge(batch['charges']+5)+self.aromatic(batch['aromatic'])+self.anchor(batch['anchor'].long())
        edges=bonds.ne(0)&mask.unsqueeze(1)&mask.unsqueeze(2)
        for layer,norm in zip(self.updates,self.norms):
            messages=h.unsqueeze(1)+self.bond(bonds)
            messages=messages.masked_fill(~edges.unsqueeze(-1),0).sum(2)/edges.sum(2,keepdim=True).clamp(min=1)
            h=norm(h+layer(torch.cat([h,messages],-1))).masked_fill(~mask.unsqueeze(-1),0)
        return h


class ConditionEncoder(nn.Module):
    def __init__(self,config,kind='graph',gate_initial=.01):
        super().__init__()
        self.kind=kind
        self.geometry_enabled=True
        self.graph=GraphEncoder(config.embedding_dim,config.encoder_layers,config.dropout) if kind not in ('atom','geo') else None
        self.atom=AtomTypeEncoder(config) if kind=='atom' else None
        self.geo=None
        if kind in ('geo','gated_geo','pocket_geo'):
            self.geo=GeoformerEncoder(GeoEncoderConfig(embedding_dim=config.embedding_dim,ffn_embedding_dim=config.encoder_ffn_dim,
                   num_layers=config.encoder_layers,num_attention_heads=config.encoder_heads,dropout=config.dropout,
                   attention_dropout=config.attention_dropout,num_rbf=config.num_rbf,cutoff=config.cutoff))
        self.gate=nn.Parameter(torch.tensor(gate_initial)) if kind in ('gated_geo','pocket_geo','pocket_distance') else None
        if kind=='pocket_distance':
            self.pocket_atoms=nn.Embedding(128,config.embedding_dim,padding_idx=0)
            self.distance=nn.Sequential(nn.Linear(16,config.embedding_dim),nn.SiLU(),nn.Linear(config.embedding_dim,config.embedding_dim))
            self.register_buffer('rbf_centers',torch.linspace(0,10,16))
        if kind not in ('atom','geo','graph','gated_geo','pocket_geo','pocket_distance'):
            raise ValueError('Unknown encoder '+kind)

    def forward(self,batch,geometry_enabled=True):
        geometry_enabled=geometry_enabled and self.geometry_enabled
        z,mask=batch['z'],batch['mask']
        if self.kind=='atom':
            atom_z=batch.get('legacy_atom_z',z)
            return self.atom(atom_z,batch['pos'])[0],atom_z.eq(0)
        if self.kind=='graph':
            return self.graph(batch),~mask
        if geometry_enabled and not bool(batch['geometry_available'].all()):
            raise ValueError('Geometry encoder requires explicit valid coordinates')
        if self.kind=='geo':
            if not geometry_enabled:
                raise ValueError('Legacy geo has no graph fallback')
            return self.geo(z,batch['pos'])[0],~mask
        graph=self.graph(batch)
        if not geometry_enabled:
            return graph,~mask
        if self.kind=='pocket_distance':
            pz,pp=batch['pocket_z'],batch['pocket_pos']
            if pz.shape[1]==0:
                return graph,~mask
            distance=torch.cdist(batch['pos'],pp)
            pairs=mask.unsqueeze(2)&pz.ne(0).unsqueeze(1)
            features=self.distance(torch.exp(-(distance.unsqueeze(-1)-self.rbf_centers)**2)) + self.pocket_atoms(pz).unsqueeze(1)
            context=features.masked_fill(~pairs.unsqueeze(-1),0).sum(2)/pairs.sum(2,keepdim=True).clamp(min=1)
        else:
            gz,gp=z,batch['pos']
            if self.kind=='pocket_geo':
                gz=torch.cat([z,batch['pocket_z']],1); gp=torch.cat([gp,batch['pocket_pos']],1)
            context=self.geo(gz,gp)[0][:,:z.shape[1]]
        return (graph+torch.tanh(self.gate)*context).masked_fill(~mask.unsqueeze(-1),0),~mask

    def numerical_counters(self):
        return {'amp_fallbacks':sum(getattr(m,'_amp_fallback_count',0) for m in self.modules()),
                'geometry_gate':float(torch.tanh(self.gate).detach()) if self.gate is not None else None}
