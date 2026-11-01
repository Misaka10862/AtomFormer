"""Conditional equivariant DDPM: fixed scaffold, learned size, atoms and bonds.

Only Condition-derived tensors enter sample(). Targets are accepted exclusively by loss().
The research head is independent of the pinned external DiffDec baseline.
"""
import torch
from torch import nn
from .encoders import ConditionEncoder,masked_mean

ELEMENTS=(6,7,8,9,15,16,17,35,53)


class EquivariantLayer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(width*2+1,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.coordinate=nn.Sequential(nn.Linear(width,1),nn.Tanh())
        self.update=nn.Sequential(nn.Linear(width*2,width),nn.SiLU(),nn.Linear(width,width))
        self.norm=nn.LayerNorm(width)

    def forward(self,h,x,mask,movable):
        n=h.shape[1]
        delta=x[:,:,None,:]-x[:,None,:,:]
        d2=(delta**2).sum(-1,keepdim=True)
        pair=mask[:,:,None]&mask[:,None,:]&~torch.eye(n,dtype=torch.bool,device=h.device)[None]
        edge=self.message(torch.cat([h[:,:,None,:].expand(-1,-1,n,-1),h[:,None,:,:].expand(-1,n,-1,-1),d2],-1))
        edge=edge.masked_fill(~pair.unsqueeze(-1),0)
        denom=pair.sum(2,keepdim=True).clamp(min=1)
        shift=(delta/((d2+1e-8).sqrt()+1)*self.coordinate(edge)*pair.unsqueeze(-1)).sum(2)/denom
        x=x+shift*movable.unsqueeze(-1)
        h=self.norm(h+self.update(torch.cat([h,edge.sum(2)/denom],-1))).masked_fill(~mask.unsqueeze(-1),0)
        return h,x


class DiffusionGenerator(nn.Module):
    def __init__(self,config,encoder_kind='gated_geo',width=128,layers=6,steps=100,max_new_atoms=48,condition_features=True):
        super().__init__()
        if steps<2 or max_new_atoms<1:
            raise ValueError('Invalid diffusion dimensions')
        self.encoder=ConditionEncoder(config,encoder_kind)
        self.max_new_atoms=max_new_atoms
        self.steps=steps
        self.condition_features=condition_features
        self.project=nn.Linear(config.embedding_dim,width)
        self.context_atoms=nn.Embedding(128,width,padding_idx=0)
        self.roles=nn.Embedding(3,width)
        self.atom_input=nn.Linear(len(ELEMENTS),width)
        self.time=nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,width))
        self.layers=nn.ModuleList([EquivariantLayer(width) for _ in range(layers)])
        self.atom_head=nn.Linear(width,len(ELEMENTS))
        self.bond_head=nn.Sequential(nn.Linear(width*2+1,width),nn.SiLU(),nn.Linear(width,5))
        self.size_head=nn.Linear(config.embedding_dim,max_new_atoms)
        grid=torch.linspace(0,steps,steps+1,dtype=torch.float64)
        cumulative=torch.cos(((grid/steps+.008)/1.008)*torch.pi/2).square()
        cumulative=cumulative/cumulative[0]
        beta=(1-cumulative[1:]/cumulative[:-1]).clamp(.0001,.999).float()
        self.register_buffer('beta',beta)
        self.register_buffer('alpha_bar',torch.cumprod(1-beta,0))
        self.register_buffer('elements',torch.tensor(ELEMENTS,dtype=torch.long))

    def condition(self,batch):
        if not bool(batch['geometry_available'].all()):
            raise ValueError('Direct 3D requires mapped scaffold coordinates')
        memory,padding=self.encoder(batch)
        return memory,self.size_head(masked_mean(memory,~padding))

    def denoise(self,batch,noisy_pos,noisy_atoms,new_mask,t,memory):
        core_n=batch['z'].shape[1]; p=batch['pocket_z'].shape[1]; n=noisy_pos.shape[1]
        center=(batch['pos']*batch['anchor'].unsqueeze(-1)).sum(1,keepdim=True)
        core_x=(batch['pos']-center)/3.0
        pocket_x=(batch['pocket_pos']-center)/3.0
        core_h=self.project(memory) if self.condition_features else self.context_atoms(batch['z'])
        core_h=core_h+self.roles.weight[0]
        pocket_h=self.context_atoms(batch['pocket_z'])+self.roles.weight[1]
        new_h=self.atom_input(noisy_atoms)+self.roles.weight[2]
        h=torch.cat([core_h,pocket_h,new_h],1)
        time=t.to(h.dtype)/max(self.steps-1,1)
        h=h+self.time(torch.stack([time,torch.sin(time*torch.pi)],-1))[:,None,:]
        x=torch.cat([core_x,pocket_x,noisy_pos],1)
        mask=torch.cat([batch['mask'],batch['pocket_z'].ne(0),new_mask],1)
        movable=torch.cat([torch.zeros_like(batch['mask']),torch.zeros_like(batch['pocket_z'],dtype=torch.bool),new_mask],1)
        for layer in self.layers:
            h,x=layer(h,x,mask,movable)
        new_x=x[:,core_n+p:]
        new_logits=self.atom_head(h[:,core_n+p:])
        ligand_h=torch.cat([h[:,:core_n],h[:,core_n+p:]],1)
        ligand_x=torch.cat([core_x,new_x],1)
        hi,hj=ligand_h[:,:,None,:],ligand_h[:,None,:,:]
        d2=((ligand_x[:,:,None,:]-ligand_x[:,None,:,:])**2).sum(-1,keepdim=True)
        bonds=self.bond_head(torch.cat([hi+hj,(hi-hj).abs(),d2],-1))
        return {'pos':new_x,'atom_logits':new_logits,'bond_logits':bonds}

    def loss(self,batch,target_z,target_pos,target_bonds,target_mask,t=None,noise_pos=None,noise_atoms=None):
        memory,size_logits=self.condition(batch)
        sizes=target_mask.sum(1)
        if bool(((sizes<1)|(sizes>self.max_new_atoms)).any()):
            raise ValueError('Target size outside model support')
        match=target_z.unsqueeze(-1).eq(self.elements)
        if not bool((match.any(-1)|~target_mask).all()):
            raise ValueError('Unsupported target atom')
        atom_ids=match.long().argmax(-1)
        x0a=nn.functional.one_hot(atom_ids,len(ELEMENTS)).float()*target_mask.unsqueeze(-1)
        center=(batch['pos']*batch['anchor'].unsqueeze(-1)).sum(1,keepdim=True)
        x0=(target_pos-center)/3.0
        b=target_z.shape[0]
        t=torch.randint(self.steps,(b,),device=target_z.device) if t is None else t
        a=self.alpha_bar[t][:,None,None]
        ep=torch.randn_like(x0) if noise_pos is None else noise_pos
        ea=torch.randn_like(x0a) if noise_atoms is None else noise_atoms
        noisy_pos=(a.sqrt()*x0+(1-a).sqrt()*ep)*target_mask.unsqueeze(-1)
        noisy_atoms=(a.sqrt()*x0a+(1-a).sqrt()*ea)*target_mask.unsqueeze(-1)
        result=self.denoise(batch,noisy_pos,noisy_atoms,target_mask,t,memory)
        pos_loss=(((((result['pos']-x0)**2).sum(-1)*target_mask).sum(1))/target_mask.sum(1)).mean()
        atom_ce=nn.functional.cross_entropy(result['atom_logits'].transpose(1,2),atom_ids,reduction='none')
        atom_loss=((atom_ce*target_mask).sum(1)/target_mask.sum(1)).mean()
        core_n=batch['z'].shape[1]
        ligand_mask=torch.cat([batch['mask'],target_mask],1)
        pairs=ligand_mask[:,:,None]&ligand_mask[:,None,:]
        pairs[:,:core_n,:core_n]=False
        pairs=pairs&torch.triu(torch.ones_like(pairs,dtype=torch.bool),diagonal=1)
        bond_ce=nn.functional.cross_entropy(result['bond_logits'].permute(0,3,1,2),target_bonds,reduction='none')
        bond_loss=((bond_ce*pairs).sum((1,2))/pairs.sum((1,2)).clamp(min=1)).mean()
        size_loss=nn.functional.cross_entropy(size_logits,sizes-1)
        return {'loss':pos_loss+atom_loss+bond_loss+size_loss,'position_loss':pos_loss,
                'atom_loss':atom_loss,'bond_loss':bond_loss,'size_loss':size_loss}

    @torch.no_grad()
    def sample(self,batch):
        memory,size_logits=self.condition(batch)
        sizes=torch.multinomial(size_logits.softmax(-1),1).squeeze(1)+1
        n=int(sizes.max()); b=len(sizes); device=sizes.device
        mask=torch.arange(n,device=device)[None,:]<sizes[:,None]
        x=torch.randn(b,n,3,device=device)*mask.unsqueeze(-1)
        atoms=torch.randn(b,n,len(ELEMENTS),device=device)*mask.unsqueeze(-1)
        result=None
        for step in range(self.steps-1,-1,-1):
            t=torch.full((b,),step,dtype=torch.long,device=device)
            result=self.denoise(batch,x,atoms,mask,t,memory)
            pred_x=result['pos']; pred_a=result['atom_logits'].softmax(-1)
            if step==0:
                x,atoms=pred_x,pred_a
            else:
                abar=self.alpha_bar[step]; prev=self.alpha_bar[step-1]; beta=self.beta[step]; alpha=1-beta
                c0=prev.sqrt()*beta/(1-abar); ct=alpha.sqrt()*(1-prev)/(1-abar)
                sigma=(beta*(1-prev)/(1-abar)).sqrt()
                x=c0*pred_x+ct*x+sigma*torch.randn_like(x)
                atoms=c0*pred_a+ct*atoms+sigma*torch.randn_like(atoms)
            x=x*mask.unsqueeze(-1); atoms=atoms*mask.unsqueeze(-1)
        center=(batch['pos']*batch['anchor'].unsqueeze(-1)).sum(1,keepdim=True)
        ncore=batch['z'].shape[1]
        predicted_bonds=result['bond_logits'].argmax(-1)
        predicted_bonds[:,:ncore,:ncore]=batch['bonds']
        return {'new_z':self.elements[atoms.argmax(-1)],'new_pos':x*3.0+center,'new_mask':mask,
                'scaffold_pos':batch['pos'].clone(),'bonds':predicted_bonds,'sizes':sizes}


def collate_targets(rows,core_size,device='cpu'):
    targets=[r['target_3d'] for r in rows]
    n=max(len(t['z']) for t in targets); b=len(rows)
    z=torch.zeros(b,n,dtype=torch.long,device=device); pos=torch.zeros(b,n,3,device=device)
    bonds=torch.zeros(b,core_size+n,core_size+n,dtype=torch.long,device=device)
    mask=torch.zeros(b,n,dtype=torch.bool,device=device)
    for i,(r,t) in enumerate(zip(rows,targets)):
        k=len(t['z']); c=len(r['z'])
        z[i,:k]=torch.tensor(t['z'],device=device);pos[i,:k]=torch.tensor(t['pos'],device=device);mask[i,:k]=True
        indices=list(range(c))+list(range(core_size,core_size+k))
        for ii,old_i in enumerate(indices):
            for jj,old_j in enumerate(indices):
                bonds[i,old_i,old_j]=t['bonds'][ii][jj]
    return z,pos,bonds,mask
