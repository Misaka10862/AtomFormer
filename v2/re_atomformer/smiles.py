"""Rooted decoder retains the AtomFormer token/action protocol with new conditions."""
from dataclasses import replace
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from condmolgen.model import ConditionalMolGenerator
from condmolgen.v4_model import build_prefix_batch,gather_last_hidden,append_tokens_at_lengths,decode_generated_decorations
from .encoders import ConditionEncoder,masked_mean,ScaffoldSmilesEncoder,PrefixOnlyCondition


class ResearchSmiles(ConditionalMolGenerator):
    def __init__(self,config,encoder_kind='graph',gate_initial=.01):
        if encoder_kind == 'prefix_only':
            config = replace(config, anchor_aux_weight=0., scaffold_aux_weight=0., decoder_cross_attention=False)
        super().__init__(replace(config,encoder_type='atom'))
        if encoder_kind == 'smiles':
            self.encoder = ScaffoldSmilesEncoder(config)
        elif encoder_kind == 'prefix_only':
            self.encoder = PrefixOnlyCondition(config.embedding_dim)
            self.anchor_head = None
            self.scaffold_aux_head = None
        else:
            self.encoder=ConditionEncoder(config,encoder_kind,gate_initial)
        self.encoder_kind=encoder_kind
        self.policy_checkpoint=True

    def forward(self,batch,geometry_enabled=True):
        memory,padding=self.encoder(batch,geometry_enabled=geometry_enabled)
        hidden=self.decoder(batch['input_ids'],memory,batch['input_ids'].eq(self.config.decoder_pad_token_id),padding)
        logits=self.lm_head(hidden)
        result={'logits':logits,'memory':memory}
        if 'labels' in batch:
            targets=batch['labels'][:,1:]
            token_losses=nn.functional.cross_entropy(logits[:,:-1].transpose(1,2),targets,ignore_index=-100,reduction='none')
            valid=targets.ne(-100)
            if not bool(valid.any()):
                raise ValueError('No target tokens')
            lm=token_losses.sum()/valid.sum()
            result.update(lm_loss=lm,loss=lm,token_count=int(valid.sum()),nll=token_losses.sum(1))
            if self.config.anchor_aux_weight>0:
                pooled=memory.mean(1) if self.config.auxiliary_pooling=='legacy_mean' else masked_mean(memory,~padding)
                aux_logits=self.anchor_head(pooled)
                aux=nn.functional.cross_entropy(aux_logits,batch['anchor_z'])
                result.update(anchor_logits=aux_logits,anchor_aux_loss=aux,loss=lm+self.config.anchor_aux_weight*aux)
        return result

    def sample(self,batch,scaffold_smiles,tokenizer,max_length=256,temperature=1.,with_grad=False):
        if temperature<=0:
            raise ValueError('Sampling temperature must be positive')
        with torch.set_grad_enabled(with_grad):
            if self.encoder_kind == 'smiles':
                from .data import add_scaffold_tokens
                batch = add_scaffold_tokens(batch, scaffold_smiles, tokenizer)
            memory,padding=self.encoder(batch)
            generated,lengths=build_prefix_batch(tokenizer,scaffold_smiles,batch['z'].device)
            if bool((lengths>=max_length).any()):
                raise ValueError('Scaffold prefix exhausts generation budget')
            prefix_lengths=lengths.clone()
            finished=torch.zeros(len(scaffold_smiles),dtype=torch.bool,device=generated.device)
            logs,masks,entropy=[],[],[]
            while bool(((~finished)&lengths.lt(max_length)).any()):
                def step_logits(tokens,encoded,encoded_padding,current_lengths):
                    hidden=self.decoder(tokens,encoded,tokens.eq(tokenizer.pad_token_id),encoded_padding)
                    return self.lm_head(gather_last_hidden(hidden,current_lengths))/temperature
                if with_grad and self.policy_checkpoint:
                    # Recompute decoder activations during backward; retain the exact dropout RNG.
                    # Pass growing tensors explicitly rather than closing over changing loop state.
                    logits=checkpoint(step_logits,generated,memory,padding,lengths,use_reentrant=False,preserve_rng_state=True)
                else:
                    logits=step_logits(generated,memory,padding,lengths)
                log_probs=logits.log_softmax(-1)
                next_token=torch.multinomial(log_probs.exp(),1).squeeze(-1)
                active=(~finished)&lengths.lt(max_length)
                logs.append(log_probs.gather(1,next_token[:,None]).squeeze(1))
                masks.append(active.float())
                entropy.append(-(log_probs.exp()*log_probs).sum(-1))
                generated=append_tokens_at_lengths(generated,lengths,next_token,active,tokenizer.pad_token_id)
                lengths=lengths+active.long()
                finished=finished|(active&next_token.eq(tokenizer.eos_token_id))|lengths.ge(max_length)
            texts=decode_generated_decorations(tokenizer,generated,prefix_lengths)
            ended=[bool((generated[i,prefix_lengths[i]:lengths[i]]==tokenizer.eos_token_id).any()) for i in range(len(texts))]
            return {'decorations':[t if ok else None for t,ok in zip(texts,ended)],'token_logprobs':torch.stack(logs,1),
                    'action_mask':torch.stack(masks,1),'entropy':torch.stack(entropy,1),'generated':generated}
