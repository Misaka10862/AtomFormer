import torch
from torch.nn.utils.rnn import pad_sequence
from common import join

def pad(encoded, device):
    if any((x is None for x in encoded)):
        raise ValueError('out-of-vocabulary input')
    seqs = [torch.tensor(x, dtype=torch.long) for x in encoded]
    return (pad_sequence(seqs, batch_first=True).to(device), torch.tensor([len(x) for x in seqs]))

class LibInvent:

    def __init__(self, rows, cfg, device):
        from models.vocabulary import DecoratorVocabulary
        from models.decorator import Decorator
        from models.model import DecoratorModel
        self.device = device
        training_strings = [r['scaffold'] for r in rows] + [r['decoration'] for r in rows]
        self.vocab = DecoratorVocabulary.from_lists(training_strings, training_strings)
        params = {'num_layers': cfg['num_layers'], 'num_dimensions': cfg['hidden_size'], 'dropout': cfg['dropout']}
        network = Decorator(dict(params, vocabulary_size=self.vocab.len_scaffold()), dict(params, vocabulary_size=self.vocab.len_decoration())).to(device)
        self.model = DecoratorModel(self.vocab, network, cfg['max_length'], no_cuda=device == 'cpu')
        self.network = network
        self.max_length = cfg['max_length']

    def nll(self, rows, strings=None):
        ss, sl = pad([self.vocab.encode_scaffold(r['scaffold']) for r in rows], self.device)
        ds, dl = pad([self.vocab.encode_decoration(s) for s in (strings if strings is not None else [r['decoration'] for r in rows])], self.device)
        return self.model.likelihood(ss, sl, ds, dl)

    def encodable(self, row):
        return self.vocab.encode_scaffold(row['scaffold']) is not None and self.vocab.encode_decoration(row['decoration']) is not None

    @torch.no_grad()
    def sample(self, rows):
        ss, sl = pad([self.vocab.encode_scaffold(r['scaffold']) for r in rows], self.device)
        memory, hidden = self.network.forward_encoder(ss, sl)
        x = torch.full((len(rows), 1), self.vocab.decoration_vocabulary['^'], device=self.device, dtype=torch.long)
        active = torch.ones_like(x, dtype=torch.bool)
        lengths = torch.ones(len(rows), dtype=torch.long)
        generated = []
        for _ in range(self.max_length - 1):
            logits, hidden, _ = self.network.forward_decoder(x, lengths, memory, hidden)
            step_logits = logits[:, 0, :].clone()
            step_logits[:, 0] = -torch.inf
            x = torch.multinomial(step_logits.softmax(-1), 1)
            x = torch.where(active, x, torch.zeros_like(x))
            generated.append(x)
            active = active & (x > 1)
            if not active.any():
                break
        seqs = torch.cat(generated, 1).cpu().tolist()
        decs = [self.vocab.decode_decoration(s) for s in seqs]
        valid_end = [1 in s and 0 not in s[:s.index(1)] for s in seqs]
        outputs = ['__assemble__' if end else None for end in valid_end]
        return (outputs, decs)

class PromptModel:

    def __init__(self, rows, cfg, device):
        from smilesrnn.rnn import Model
        from smilesrnn.vocabulary import SMILESTokenizer, create_vocabulary
        self.device = device
        tokenizer = SMILESTokenizer()
        grammar = ['0123456789'] + ['%%%02d' % i for i in range(10, 100)] if cfg.get('prompt_ring_grammar', False) else []
        vocab = create_vocabulary([r['source'] for r in rows] + grammar, tokenizer)
        self.model = Model(vocab, tokenizer, network_params={'layer_size': cfg['hidden_size'], 'num_layers': cfg['num_layers'], 'embedding_layer_size': 256 if cfg['hidden_size'] >= 256 else cfg['hidden_size'], 'dropout': cfg['dropout']}, max_sequence_length=cfg['max_length'], device=torch.device(device))
        self.network = self.model.network
        self.prompts = {}

    def prompt(self, row):
        if row.get('prompt'):
            return row['prompt']
        if row['scaffold'] not in self.prompts:
            from promptsmiles import ScaffoldDecorator
            wrapper = ScaffoldDecorator(scaffold=row['scaffold'], batch_size=1, sample_fn=self.model._pSMILES_sample, evaluate_fn=self.model._pSMILES_evaluate, batch_prompts=True, optimize_prompts=False, shuffle=False)
            self.prompts[row['scaffold']] = wrapper.scaffolds[0]['variants'][0].strip_smiles
        return self.prompts[row['scaffold']]

    def encodable(self, row):
        try:
            for s in [row['source'], self.prompt(row).replace('[SH]', 'S')]:
                self.model.vocabulary.encode(self.model.tokenizer.tokenize(s))
            return True
        except (KeyError, ValueError, TypeError, IndexError):
            return False

    def nll(self, rows, strings=None):
        if strings is not None:
            strings = [s.replace('[SH]', 'S') for s in strings]
        return self.model.likelihood_smiles(strings if strings is not None else [r['source'] for r in rows])

    @torch.no_grad()
    def sample(self, rows):
        prompts = [self.prompt(r) for r in rows]
        out = self.model._pSMILES_sample(prompt=prompts, batch_size=len(rows), temperature=1.0)
        return (out, [None] * len(rows))

def create(method, rows, cfg, device):
    return {'libinvent': LibInvent, 'promptsmiles': PromptModel}[method](rows, cfg, device)
