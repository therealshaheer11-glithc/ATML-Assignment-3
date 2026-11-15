"""Native HFLM generation, with whole-batch raw-token saves and row provenance."""
import collections
import time
from pathlib import Path

import torch
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import stop_sequences_criteria

from baseline_runtime import boundary, progress
from batch8_records import load_batch, save_batch
from evaluation_records import identity


def semantic_tokens(tokenizer, new_ids, stop, eos_ids):
    """Replay the SAME stock single-row criteria to find its semantic stop.

    Native batched stock string stops may let an already stopped row continue
    while other rows finish. Preserve every raw token, but compare/flag answers
    only through the earliest stock single-row stop/EOS, including its last token.
    """
    criteria = stop_sequences_criteria(tokenizer, stop, 0, 1)
    eos_ids = set(eos_ids)
    for length, token in enumerate(new_ids, 1):
        if token in eos_ids:
            return new_ids[:length], 'eos'
        stopped = criteria(torch.tensor([new_ids[:length]], dtype=torch.long), None)
        if bool(stopped.all()): return new_ids[:length], 'stock_stop'
    return list(new_ids), 'cap_or_unfinished'


class BatchJournalHFLM(HFLM):
    def __init__(self, *args, job_config, audit_contract, journal, cpu_fixture=False, **kwargs):
        self.job_config, self.audit_contract = job_config, audit_contract
        self.journal = Path(journal)
        self.cpu_fixture = cpu_fixture
        self.events = []
        super().__init__(*args, **kwargs)
        if cpu_fixture:
            assert self.device.type == 'cpu' and audit_contract['condition'] == 'CPU_FIXTURE_ONLY'
        else:
            assert self.device.type == 'cuda' and not self.model.training
            assert all(p.dtype == torch.float16 for p in self.model.parameters())

    def _uncached_generate(self, context, max_length, stop, **kwargs):
        return super()._model_generate(context, max_length=max_length, stop=stop, **kwargs)

    def _model_generate(self, context, max_length, stop, **kwargs):
        assert 1 <= context.shape[0] <= 8
        if not self.cpu_fixture:
            assert not self.model.training and torch.are_deterministic_algorithms_enabled()
            assert not torch.is_deterministic_algorithms_warn_only_enabled()
            assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
            assert self.model.generation_config.num_beams == self.model.generation_config.num_return_sequences == 1
        converted = {k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        contract = {'audit': self.audit_contract, 'padded_input_ids': context.tolist(),
                    'max_length': max_length, 'stop': stop, 'generation_kwargs': converted}
        key = identity(contract); path = self.journal / (key + '.json')
        record = load_batch(path, contract); reused = record is not None
        if record is None:
            boundary(self.job_config, 'native batched greedy generation')
            start = time.monotonic()
            output = self._uncached_generate(context, max_length, stop, **kwargs)
            if self.device.type == 'cuda': torch.cuda.synchronize()
            seconds = time.monotonic() - start
            sequences = output.tolist(); width = context.shape[1]
            mask = converted['attention_mask']
            eos = self.model.generation_config.eos_token_id
            eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
            eos = [value for value in eos if value is not None]
            metadata = []
            for prefix, sequence, keep in zip(context.tolist(), sequences, mask):
                assert sequence[:width] == prefix
                actual_prompt = [t for t, flag in zip(prefix, keep) if flag]
                new = sequence[width:]
                semantic, reason = semantic_tokens(self.tokenizer, new, stop, eos)
                cap = max_length - width
                assert len(new) <= cap
                metadata.append({'input_ids': actual_prompt, 'semantic_new_ids': semantic,
                    'semantic_generated_tokens': len(semantic), 'raw_generated_tokens': len(new),
                    'stock_finish_reason': reason, 'max_new_tokens': cap,
                    'at_token_cap': len(semantic) == cap,
                    'truncated': len(semantic) == cap and reason == 'cap_or_unfinished',
                    'physical_batch_size': len(sequences), 'physical_batch_identity': key})
            progress(self.job_config['state'], 'autosave whole raw generation batch', saving=True)
            record = save_batch(path, contract, sequences, metadata, seconds)
        event = {'cache_key': key, 'raw_record': str(path), 'reused': reused, **record}
        self.events.append(event)
        progress(self.job_config['state'], 'raw generation batch verified')
        return torch.tensor(record['sequence_ids'], dtype=context.dtype, device=context.device)

    def generate_journaled(self, requests, batch_size):
        """Pass the FULL fixed chunk on resume; do not regroup missing rows."""
        assert batch_size in (1, 8)
        assert requests and len({identity(request.args[1]) for request in requests}) == 1
        self.batch_size_per_gpu = batch_size; self.events = []
        start = time.monotonic()
        outputs = self.generate_until(requests, disable_tqdm=True)
        elapsed = time.monotonic() - start
        queues = collections.defaultdict(list)
        for event in self.events:
            for row in event['row_metadata']:
                queues[tuple(row['input_ids'])].append({**row,
                    'raw_batch_record': event['raw_record'], 'raw_batch_reused': event['reused'],
                    'shared_model_inference_seconds': event['shared_model_inference_seconds']})
        metadata = []
        for request in requests:
            key = tuple(self.tok_encode(request.args[0]))
            assert queues[key], 'Native batching metadata was lost/reordered'
            metadata.append(queues[key].pop(0))
        assert not any(queues.values()) and len(outputs) == len(metadata) == len(requests)
        measured = sum(event['shared_model_inference_seconds'] for event in self.events if not event['reused'])
        return outputs, metadata, {'total_call_seconds': elapsed, 'model_inference_seconds_this_call': measured,
            'physical_batches': len(self.events), 'cache_hits': sum(event['reused'] for event in self.events)}
