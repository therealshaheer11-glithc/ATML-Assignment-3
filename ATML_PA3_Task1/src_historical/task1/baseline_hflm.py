"""Native pinned HFLM with token-level autosave and bounded work boundaries."""
import time
from pathlib import Path

import torch
from lm_eval.models.huggingface import HFLM

from baseline_runtime import boundary, progress, termination, save_generation, load_generation
from evaluation_records import identity


class AuditedHFLM(HFLM):
    def __init__(self, *args, job_config, audit_contract, **kwargs):
        self.job_config = job_config
        self.audit_contract = audit_contract
        self.last_generation = None
        self.raw_directory = Path(job_config['root']) / 'results/original/raw_generation'
        super().__init__(*args, **kwargs)

    def _model_call(self, inps, *args, **kwargs):
        assert not self.model.training
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        boundary(self.job_config, 'baseline likelihood forward')
        result = super()._model_call(inps, *args, **kwargs)
        progress(self.job_config['state'], 'baseline likelihood forward completed')
        return result

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        assert not self.model.training
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        assert self.model.generation_config.num_beams == 1 and self.model.generation_config.num_return_sequences == 1
        assert context.shape[0] == 1, 'The approved generation batch size is one'
        semantic_kwargs = {key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                           for key, value in generation_kwargs.items()}
        contract = {'audit': self.audit_contract, 'input_ids': context[0].tolist(),
            'max_length': max_length, 'stop': stop, 'generation_kwargs': semantic_kwargs}
        key = identity(contract); path = self.raw_directory / (key + '.json')
        cached = load_generation(path, contract)
        if cached is not None:
            self.last_generation = {**cached['metadata'], 'raw_record': str(path.relative_to(self.job_config['root'])),
                                    'raw_cache_reused': True, 'inference_seconds_this_call': 0.}
            return torch.tensor(cached['sequence_ids'], dtype=context.dtype, device=context.device)
        boundary(self.job_config, 'baseline greedy generation')
        started = time.monotonic()
        result = super()._model_generate(context, max_length=max_length, stop=stop, **generation_kwargs)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        sequence = result.tolist()
        assert len(sequence) == 1 and sequence[0][:context.shape[1]] == context[0].tolist()
        new = sequence[0][context.shape[1]:]
        eos = self.model.generation_config.eos_token_id
        eos = list(eos) if isinstance(eos, (tuple, list)) else [eos]
        eos = [int(value) for value in eos if value is not None]
        metadata = termination(new, eos, stop,
            self.tokenizer.decode(new, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            max_length - context.shape[1])
        metadata.update(original_inference_seconds=seconds, input_tokens=context.shape[1],
                        raw_cache_key=key, seed=6304, do_sample=False)
        progress(self.job_config['state'], 'autosave generated tokens', saving=True)
        save_generation(path, contract, sequence, metadata)
        self.last_generation = {**metadata, 'raw_record': str(path.relative_to(self.job_config['root'])),
            'raw_cache_reused': False, 'inference_seconds_this_call': seconds}
        progress(self.job_config['state'], 'generated tokens saved and read back')
        return result

    def infer(self, kind, requests):
        if kind == 'generate_until':
            self.batch_size_per_gpu = 1
            answers, metadata = [], []
            for request in requests:
                # Native stock stop handling/postprocessing remains in generate_until.
                answer = self.generate_until([request], disable_tqdm=True)
                assert len(answer) == 1 and self.last_generation is not None
                answers.extend(answer); metadata.append(dict(self.last_generation))
            progress(self.job_config['state'], 'save generated-answer score records', saving=True)
            return answers, metadata
        if kind != 'loglikelihood': raise ValueError('Unexpected inference request type')
        self.batch_size_per_gpu = 2
        boundary(self.job_config, 'baseline likelihood chunk')
        started = time.monotonic()
        answers = self.loglikelihood(requests, disable_tqdm=True)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        chunk_id = identity({'contract': self.audit_contract,
                             'requests': [request.args for request in requests]})
        metadata = [{'likelihood_chunk_identity': chunk_id,
            'shared_chunk_inference_seconds': elapsed, 'chunk_requests': len(requests),
            'shared_duration_must_not_be_summed_per_choice': True} for _ in answers]
        progress(self.job_config['state'], 'save likelihood score records', saving=True)
        return answers, metadata
