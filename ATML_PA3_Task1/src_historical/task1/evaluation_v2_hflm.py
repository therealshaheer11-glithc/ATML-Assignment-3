"""Approved batch-eight generation and unchanged native batch-two likelihood."""
import time
import torch
from batch8_hflm import BatchJournalHFLM
from baseline_runtime import boundary, progress
from evaluation_records import identity


class EvaluationHFLM(BatchJournalHFLM):
    def _model_call(self, inps, *args, **kwargs):
        assert not self.model.training
        assert torch.are_deterministic_algorithms_enabled() and not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32
        boundary(self.job_config, 'native likelihood batch-two forward')
        result = super()._model_call(inps, *args, **kwargs)
        progress(self.job_config['state'], 'native likelihood forward completed')
        return result

    def infer(self, kind, requests):
        if kind == 'generate_until':
            answers, metadata, timing = self.generate_journaled(requests, 8)
            progress(self.job_config['state'], 'save batch-eight scored answer records', saving=True)
            return answers, metadata
        assert kind == 'loglikelihood'
        self.batch_size_per_gpu = 2
        boundary(self.job_config, 'native fixed likelihood chunk')
        started = time.monotonic()
        answers = self.loglikelihood(requests, disable_tqdm=True)
        torch.cuda.synchronize()
        duration = time.monotonic() - started
        key = identity({'audit': self.audit_contract, 'requests': [r.args for r in requests]})
        metadata = [{'likelihood_chunk_identity': key, 'shared_chunk_inference_seconds': duration,
            'shared_duration_must_not_be_summed_per_choice': True} for _ in answers]
        progress(self.job_config['state'], 'save complete native likelihood responses', saving=True)
        return answers, metadata
