"""Check approved batching semantics on CPU; preserve approval and shutdown evidence."""
import hashlib
import gzip
import importlib.metadata
import json
import re
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluation_records import identity, publish
from baseline_runtime import publish_bytes


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root, local, bundle):
    import torch
    import lm_eval
    from transformers import AutoTokenizer
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import load_yaml_config, apply_template
    from batch8_pilot import PROMPTS, make_requests
    from batch8_records import cache_checks
    from batch8_cpu_checks import run as native_checks
    root, local, bundle = Path(root), Path(local), Path(bundle)
    started = time.monotonic()
    assert not torch.cuda.is_available()
    assert torch.__version__ == '2.8.0+cu126'
    assert importlib.metadata.version('lm_eval') == '0.4.9.2'
    assert importlib.metadata.version('transformers') == '4.57.1'
    approval_path = bundle / 'APPROVAL_T1_A003.json'
    approval = json.loads(approval_path.read_text())
    proposal = bundle / 'PROPOSED_D10B.md'
    assert approval['approval_id'] == 'T1-A003' and sha(proposal) == approval['proposal_sha256']
    assert approval['user_answer'] == 'Approve pilot and conditional adoption (Recommended)'
    section = proposal.read_text().split('## Fixed non-benchmark pilot prompts', 1)[1].split('## Shutdown evidence', 1)[0]
    approved_prompts = re.findall(r'^\d\. (.+)$', section, flags=re.MULTILINE)
    assert approved_prompts == PROMPTS and len(PROMPTS) == 8
    print('1/4 Verifying frozen software and approved fixed non-benchmark prompts...', flush=True)
    hashes = json.loads((root / 'configs/HARNESS_SOURCE_HASHES.json').read_text())
    package = Path(lm_eval.__file__).parent
    for rel, expected in hashes.items(): assert sha(package / rel) == expected
    token_hashes = json.loads((root / 'manifests/TOKENIZER_FILES.json').read_text())
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
        assert sha(root / 'sources/original_model' / name) == token_hashes[name]
        assert sha(root / 'sources/tokenizer' / name) == token_hashes[name]
    tokenizer = AutoTokenizer.from_pretrained(root / 'sources/original_model', local_files_only=True)
    manager = TaskManager()
    cfg = load_yaml_config(manager.task_index['gsm8k_cot']['yaml_path'], mode='full')
    requests = make_requests(tokenizer)
    for question, request in zip(PROMPTS, requests):
        stock = apply_template(cfg['doc_to_text'], {'question': question})
        assert request.args[0] == tokenizer.apply_chat_template([{'role': 'user', 'content': stock}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert request.args[1]['until'] == cfg['generation_kwargs']['until']
        assert request.args[1]['max_gen_toks'] == 1024 and request.args[1]['do_sample'] is False
    original_ready = json.loads((root / 'configs/BASELINE_CPU_READY_V1.json').read_text())
    gsm = original_ready['benchmarks']['gsm8k_cot']; path = root / gsm['path']
    assert sha(path) == gsm['sha256']
    with gzip.open(path, 'rt') as stream: lengths = [len(json.loads(line)['prompt_input_ids']) for line in stream]
    assert len(lengths) == 1319
    work = local / ('batch8_CPU_FIXTURES_' + uuid.uuid4().hex)
    print('2/4 Testing physical-batch cache integrity and interruption handling...', flush=True)
    record_checks = cache_checks(work / 'records')
    print('3/4 Checking native harness sorting, padding, EOS/stops and resumed 25-question chunks...', flush=True)
    native = native_checks(root, work / 'native')
    print(json.dumps(native, indent=2), flush=True)
    print('4/4 Preserving the approval, pilot source and CPU readiness record...', flush=True)
    names = ('batch8_records.py', 'batch8_hflm.py', 'batch8_pilot.py', 'batch8_cpu_checks.py',
             'prepare_batch8_cpu.py', 'restore_batch8_cpu.py', 'archive_io.py',
             'restore_baseline_cpu.py', 'baseline_runtime.py', 'evaluation_records.py')
    files = {name: sha(bundle / name) for name in names}
    implementation = identity(files)
    destination = root / 'execution' / ('batch8_pilot_' + implementation)
    for name in names: publish_bytes(destination / name, (bundle / name).read_bytes())
    publish_bytes(root / 'provenance/APPROVAL_T1_A003.json', approval_path.read_bytes())
    publish_bytes(root / 'provenance' / ('D10B_proposal_' + approval['proposal_sha256'] + '.md'), proposal.read_bytes())
    ready = {'status': 'TASK1_BATCH8_CPU_CHECKS_OK', 'approval_id': 'T1-A003',
        'implementation_identity': implementation, 'implementation_path': str(destination.relative_to(root)),
        'files': files, 'fixed_prompt_identity': identity(PROMPTS),
        'maximum_prepared_GSM_prompt_tokens': max(lengths),
        'record_checks': record_checks, 'native_CPU_checks': native,
        'pilot_limits': {'maximum_minutes': 15, 'maximum_estimated_units': .6,
                         'rate_margin': 1.1, 'minimum_headroom': .1, 'minimum_speedup': 2.},
        'FP16_absolute_logit_limit': .1, 'FP16_relative_RMS_limit': .005,
        'GPU_pilot_passed': False, 'full_adoption_gate_passed': False,
        'shutdown_trace_review_required_before_GPU': True,
        'official_benchmark_answers_generated': 0, 'official_training_updates': 0}
    publish(root / 'configs' / ('BATCH8_CPU_CHECKS_' + implementation + '.json'), ready)
    receipt = {**ready, 'elapsed_CPU_check_seconds': time.monotonic() - started,
        'status': 'TASK1_BLOCK13_CPU_BATCH8_OK',
        'next': 'Return this summary AND the shutdown log readback above. Keep CPU selected; no GPU pilot authorization to launch until the trace is reviewed.'}
    publish(root / 'logs' / ('block13_cpu_batch8_' + uuid.uuid4().hex + '.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__': main(*sys.argv[1:])
