"""Regression checks for the actual split model/tokenizer manifest schemas."""
import hashlib
import json
from pathlib import Path


def run_checks(verify, directory):
    directory = Path(directory)
    model = {'repo': 'Qwen/Qwen3-0.6B', 'revision': 'a' * 40}
    checks = {}

    def fixture(name):
        root = directory / name
        payloads = {'config.json': b'{"model_type":"qwen3"}\n',
                    'generation_config.json': b'{"eos_token_id":151645}\n',
                    'tokenizer.json': b'{"fixture":"canonical tokenizer bytes"}\n',
                    'tokenizer_config.json': b'{"fixture":"canonical tokenizer configuration"}\n'}
        for folder in ('sources/original_model', 'sources/tokenizer', 'manifests', 'configs'):
            (root / folder).mkdir(parents=True)
        for filename, data in payloads.items():
            (root / 'sources/original_model' / filename).write_bytes(data)
            if filename != 'generation_config.json':
                (root / 'sources/tokenizer' / filename).write_bytes(data)
        hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
        # These are the schemas emitted by prepare_assets.py and prepare_data.py.
        originals = {'model': model, 'files': {name: {'sha256': hashes[name],
            'bytes': len(payloads[name]), 'official_lfs_hash_checked': False}
            for name in ('config.json', 'generation_config.json')}}
        tokenizer = {name: hashes[name] for name in
                     ('config.json', 'tokenizer_config.json', 'tokenizer.json')}
        (root / 'manifests/ORIGINAL_MODEL_FILES.json').write_text(json.dumps(originals))
        (root / 'manifests/TOKENIZER_FILES.json').write_text(json.dumps(tokenizer))
        (root / 'configs/SOURCE_REVISIONS.json').write_text(json.dumps({'model': model}))
        return root

    root = fixture('valid_split_manifests')
    result = verify(root)
    assert len(result) == 7
    assert 'tokenizer.json' not in json.loads((root / 'manifests/ORIGINAL_MODEL_FILES.json').read_text())['files']
    checks['tokenizer_absent_from_model_manifest_is_valid'] = True
    checks['canonical_and_copied_tokenizer_bytes_checked'] = True

    corrupt_paths = (
        'sources/tokenizer/tokenizer.json',
        'sources/original_model/tokenizer.json',
        'sources/original_model/config.json',
        'sources/original_model/generation_config.json',
    )
    for i, rel in enumerate(corrupt_paths):
        root = fixture('corruption_' + str(i))
        (root / rel).write_bytes(b'CHANGED')
        try:
            verify(root)
        except RuntimeError:
            checks['reject_changed_' + rel.replace('/', '_')] = True
        else:
            raise AssertionError('Accepted changed asset: ' + rel)
    root = fixture('missing_tokenizer_hash')
    manifest_path = root / 'manifests/TOKENIZER_FILES.json'
    manifest = json.loads(manifest_path.read_text()); del manifest['tokenizer.json']
    manifest_path.write_text(json.dumps(manifest))
    try:
        verify(root)
    except KeyError:
        checks['missing_frozen_hash_rejected'] = True
    else:
        raise AssertionError('Missing hash was accepted')
    root = fixture('different_source_revision')
    (root / 'configs/SOURCE_REVISIONS.json').write_text(json.dumps({'model': {**model, 'revision': 'b' * 40}}))
    try:
        verify(root)
    except RuntimeError:
        checks['source_revision_mismatch_rejected'] = True
    else:
        raise AssertionError('Changed source revision was accepted')
    return checks
