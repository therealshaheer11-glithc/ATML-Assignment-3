"""Stage immutable, source-verified model files on CPU for later offline GPU use."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import re
import tarfile
import time
import uuid
from .images import encoded, file_sha, immutable_bytes, verified_copy


def safe_name(name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in ('.', '..') or '\\' in name:
        raise ValueError('Model asset must have a safe root-level filename.')
    return name


def field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def make_plan(key, lock, info):
    if key not in ('siglip', 'qwen') or field(info, 'sha') != lock['revision']:
        raise ValueError('Model source does not match the pinned revision.')
    siblings = {field(row, 'rfilename'): row for row in field(info, 'siblings')}
    selected = {'config.json', 'README.md', 'LICENSE', 'LICENSE.txt', 'NOTICE'} & siblings.keys()
    if key == 'siglip':
        selected |= {'preprocessor_config.json'} & siblings.keys()
        required = {'config.json', 'preprocessor_config.json'}
    else:
        selected |= {'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json',
            'merges.txt', 'special_tokens_map.json', 'added_tokens.json', 'chat_template.jinja'} & siblings.keys()
        required = {'config.json', 'tokenizer.json', 'tokenizer_config.json'}
    if 'model.safetensors' in siblings:
        selected.add('model.safetensors')
    else:
        shards = {name for name in siblings if re.fullmatch(r'model-\d{5}-of-\d{5}\.safetensors', name)}
        if not shards or 'model.safetensors.index.json' not in siblings:
            raise ValueError('A complete original safetensors checkpoint is required.')
        selected |= shards | {'model.safetensors.index.json'}
    if not required <= selected:
        raise ValueError('Pinned model is missing required configuration or tokenizer files.')
    records = []
    for name in sorted(selected):
        safe_name(name)
        row = siblings[name]
        lfs = field(row, 'lfs')
        size = field(row, 'size')
        source_hash = field(lfs, 'sha256') if lfs is not None else field(row, 'blob_id')
        algorithm = 'sha256' if lfs is not None else 'git-blob-sha1'
        if type(size) is not int or size <= 0 or not isinstance(source_hash, str):
            raise ValueError('Model source metadata lacks a size or content hash.')
        if not re.fullmatch('[0-9a-f]{' + ('64' if lfs is not None else '40') + '}', source_hash):
            raise ValueError('Model source hash has an unexpected format.')
        if lfs is not None and field(lfs, 'size') != size:
            raise ValueError('Model source size metadata conflicts.')
        records.append({'file_name': name, 'bytes': size, 'source_hash_algorithm': algorithm,
            'source_hash': source_hash})
    return {'schema': 'atml-pa3-task2-model-source-plan-v1', 'model_key': key, **lock, 'files': records}


def verify_source(path, record):
    path = Path(path)
    if path.stat().st_size != record['bytes']:
        raise IOError('Model asset size differs from its pinned source.')
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(b'blob ' + str(record['bytes']).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha256.update(block)
            blob.update(block)
    observed = sha256.hexdigest() if record['source_hash_algorithm'] == 'sha256' else blob.hexdigest()
    if record['source_hash_algorithm'] not in ('sha256', 'git-blob-sha1') or observed != record['source_hash']:
        raise IOError('Model asset differs from the hash published for its pinned source.')
    return sha256.hexdigest()


def safetensors_header(path):
    """Read tensor shapes and bounds without loading or executing model weights."""
    path = Path(path)
    with path.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Truncated safetensors header prefix.')
        size = int.from_bytes(prefix, 'little')
        if not 2 <= size <= min(100 * 1024 * 1024, path.stat().st_size - 8):
            raise ValueError('Invalid safetensors header length.')
        header = json.loads(stream.read(size))
    data_size = path.stat().st_size - 8 - size
    tensors = {k: v for k, v in header.items() if k != '__metadata__'}
    dtype_bytes = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4,
        'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    intervals = []
    for name, tensor in tensors.items():
        shape, offsets, dtype = tensor['shape'], tensor['data_offsets'], tensor['dtype']
        if dtype not in dtype_bytes or not all(type(n) is int and n >= 0 for n in shape):
            raise ValueError('Unexpected checkpoint tensor representation.')
        if len(offsets) != 2 or not all(type(n) is int for n in offsets):
            raise ValueError('Invalid checkpoint tensor offsets.')
        start, end = offsets
        if not 0 <= start <= end <= data_size or end - start != math.prod(shape) * dtype_bytes[dtype]:
            raise ValueError('Checkpoint tensor bytes do not match its shape.')
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError('Checkpoint tensor data overlaps or contains an unexplained gap.')
        cursor = end
    if not tensors or cursor != data_size:
        raise ValueError('Checkpoint tensor data is incomplete.')
    return tensors


def verify_model_structure(key, folder, records):
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_bytes())
    tensors, locations = {}, {}
    for record in records:
        name = record['file_name']
        if name.endswith('.safetensors'):
            header = safetensors_header(folder / name)
            if set(tensors) & set(header):
                raise ValueError('Duplicate tensors across checkpoint files.')
            tensors.update(header)
            locations.update({tensor: name for tensor in header})
    index = folder / 'model.safetensors.index.json'
    if index.exists():
        assert json.loads(index.read_bytes())['weight_map'] == locations
    if key == 'siglip':
        vision = config['vision_config']
        assert config['model_type'] == 'siglip'
        # The pinned original omits default hidden/image sizes. The tensor shapes
        # below independently verify the defaults already checked in Block 7.
        assert (vision.get('hidden_size', 768), vision.get('image_size', 224), vision['patch_size']) == (768, 224, 16)
        assert tensors['vision_model.embeddings.patch_embedding.weight']['shape'] == [768, 3, 16, 16]
        assert tensors['vision_model.embeddings.position_embedding.weight']['shape'] == [196, 768]
        assert tensors['vision_model.post_layernorm.weight']['shape'] == [768]
        assert tensors['vision_model.post_layernorm.bias']['shape'] == [768]
        processor = json.loads((folder / 'preprocessor_config.json').read_bytes())
        assert processor['size'] == {'height': 224, 'width': 224} and processor['resample'] == 3
        assert processor['do_resize'] and processor['do_rescale'] and processor['do_normalize']
        assert processor['image_mean'] == processor['image_std'] == [0.5, 0.5, 0.5]
        assert processor['rescale_factor'] == 1 / 255
    elif key == 'qwen':
        assert config['model_type'] == 'qwen3' and config['hidden_size'] == 1024
        assert config['eos_token_id'] == 151645
        assert tensors['model.embed_tokens.weight']['shape'] == [config['vocab_size'], 1024]
        assert 'quantization_config' not in config
    else:
        raise ValueError('Unexpected model key.')
    return {'model_type': config['model_type'], 'tensor_count': len(tensors),
        'tensor_elements': sum(math.prod(t['shape']) for t in tensors.values()),
        'tensor_dtype_counts': dict(Counter(t['dtype'] for t in tensors.values())),
        'weight_files': sorted(set(locations.values())), 'weights_instantiated': False}


def verify_model_archive(path, records):
    expected = [safe_name(r['file_name']) for r in records]
    if len(set(expected)) != len(expected):
        raise ValueError('Duplicate model archive members.')
    with tarfile.open(path, 'r:') as archive:
        members = archive.getmembers()
        if [m.name for m in members] != expected or not all(m.isfile() for m in members):
            raise ValueError('Model archive differs from the verified file list.')
        for member, record in zip(members, records):
            assert member.size == record['bytes']
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            assert digest.hexdigest() == record['sha256'], member.name


class ModelAssetStore:
    def __init__(self, run_root, local_root, key, lock, fetch):
        self.key, self.lock, self.fetch = key, lock, fetch
        self.durable_dir = Path(run_root) / 'model_assets' / key / lock['revision']
        self.local_dir = Path(local_root) / 'model_assets' / key / lock['revision']

    def prepare_file(self, plan, record):
        safe_name(record['file_name'])
        assert plan['model_key'] == self.key and all(plan[k] == v for k, v in self.lock.items())
        assert record in plan['files']
        plan_sha = hashlib.sha256(encoded(plan)).hexdigest()
        durable = self.durable_dir / 'originals' / record['file_name']
        local = self.local_dir / record['file_name']
        receipt_path = self.durable_dir / 'file_receipts' / (record['file_name'] + '.json')
        previous = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else None
        if previous is not None:
            assert previous['source_plan_sha256'] == plan_sha and previous['source'] == record
            assert previous['relative_path'] == str(durable.relative_to(self.durable_dir))
            expected = previous['sha256']
        else:
            expected = None
        for candidate in (local, durable):
            if candidate.is_file() and (expected is None or file_sha(candidate) == expected):
                try:
                    digest = verify_source(candidate, record)
                except (IOError, ValueError):
                    continue
                activity = 'reused'
                source = candidate
                break
        else:
            for attempt in range(3):
                try:
                    source = Path(self.fetch(self.lock, record['file_name'], attempt))
                    digest = verify_source(source, record)
                    if expected is not None and digest != expected:
                        raise IOError('Re-fetched model differs from its recorded original.')
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(attempt + 1)
            activity = 'downloaded'
        verified_copy(source, local, digest)
        verified_copy(local, durable, digest)
        receipt = {'schema': 'atml-pa3-task2-model-file-v1', 'source_plan_sha256': plan_sha,
            'source': record, 'sha256': digest, 'relative_path': str(durable.relative_to(self.durable_dir))}
        if previous is not None:
            assert previous == receipt
        immutable_bytes(receipt_path, encoded(receipt))
        return {**record, 'sha256': digest}, activity

    def pack(self, plan, records, structure):
        receipt_path = self.durable_dir / 'model_complete.json'
        plan_sha = hashlib.sha256(encoded(plan)).hexdigest()
        previous = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else None
        if previous is not None:
            assert previous['source_plan_sha256'] == plan_sha and previous['files'] == records
            assert previous['structure'] == structure
            relative = Path(previous['archive_relative_path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            durable = self.durable_dir / relative
            local = self.local_dir / durable.name
            if durable.is_file() and file_sha(durable) == previous['archive_sha256']:
                verify_model_archive(durable, records)
                return previous, 'reused'
            if local.is_file() and file_sha(local) == previous['archive_sha256']:
                verify_model_archive(local, records)
                verified_copy(local, durable, previous['archive_sha256'])
                return previous, 'restored'
        self.local_dir.mkdir(parents=True, exist_ok=True)
        temporary = self.local_dir / ('packing_' + uuid.uuid4().hex + '.tar')
        try:
            with tarfile.open(temporary, 'w', format=tarfile.USTAR_FORMAT) as archive:
                for record in records:
                    name = safe_name(record['file_name'])
                    source = self.local_dir / name
                    assert verify_source(source, record) == record['sha256']
                    info = tarfile.TarInfo(name)
                    info.size, info.mtime, info.mode = record['bytes'], 0, 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with source.open('rb') as stream:
                        archive.addfile(info, stream)
            verify_model_archive(temporary, records)
            digest = file_sha(temporary)
            if previous is not None:
                assert digest == previous['archive_sha256']
            name = 'original_model_' + digest[:16] + '.tar'
            local, durable = self.local_dir / name, self.durable_dir / name
            verified_copy(temporary, local, digest)
            verified_copy(local, durable, digest)
            receipt = {'schema': 'atml-pa3-task2-model-complete-v1', 'status': 'passed',
                'model_key': self.key, **self.lock, 'source_plan_sha256': plan_sha,
                'archive_relative_path': name, 'archive_sha256': digest, 'archive_bytes': durable.stat().st_size,
                'files': records, 'structure': structure}
            immutable_bytes(receipt_path, encoded(receipt))
            return receipt, 'packed'
        finally:
            temporary.unlink(missing_ok=True)
