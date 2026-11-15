"""Lossless bounded-memory checkpoints; tensor shards in one verified tar per state."""
import io
import json
import os
import shutil
import tarfile
import time
import uuid
from pathlib import Path

import torch

from archive_io import digest, safe_relative
from evaluation_records import encoded, identity, publish
from official_training import MANDATORY, rng_state, restore_rng


def parameter_groups(optimizer, named):
    names = {id(p): n for n, p in named}
    return [{**{k: v for k, v in g.items() if k != 'params'}, 'names': [names[id(p)] for p in g['params']]}
            for g in optimizer.param_groups]


def tensor_sources(named, optimizer):
    for n, p in named: yield ('model', n, None), p.detach()
    for n, p in named:
        for key, value in sorted(optimizer.state.get(p, {}).items()):
            assert isinstance(value, torch.Tensor), 'Unspecified non-tensor optimizer state'
            yield ('optimizer', n, key), value.detach()


def state_header(method, contract, update, history, optimizer, named, scaler, cuda):
    assert update >= 0 and len(history) == update
    assert [r['update'] for r in history] == list(range(1, update + 1))
    return {'schema_version': 1, 'method': method, 'contract_identity': identity(contract), 'contract': contract,
        'successful_updates': update, 'examples_seen': update * 32, 'history': history,
        'optimizer_groups': parameter_groups(optimizer, named), 'scaler': scaler.state_dict(), 'rng': rng_state(cuda),
        'scheduler': {'successful_updates': update, 'next_update': update + 1,
                      'schedule': 'approved_cosine_38_warmup_1250_updates_no_floor'},
        'frozen_weights_and_L2_reference': 'Rebuild from unchanged source/bases/initializers in contract; learned tensors saved below'}


def write_archive(path, header, named, optimizer, max_bytes=128 * 2**20, progress=lambda phase: None):
    """At most one bounded CPU shard buffer, plus its serialization/readback."""
    assert max_bytes >= 32
    path = Path(path); assert not path.exists()
    manifest = {'header': header, 'tensors': [], 'shards': []}
    buffer, pieces, used = {}, [], 0
    with tarfile.open(path, 'w:') as archive:
        def add(name, data):
            info = tarfile.TarInfo(name); info.size = len(data); info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
        def flush():
            nonlocal buffer, pieces, used
            if not buffer: return
            name = f'tensors_{len(manifest["shards"]):05}.pt'
            stream = io.BytesIO(); torch.save(buffer, stream); data = stream.getvalue()
            checked = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
            assert checked.keys() == buffer.keys()
            for key, value in buffer.items():
                assert torch.equal(value, checked[key]) and value.dtype == checked[key].dtype
            import hashlib
            add(name, data)
            manifest['shards'].append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'pieces': pieces})
            buffer, pieces, used = {}, [], 0
            progress('serialized and checked ' + name)
        for number, (source, tensor) in enumerate(tensor_sources(named, optimizer)):
            assert tensor.is_contiguous() and tensor.numel() > 0
            assert tensor.dtype == torch.float32, 'All learned tensors and Adam states must be FP32'
            spec = {'id': number, 'source': source, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                    'device_type': tensor.device.type, 'numel': tensor.numel()}
            manifest['tensors'].append(spec)
            flat = tensor.view(-1); offset = 0
            while offset < tensor.numel():
                room = (max_bytes - used) // tensor.element_size()
                if room == 0: flush(); continue
                count = min(room, tensor.numel() - offset)
                # clone prevents torch.save from serializing the large backing storage of a view.
                cpu = flat[offset:offset+count].to('cpu').clone().contiguous()
                assert bool(torch.isfinite(cpu).all())
                key = f'p{len(pieces):05}'
                buffer[key] = cpu
                pieces.append({'key': key, 'tensor_id': number, 'offset': offset, 'count': count})
                used += cpu.numel() * cpu.element_size(); offset += count
                if used >= max_bytes: flush()
        flush(); add('MANIFEST.json', encoded(manifest))
    return manifest


def verify_archive(path, expected=None, apply=None, progress=lambda phase: None, expected_archive_sha=None):
    """One sequential read verifies archive bytes and every decoded tensor shard."""
    import hashlib
    if expected is None:
        with tarfile.open(path, 'r:') as stream:
            expected = json.loads(stream.extractfile('MANIFEST.json').read())
    manifest = json.loads(encoded(expected))
    specs = {s['id']: s for s in manifest['tensors']}
    assert list(specs) == list(range(len(manifest['tensors'])))
    assert len({tuple(s['source']) for s in specs.values()}) == len(specs)
    covered = {k: 0 for k in specs}; shards = {s['path']: s for s in manifest['shards']}
    seen = set(); hasher = hashlib.sha256()
    class HashingReader:
        def __init__(self, stream): self.stream = stream
        def read(self, size=-1):
            data = self.stream.read(size); hasher.update(data); return data
    with Path(path).open('rb') as source:
        reader = HashingReader(source)
        with tarfile.open(fileobj=reader, mode='r|', bufsize=8*2**20) as archive:
            for member in archive:
                assert member.isfile() and '/' not in member.name and not member.name.startswith('.')
                assert member.name not in seen; seen.add(member.name)
                data = archive.extractfile(member).read()
                if member.name == 'MANIFEST.json':
                    assert json.loads(data) == manifest
                    continue
                assert member.name in shards
                shard = shards[member.name]
                assert len(data) == shard['bytes'] and hashlib.sha256(data).hexdigest() == shard['sha256']
                tensors = torch.load(io.BytesIO(data), map_location='cpu', weights_only=True)
                assert set(tensors) == {p['key'] for p in shard['pieces']}
                for piece in shard['pieces']:
                    spec = specs[piece['tensor_id']]; t = tensors[piece['key']]
                    assert t.dtype == torch.float32 and t.ndim == 1 and t.numel() == piece['count']
                    assert piece['offset'] == covered[spec['id']] and bool(torch.isfinite(t).all())
                    if apply is not None: apply(spec, piece['offset'], t)
                    covered[spec['id']] += t.numel()
                progress('verified ' + shard['path'])
        # Include any tar end padding not already consumed by the stream's read-ahead.
        while reader.read(8*2**20): pass
    assert seen == set(shards) | {'MANIFEST.json'}
    assert all(covered[k] == spec['numel'] == math_numel(spec['shape']) for k, spec in specs.items())
    if expected_archive_sha is not None: assert hasher.hexdigest() == expected_archive_sha
    return manifest


def math_numel(shape):
    count = 1
    for v in shape: assert isinstance(v, int) and v >= 1; count *= v
    return count


def save_checkpoint(directory, local, method, contract, update, history, named, optimizer, scaler,
                    cuda, kind, progress=lambda phase: None):
    assert kind in ('mandatory', 'rolling', 'exit', 'error')
    if kind == 'mandatory': assert update in MANDATORY
    reason = kind
    # Deadline/error recovery states use the same last-two rolling retention policy.
    kind = 'mandatory' if kind == 'mandatory' else 'rolling'
    directory, local = Path(directory), Path(local)
    header = state_header(method, contract, update, history, optimizer, named, scaler, cuda)
    signature = identity(header)
    for marker in directory.glob('*/COMPLETE.json'):
        prior = json.loads(marker.read_text())
        if prior.get('state_signature') == signature:
            assert prior['contract_identity'] == identity(contract)
            return marker.parent, prior
    needed = sum(t.numel() * t.element_size() for _, t in tensor_sources(named, optimizer))
    assert shutil.disk_usage(local).free >= needed + 2 * 2**30, 'Insufficient local checkpoint staging space'
    name = f'{kind}_{update:06}_{uuid.uuid4().hex[:12]}'
    local_tar = local / (name + '.tar')
    manifest = write_archive(local_tar, header, named, optimizer, progress=progress)
    archive_sha = digest(local_tar); destination = directory / name
    destination.mkdir(parents=True, exist_ok=False)
    publish(destination / 'OWNER.json', {'workflow': 'ATML_PA3_TASK1_TRAINING_V1', 'method': method,
        'contract_identity': identity(contract), 'kind': kind, 'reason': reason, 'update': update})
    remote_tar = destination / 'state.tar'
    progress('copy one checkpoint archive to Drive')
    with local_tar.open('rb') as src, remote_tar.open('xb') as dst:
        last = time.monotonic()
        for block in iter(lambda: src.read(8 * 2**20), b''):
            dst.write(block)
            if time.monotonic() - last > 2: progress('copy checkpoint archive'); last = time.monotonic()
    publish(destination / 'MANIFEST.json', manifest)
    verify_archive(remote_tar, manifest, progress=progress, expected_archive_sha=archive_sha)
    marker = {'schema_version': 1, 'method': method, 'kind': kind, 'successful_updates': update,
        'reason': reason,
        'examples_seen': update * 32, 'contract_identity': identity(contract), 'state_signature': signature,
        'archive_sha256': archive_sha, 'archive_bytes': remote_tar.stat().st_size,
        'manifest_identity': identity(manifest), 'created_wall': time.time()}
    publish(destination / 'COMPLETE.json', marker)  # Completeness is published LAST.
    local_tar.unlink()
    return destination, marker


def restore_checkpoint(path, contract, model, named, optimizer, scaler, progress=lambda phase: None):
    path = Path(path); marker = json.loads((path / 'COMPLETE.json').read_text())
    assert marker['contract_identity'] == identity(contract)
    archive = path / 'state.tar'; assert archive.stat().st_size == marker['archive_bytes']
    # Check the header before allocating optimizer tensors or changing fresh parameters.
    meta = json.loads((path / 'MANIFEST.json').read_text())
    assert identity(meta) == marker['manifest_identity'] and meta['header']['contract'] == contract
    header = meta['header']; assert identity(header) == marker['state_signature']
    assert header['successful_updates'] == marker['successful_updates'] and header['examples_seen'] == marker['examples_seen']
    assert len(header['history']) == header['successful_updates'] and header['examples_seen'] == header['successful_updates'] * 32
    current_names = dict(named); assert set(s['source'][1] for s in meta['tensors'] if s['source'][0] == 'model') == set(current_names)
    groups = parameter_groups(optimizer, named); assert len(groups) == len(header['optimizer_groups'])
    for current, saved, group in zip(groups, header['optimizer_groups'], optimizer.param_groups):
        assert current['names'] == saved['names']
        for k in ('betas', 'eps', 'weight_decay', 'foreach', 'fused'): assert current[k] == saved[k] or (k == 'betas' and list(current[k]) == saved[k])
        for key, value in saved.items():
            if key != 'names': group[key] = tuple(value) if key == 'betas' else value
    optimizer.state.clear()
    destinations = {}
    for spec in meta['tensors']:
        category, name, key = spec['source']; parameter = current_names[name]
        assert spec['dtype'] == 'torch.float32' and spec['device_type'] in ('cpu', 'cuda')
        if category == 'model':
            assert list(parameter.shape) == spec['shape'] and parameter.dtype == torch.float32
            destinations[spec['id']] = parameter
        else:
            assert category == 'optimizer' and key in ('step', 'exp_avg', 'exp_avg_sq')
            assert spec['shape'] == ([] if key == 'step' else list(parameter.shape))
            target_device = 'cpu' if spec['device_type'] == 'cpu' else parameter.device
            value = torch.empty(spec['shape'], dtype=torch.float32, device=target_device)
            optimizer.state[parameter][key] = value; destinations[spec['id']] = value
    def apply(spec, offset, value):
        with torch.no_grad(): destinations[spec['id']].view(-1)[offset:offset+value.numel()].copy_(value)
    verify_archive(archive, meta, apply, progress, expected_archive_sha=marker['archive_sha256'])
    if header['successful_updates']:
        for _, parameter in named:
            assert set(optimizer.state[parameter]) == {'step', 'exp_avg', 'exp_avg_sq'}
            assert int(optimizer.state[parameter]['step'].item()) == header['successful_updates']
    scaler.load_state_dict(header['scaler']); restore_rng(header['rng'])
    return header


def prune_rolling(directory, method, contract_identity, mandatory_update=None):
    """Only owner-verified rolling checkpoints of this exact run may be removed."""
    directory = Path(directory); eligible = []
    if mandatory_update is not None:
        verified = [json.loads(p.read_text()) for p in directory.glob('*/COMPLETE.json')]
        assert any(v.get('kind') == 'mandatory' and v.get('method') == method
            and v.get('contract_identity') == contract_identity and v.get('successful_updates') == mandatory_update
            for v in verified), 'A verified mandatory checkpoint must precede rolling cleanup'
    for marker in directory.glob('*/COMPLETE.json'):
        folder = marker.parent
        if folder.is_symlink() or not (folder / 'OWNER.json').is_file(): continue
        owner = json.loads((folder / 'OWNER.json').read_text()); complete = json.loads(marker.read_text())
        if (owner.get('workflow') != 'ATML_PA3_TASK1_TRAINING_V1' or owner.get('method') != method
            or owner.get('contract_identity') != contract_identity or owner.get('kind') != 'rolling'): continue
        assert complete['kind'] == 'rolling' and complete['contract_identity'] == contract_identity
        assert complete['method'] == method and complete['successful_updates'] == owner['update']
        eligible.append((complete['successful_updates'], complete['created_wall'], folder))
    eligible.sort()
    remove = [item for item in eligible if item[0] <= mandatory_update] if mandatory_update is not None else eligible[:-2]
    for _, _, folder in remove:
        assert folder.resolve().is_relative_to(directory.resolve())
        shutil.rmtree(folder)
    return [str(item[2]) for item in remove]
