"""Restore the trained OPLoRA factors with the original frozen FP32 bases."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

CONTRACT = '8cafd5243fcac8c1e3cd615a0497dfb2812b73e4ce46a997b161e85f504165a8'
MANIFEST = 'c51b64f04a6e9ebb37c9626b4159da4806eba054255b146cd16a253730168606'
CHECKPOINT = '69862cbaad0e36fef5f8ffe87028c0b8bb3b5f7b768d9e9cdd662cf8434682b9'
WEIGHTS = 'b39053eb70727fa6ad0b9b50a824f186e33249d5f0c281b4d46bcaeac23d31b9'


def prepare_bases(root, job, bundle, state):
    from archive_io import digest, safe_relative
    from evaluation_records import identity
    from baseline_runtime import progress, publish_bytes
    from launch_batch8_pilot import copy_model_file
    root, job, bundle = Path(root), Path(job), Path(bundle)
    folder = root / 'checkpoints/oplora' / CONTRACT / 'mandatory_001250_3575a06121f0'
    manifest = json.loads((folder / 'MANIFEST.json').read_bytes())
    assert identity(manifest) == MANIFEST
    header = manifest['header']; contract = header['contract']
    assert identity(contract) == header['contract_identity'] == CONTRACT
    assert header['method'] == contract['method'] == 'oplora'
    assert header['successful_updates'] == 1250 and header['examples_seen'] == 40000
    assert contract['rank'] == 16 and contract['alpha'] == 32 and contract['protection_rank'] == 128
    protocol = json.loads((root / 'configs' / ('TRAINING_PROTOCOL_' + contract['protocol_identity'] + '.json')).read_bytes())
    assert identity(protocol) == contract['protocol_identity']
    source = root / 'manifests/CPU_ASSETS_READY.json'
    assets_sha = protocol['CPU_assets_manifest_sha256']
    assert digest(source) == assets_sha
    assets = json.loads(source.read_bytes())
    assert assets['protection_rank'] == 128 and assets['target_matrices'] == len(assets['bases']) == 196
    assert len({item['key'] for item in assets['bases']}) == 196
    assert digest(bundle / 'training_core.py') == assets['training_core_sha256']
    assert protocol['numerical_source_files']['training_core.py'] == assets['training_core_sha256']
    target = job / 'oplora_bases'; target.mkdir(exist_ok=True)
    publish_bytes(target / 'CPU_ASSETS_READY.json', source.read_bytes())
    for i, item in enumerate(assets['bases']):
        relative = 'bases/' + item['key'] + '.safetensors'; safe_relative(relative)
        progress(state, 'verify and copy OPLoRA basis %d/196' % (i + 1))
        copy_model_file(root / relative, target / relative, item['basis_sha256'], state)
    return assets_sha


def attach_adapters(model, weights, basis_root, assets_sha, state):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from archive_io import digest
    from baseline_runtime import progress
    from training_core import AdapterLinear, TARGETS
    basis_root, weights = Path(basis_root), Path(weights)
    assert digest(weights) == WEIGHTS
    assert digest(basis_root / 'CPU_ASSETS_READY.json') == assets_sha
    assets = json.loads((basis_root / 'CPU_ASSETS_READY.json').read_bytes())
    assert digest(Path(__file__).with_name('training_core.py')) == assets['training_core_sha256']
    bases = {item['key'][:-len('.weight')]: item for item in assets['bases']}
    assert len(bases) == 196 and all(item['key'].endswith('.weight') for item in bases.values())
    with zipfile.ZipFile(weights) as z, torch.no_grad():
        idx = json.loads(z.read('index.json'))
        assert idx['condition'] == 'oplora_final' and idx['update'] == 1250
        assert idx['checkpoint_sha256'] == CHECKPOINT and idx['dtype'] == 'float32'
        entries = idx['unique_parameters']
        assert len(entries) == 392 and set(entries) == {n+s for n in bases for s in ('.A', '.B')}
        count = 0
        for i, name in enumerate(sorted(bases)):
            progress(state, 'load OPLoRA projection %d/196' % (i + 1))
            assert name.rsplit('.', 1)[-1] in TARGETS
            factors = []
            for suffix in ('.A', '.B'):
                entry = entries[name+suffix]; raw = z.read(entry['member'])
                assert hashlib.sha256(raw).hexdigest() == entry['sha256']
                arr = np.load(io.BytesIO(raw), allow_pickle=False)
                assert arr.dtype == np.float32 and list(arr.shape) == entry['shape'] and np.isfinite(arr).all()
                factors.append(torch.from_numpy(arr)); count += arr.size
            A, B = factors; layer = model.get_submodule(name)
            assert A.shape == (16, layer.in_features) and B.shape == (layer.out_features, 16)
            item = bases[name]; path = basis_root / 'bases' / (item['key'] + '.safetensors')
            assert digest(path) == item['basis_sha256']
            saved = load_file(path, device='cpu'); U, V = saved['U'], saved['V']
            assert U.dtype == V.dtype == torch.float32
            assert U.shape == (layer.out_features, 128) and V.shape == (layer.in_features, 128)
            assert torch.isfinite(U).all() and torch.isfinite(V).all()
            parent, leaf = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), leaf, AdapterLinear(layer, A, B, U, V, alpha=32, dropout=.05))
        assert count == 10092544
    layers = [m for m in model.modules() if isinstance(m, AdapterLinear)]
    assert len(layers) == 196 and all(m.U is not None and m.V is not None for m in layers)
