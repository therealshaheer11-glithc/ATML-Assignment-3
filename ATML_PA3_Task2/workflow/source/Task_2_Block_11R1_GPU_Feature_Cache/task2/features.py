"""Verified feature shards and a final-norm-checked frozen SigLIP forward path."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import json
import os
import shutil
import signal
import tarfile
import time
import uuid
import numpy as np
import torch
from .images import encoded, file_sha, immutable_bytes, verified_copy
from .model_assets import safe_name


@contextmanager
def bounded(seconds, label):
    """Bound handled filesystem operations in the Linux worker's main thread."""
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    def expired(signum, frame):
        raise TimeoutError(label + ' exceeded its time limit.')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, previous_timer[0] - (time.monotonic()-started)), previous_timer[1])


def stage_file(source, destination, expected_sha):
    """Read Drive once, hashing while copying, then verify the local readback."""
    source, destination = Path(source), Path(destination)
    if destination.is_file() and file_sha(destination) == expected_sha:
        return 'reused'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.part_' + uuid.uuid4().hex)
    try:
        digest = hashlib.sha256()
        with source.open('rb') as incoming, temporary.open('wb') as outgoing:
            for block in iter(lambda: incoming.read(8*1024*1024), b''):
                outgoing.write(block)
                digest.update(block)
        if digest.hexdigest() != expected_sha or file_sha(temporary) != expected_sha:
            raise IOError('Staged input checksum differs from its verified receipt.')
        if destination.exists():
            old_sha = file_sha(destination)
            backup = destination.with_name(destination.name + '.damaged_' + old_sha[:16])
            if not backup.exists():
                shutil.copyfile(destination, backup)
            assert file_sha(backup) == old_sha
        temporary.replace(destination)
        assert file_sha(destination) == expected_sha
        return 'staged'
    finally:
        temporary.unlink(missing_ok=True)


def unpack_model(path, folder, records):
    """Unpack only the explicitly named, hashed original model files."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    expected = [safe_name(row['file_name']) for row in records]
    assert len(expected) == len(set(expected))
    with tarfile.open(path, 'r:') as archive:
        members = archive.getmembers()
        assert [m.name for m in members] == expected and all(m.isfile() for m in members)
        for member, row in zip(members, records):
            assert member.size == row['bytes']
            destination = folder / member.name
            if destination.is_file() and file_sha(destination) == row['sha256']:
                continue
            temporary = destination.with_name(destination.name + '.part_' + uuid.uuid4().hex)
            try:
                digest = hashlib.sha256()
                with archive.extractfile(member) as incoming, temporary.open('wb') as outgoing:
                    for block in iter(lambda: incoming.read(8*1024*1024), b''):
                        outgoing.write(block)
                        digest.update(block)
                assert digest.hexdigest() == row['sha256'] and file_sha(temporary) == row['sha256']
                verified_copy(temporary, destination, row['sha256'])
            finally:
                temporary.unlink(missing_ok=True)


class FinalNormExtractor:
    def __init__(self, model):
        self.model = model
        modules = [(name, module) for name, module in model.named_modules() if name.split('.')[-1] == 'post_layernorm']
        if len(modules) != 1:
            raise ValueError('Expected exactly one final vision post-layer norm module.')
        self.norm_name, norm = modules[0]
        self.outputs = []
        self.hook = norm.register_forward_hook(lambda module, inputs, output: self.outputs.append(output))

    def __call__(self, pixels):
        self.outputs.clear()
        try:
            if self.model.training or any(p.requires_grad for p in self.model.parameters()):
                raise ValueError('Vision encoder must remain frozen and in evaluation mode.')
            with torch.inference_mode():
                features = self.model(pixel_values=pixels, interpolate_pos_encoding=False, return_dict=True).last_hidden_state
            if len(self.outputs) != 1 or not torch.equal(features, self.outputs[0]):
                raise ValueError('Returned patches must equal one application of the final layer norm.')
            if tuple(features.shape) != (pixels.shape[0], 196, 768) or features.dtype != torch.float16:
                raise ValueError('Vision output must have FP16 shape batch x 196 x 768.')
            if features.requires_grad or not torch.isfinite(features).all().item():
                raise ValueError('Vision output is nonfinite or attached to a gradient graph.')
            return features
        finally:
            self.outputs.clear()

    def close(self):
        self.hook.remove()


def verify_features(path, count):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        if values.shape != (count, 196, 768) or values.dtype != np.dtype('<f2') or not values.flags.c_contiguous:
            raise ValueError('Feature file shape, order or dtype differs.')
        for start in range(0, count, 32):
            if not np.isfinite(values[start:start+32]).all():
                raise ValueError('Feature file contains nonfinite values.')
    finally:
        del values


class FeatureStore:
    def __init__(self, dataset_dir, local_root, recipe):
        self.recipe = recipe
        self.recipe_sha = hashlib.sha256(encoded(recipe)).hexdigest()
        self.dataset_dir = Path(dataset_dir)
        self.durable_dir = self.dataset_dir / 'features' / ('siglip196_' + self.recipe_sha[:16])
        self.local_dir = Path(local_root) / 'features' / self.recipe_sha[:16]

    def receipt_path(self, index):
        if type(index) is not int or index < 0:
            raise ValueError('Invalid feature shard index.')
        return self.durable_dir / f'shard_{index:04d}.json'

    def previous(self, index, images):
        path = self.receipt_path(index)
        if not path.exists():
            return None
        receipt = json.loads(path.read_bytes())
        assert receipt['recipe_sha256'] == self.recipe_sha and receipt['index'] == index
        assert receipt['images'] == images and receipt['image_count'] == len(images)
        assert receipt['shape'] == [len(images),196,768] and receipt['dtype'] == 'float16'
        relative = Path(receipt['feature_relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        return receipt

    def reuse(self, index, images):
        previous = self.previous(index, images)
        if previous is None:
            return None
        durable = self.dataset_dir / previous['feature_relative_path']
        local = self.local_dir / durable.name
        if durable.is_file() and file_sha(durable) == previous['feature_sha256']:
            verify_features(durable, len(images))
            return previous
        if local.is_file() and file_sha(local) == previous['feature_sha256']:
            verify_features(local, len(images))
            verified_copy(local, durable, previous['feature_sha256'])
            return previous
        return None

    def save(self, index, images, values):
        if not images or len(images) > 256:
            raise ValueError('Feature shard must contain between one and 256 images.')
        if len({(r['source_split'],r['image_id']) for r in images}) != len(images):
            raise ValueError('Feature shard contains duplicate images.')
        if any(r['dataset_identity'] != self.recipe['dataset_identity'] for r in images):
            raise ValueError('Feature inputs belong to a different dataset.')
        if values.dtype != np.dtype('<f2') or values.shape != (len(images),196,768) or not np.isfinite(values).all():
            raise ValueError('Only finite FP16 patch arrays with the expected shape can be saved.')
        previous = self.previous(index, images)
        self.local_dir.mkdir(parents=True,exist_ok=True)
        temporary = self.local_dir / ('features_' + uuid.uuid4().hex + '.npy')
        try:
            with temporary.open('wb') as output:
                np.lib.format.write_array(output,np.ascontiguousarray(values),version=(2,0),allow_pickle=False)
            verify_features(temporary,len(images))
            digest = file_sha(temporary)
            if previous is not None:
                assert digest == previous['feature_sha256'], 'Recomputed features differ from the recorded shard.'
            name = f'shard_{index:04d}_{digest[:16]}.npy'
            local, durable = self.local_dir/name, self.durable_dir/name
            verified_copy(temporary,local,digest)
            verified_copy(local,durable,digest)
            verify_features(durable,len(images))
            receipt = {'schema':'atml-pa3-task2-feature-shard-v1','recipe_sha256':self.recipe_sha,
                'index':index,'dataset_identity':self.recipe['dataset_identity'],'image_count':len(images),
                'images':images,'shape':[len(images),196,768],'dtype':'float16',
                'feature_relative_path':str(durable.relative_to(self.dataset_dir)),
                'feature_sha256':digest,'feature_bytes':durable.stat().st_size}
            immutable_bytes(self.receipt_path(index),encoded(receipt))
            return receipt
        finally:
            temporary.unlink(missing_ok=True)
