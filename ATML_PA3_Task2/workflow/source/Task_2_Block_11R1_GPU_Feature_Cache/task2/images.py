"""Resumable original-JPEG preparation with immutable receipts and durable readback."""
from pathlib import Path
import hashlib
import json
import shutil
import threading
import time
import urllib.request
import uuid
from PIL import Image


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def inspect_jpeg(path, job):
    with Image.open(path) as image:
        image.load()
        if image.format != 'JPEG' or image.size != (job['width'], job['height']):
            raise ValueError(f"Source image format/dimensions differ for {job['source_split']}/{job['image_id']}")
        return {'width': image.width, 'height': image.height, 'mode': image.mode,
            'exif_orientation': int(image.getexif().get(274, 1))}


def immutable_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise IOError('Different existing artifact preserved: ' + str(path))
        return
    temporary = path.with_name(path.name + '.part_' + uuid.uuid4().hex[:8])
    try:
        temporary.write_bytes(content)
        if temporary.read_bytes() != content:
            raise IOError('Temporary save readback failed.')
        # Callers use a content hash in the name, or deterministic completion data.
        temporary.replace(path)
        if path.read_bytes() != content:
            raise IOError('Durable save readback failed.')
    finally:
        temporary.unlink(missing_ok=True)


def verified_copy(source, destination, expected_sha):
    source, destination = Path(source), Path(destination)
    if file_sha(source) != expected_sha:
        raise IOError('Copy source checksum differs.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing_sha = file_sha(destination)
        if existing_sha == expected_sha:
            return False
        # Preserve an interrupted or damaged cache entry before restoring its known original.
        backup = destination.with_name(destination.name + '.damaged_' + existing_sha[:16])
        if backup.exists():
            assert file_sha(backup) == existing_sha
        else:
            shutil.copyfile(destination, backup)
            assert file_sha(backup) == existing_sha
    temporary = destination.with_name(destination.name + '.part_' + uuid.uuid4().hex[:8])
    try:
        shutil.copyfile(source, temporary)
        if file_sha(temporary) != expected_sha:
            raise IOError('Temporary image copy checksum differs.')
        temporary.replace(destination)
        if file_sha(destination) != expected_sha:
            raise IOError('Durable image copy checksum differs.')
    finally:
        temporary.unlink(missing_ok=True)
    return True


def download_original(job, destination):
    request = urllib.request.Request(job['url'], headers={'User-Agent': 'ATML-PA3-original-image-preparation'})
    with urllib.request.urlopen(request, timeout=30) as response:
        if not response.geturl().startswith('https://'):
            raise IOError('Unexpected non-HTTPS image redirect.')
        expected_size = response.headers.get('Content-Length')
        with Path(destination).open('wb') as output:
            shutil.copyfileobj(response, output)
        if expected_size is not None and Path(destination).stat().st_size != int(expected_size):
            raise IOError('Incomplete image download.')


class ImagePreparer:
    def __init__(self, run_root, local_root, dataset_dir, dataset_identity, trusted_seeds=None, fetch=None):
        self.run_root = Path(run_root)
        self.local_root = Path(local_root)
        self.dataset_dir = Path(dataset_dir)
        self.dataset_identity = dataset_identity
        self.trusted_seeds = trusted_seeds or {}
        self.fetch = fetch or download_original
        self.stop = threading.Event()

    @staticmethod
    def validate_job(job):
        split, image_id = job['source_split'], job['image_id']
        if split not in ('train2014', 'val2014') or type(image_id) is not int or image_id < 0:
            raise ValueError('Invalid source image identity.')
        name = f'COCO_{split}_{image_id:012d}.jpg'
        if job['file_name'] != name or job['url'] != f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{name}':
            raise ValueError('Image job differs from its official COCO identity.')
        if type(job['width']) is not int or type(job['height']) is not int or min(job['width'], job['height']) <= 0:
            raise ValueError('Image dimensions must be positive integers.')

    def receipt_folder(self, job):
        return self.dataset_dir / 'inputs/image_receipts' / job['source_split'] / f"{job['image_id']:012d}"

    def load_receipt(self, job):
        paths = sorted(self.receipt_folder(job).glob('record_*.json'))
        if len(paths) > 1:
            raise IOError('Conflicting original-image receipts found. Preserve them and report the conflict.')
        if not paths:
            return None
        path = paths[0]
        assert path.name == 'record_' + file_sha(path) + '.json'
        record = json.loads(path.read_bytes())
        assert record['dataset_identity'] == self.dataset_identity
        for key in ('source_split', 'image_id', 'file_name', 'url'):
            assert record[key] == job[key], key
        relative = Path(record['durable_relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        return record

    def download(self, job, destination, expected_sha=None):
        for attempt in range(3):
            if self.stop.is_set():
                raise InterruptedError('Image preparation stopped.')
            try:
                self.fetch(job, destination)
                metadata = inspect_jpeg(destination, job)
                observed_sha = file_sha(destination)
                if expected_sha is not None and observed_sha != expected_sha:
                    raise IOError('Downloaded original differs from its saved source checksum.')
                return observed_sha, metadata
            except Exception:
                if attempt == 2 or self.stop.is_set():
                    raise
                time.sleep(attempt + 1)

    def prepare(self, job):
        self.validate_job(job)
        if self.stop.is_set():
            raise InterruptedError('Image preparation stopped.')
        local = self.local_root / 'images' / job['source_split'] / job['file_name']
        local.parent.mkdir(parents=True, exist_ok=True)
        existing = self.load_receipt(job)
        legacy_path = self.run_root / 'manifests/images' / job['source_split'] / f"{job['image_id']:012d}.json"
        legacy = json.loads(legacy_path.read_bytes()) if legacy_path.exists() else None
        expected_sha = existing['sha256'] if existing is not None else None
        if legacy is not None:
            assert legacy['url'] == job['url'] and legacy['image_id'] == job['image_id']
            if expected_sha is not None:
                assert expected_sha == legacy['sha256']
            expected_sha = legacy['sha256']
        candidates = []
        if existing is not None:
            candidates.append(self.run_root / existing['durable_relative_path'])
        if legacy is not None:
            candidates.append(self.run_root / 'raw_images' / job['source_split'] / job['file_name'])
        seed = self.trusted_seeds.get((job['source_split'], job['image_id']))
        if seed is not None:
            if expected_sha is not None:
                assert seed['sha256'] == expected_sha
            expected_sha = seed['sha256']
            candidates.append(Path(seed['path']))
        if expected_sha is not None:
            candidates.append(local)
        source = next((path for path in candidates if path.is_file() and file_sha(path) == expected_sha), None)
        temporary = local.with_name(local.name + '.download_' + uuid.uuid4().hex[:8])
        try:
            activity = 'reused'
            if source is None:
                observed_sha, metadata = self.download(job, temporary, expected_sha)
                expected_sha, source = observed_sha, temporary
                activity = 'downloaded'
            else:
                metadata = inspect_jpeg(source, job)
            if existing is not None:
                assert metadata == existing['decoded_metadata']
                durable = self.run_root / existing['durable_relative_path']
            elif legacy is not None:
                assert all(metadata[key] == value for key, value in legacy['decoded_metadata'].items())
                durable = self.run_root / 'raw_images' / job['source_split'] / job['file_name']
            else:
                durable = self.run_root / 'raw_images/verified' / job['source_split'] / expected_sha[:2] / (expected_sha + '.jpg')
            verified_copy(source, durable, expected_sha)
            verified_copy(durable, local, expected_sha)
            assert inspect_jpeg(local, job) == metadata
            record = {'schema': 'atml-pa3-task2-original-image-v1', 'dataset_identity': self.dataset_identity,
                'source_split': job['source_split'], 'image_id': job['image_id'], 'file_name': job['file_name'],
                'url': job['url'], 'sha256': expected_sha, 'bytes': local.stat().st_size,
                'decoded_metadata': metadata, 'durable_relative_path': str(durable.relative_to(self.run_root))}
            if existing is not None:
                assert record == existing
            content = encoded(record)
            receipt_sha = hashlib.sha256(content).hexdigest()
            immutable_bytes(self.receipt_folder(job) / ('record_' + receipt_sha + '.json'), content)
            assert self.load_receipt(job) == record
            return record, activity
        finally:
            temporary.unlink(missing_ok=True)
