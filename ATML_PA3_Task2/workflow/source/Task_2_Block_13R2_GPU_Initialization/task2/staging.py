"""Pack verified original JPEG bytes into deterministic archives for GPU reads."""
from pathlib import Path
import hashlib
import json
import tarfile
import uuid
from .images import encoded, file_sha, immutable_bytes, verified_copy


def member_name(record):
    split, image_id = record['source_split'], record['image_id']
    if split not in ('train2014', 'val2014') or type(image_id) is not int or image_id < 0:
        raise ValueError('Invalid source image identity.')
    if record['file_name'] != f'COCO_{split}_{image_id:012d}.jpg':
        raise ValueError('Invalid source image filename.')
    name = record['source_split'] + '/' + record['file_name']
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Invalid image member path.')
    return name


def verify_archive(path, records):
    expected = [member_name(record) for record in records]
    if len(expected) != len(set(expected)):
        raise ValueError('Archive input contains duplicate image identities.')
    with tarfile.open(path, 'r:') as archive:
        members = archive.getmembers()
        if [member.name for member in members] != expected or not all(member.isfile() for member in members):
            raise ValueError('Archive members differ from the frozen image order.')
        for member, record in zip(members, records):
            assert member.size == record['bytes']
            with archive.extractfile(member) as stream:
                digest = hashlib.sha256()
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            assert digest.hexdigest() == record['sha256'], member.name


def image_bytes_from_archive(path, records):
    """Yield verified JPEG bytes in frozen order without extracting filesystem paths."""
    with tarfile.open(path, 'r:') as archive:
        expected = [member_name(record) for record in records]
        assert [member.name for member in archive.getmembers()] == expected
        for record in records:
            member = archive.getmember(member_name(record))
            assert member.isfile() and member.size == record['bytes']
            with archive.extractfile(member) as stream:
                content = stream.read()
            assert hashlib.sha256(content).hexdigest() == record['sha256']
            yield record, content


class InputArchiveStore:
    def __init__(self, run_root, local_root, dataset_dir, recipe):
        self.run_root = Path(run_root)
        self.local_root = Path(local_root)
        self.dataset_dir = Path(dataset_dir)
        self.recipe = recipe
        self.recipe_sha = hashlib.sha256(encoded(recipe)).hexdigest()
        self.durable_dir = self.dataset_dir / 'inputs/image_shards' / self.recipe_sha[:16]
        self.local_dir = self.local_root / 'image_shards' / self.recipe_sha[:16]

    def source_path(self, record):
        local = self.local_root / 'images' / record['source_split'] / record['file_name']
        relative = Path(record['durable_relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        durable = self.run_root / relative
        if local.is_file() and file_sha(local) == record['sha256']:
            return local
        if not durable.is_file() or file_sha(durable) != record['sha256']:
            raise IOError('Original image differs from its completed preparation receipt.')
        verified_copy(durable, local, record['sha256'])
        return local

    def pack(self, index, records):
        if type(index) is not int or index < 0 or not records:
            raise ValueError('A nonempty archive and nonnegative index are required.')
        if len(records) > self.recipe['images_per_shard']:
            raise ValueError('Archive exceeds its recorded image limit.')
        if any(r['dataset_identity'] != self.recipe['dataset_identity'] for r in records):
            raise ValueError('Archive input belongs to a different dataset.')
        for record in records:
            member_name(record)
        self.local_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = self.durable_dir / f'shard_{index:04d}.json'
        previous = json.loads(receipt_path.read_bytes()) if receipt_path.exists() else None
        keys = [(r['source_split'], r['image_id']) for r in records]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate source images in one archive.')
        if previous is not None:
            assert previous['recipe_sha256'] == self.recipe_sha and previous['images'] == records
            relative = Path(previous['tar_relative_path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            durable = self.dataset_dir / relative
            local = self.local_dir / durable.name
            if durable.is_file() and file_sha(durable) == previous['tar_sha256']:
                verify_archive(durable, records)
                return previous, 'reused'
            if local.is_file() and file_sha(local) == previous['tar_sha256']:
                verify_archive(local, records)
                verified_copy(local, durable, previous['tar_sha256'])
                return previous, 'restored'
        temporary = self.local_dir / ('packing_' + uuid.uuid4().hex + '.tar')
        try:
            with tarfile.open(temporary, 'w', format=tarfile.USTAR_FORMAT) as archive:
                for record in records:
                    source = self.source_path(record)
                    info = tarfile.TarInfo(member_name(record))
                    info.size = record['bytes']
                    info.mtime = 0
                    info.mode = 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = ''
                    with source.open('rb') as stream:
                        archive.addfile(info, stream)
            verify_archive(temporary, records)
            digest = file_sha(temporary)
            if previous is not None:
                assert digest == previous['tar_sha256'], 'Rebuilt source archive differs from its recorded bytes.'
            name = f'shard_{index:04d}_{digest[:16]}.tar'
            local, durable = self.local_dir / name, self.durable_dir / name
            verified_copy(temporary, local, digest)
            verified_copy(local, durable, digest)
            receipt = {'schema': 'atml-pa3-task2-input-shard-v1', 'recipe_sha256': self.recipe_sha,
                'dataset_identity': self.recipe['dataset_identity'], 'index': index,
                'tar_relative_path': str(durable.relative_to(self.dataset_dir)),
                'tar_sha256': digest, 'tar_bytes': durable.stat().st_size, 'image_count': len(records), 'images': records}
            immutable_bytes(receipt_path, encoded(receipt))
            assert json.loads(receipt_path.read_bytes()) == receipt
            return receipt, 'packed'
        finally:
            temporary.unlink(missing_ok=True)
