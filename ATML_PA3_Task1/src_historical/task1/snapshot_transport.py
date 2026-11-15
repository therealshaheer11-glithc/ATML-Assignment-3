"""Verified snapshot transport; preserve the original numerical/cache identity."""
import json
import stat
import time
import zipfile
from pathlib import Path

from archive_io import digest, safe_relative
from baseline_runtime import progress, publish_bytes
from evaluation_records import RecordStore, identity, publish
from launch_batch8_pilot import copy_model_file


def restore_snapshot(root, job, ready, state):
    root, job = Path(root), Path(job)
    source = root / str(safe_relative(ready['archive_path']))
    archive_path = job / 'verified_resume.zip'
    progress(state, 'restore single packed answer snapshot')
    copy_model_file(source, archive_path, ready['archive_sha256'], state)
    assert archive_path.stat().st_size == ready['archive_bytes']
    destination = job / 'snapshot'; destination.mkdir()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)) and 'SNAPSHOT_INDEX.json' in names
        info = archive.getinfo('SNAPSHOT_INDEX.json'); assert info.file_size <= 32 * 2**20
        index_data = archive.read(info)
        import hashlib
        assert hashlib.sha256(index_data).hexdigest() == ready['index_sha256']
        index = json.loads(index_data); assert identity(index) == ready['snapshot_identity']
        assert index['contract'] == ready['contract'] and index['counts'] == ready['counts']
        assert index['numerical_implementation_identity'] == ready['numerical_implementation_identity']
        assert set(names) == set(index['files']) | {'SNAPSHOT_INDEX.json'}
        last = time.monotonic()
        for name, entry in index['files'].items():
            safe_relative(name)
            assert name.startswith('results_batch8_V2/original/') and name.endswith('.json')
            info = archive.getinfo(name)
            assert stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG)
            assert info.file_size == entry['bytes'] and info.file_size <= 16 * 2**20
            data = archive.read(info)
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
            path = destination / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            assert digest(path) == entry['sha256']
            if time.monotonic() - last > 2:
                progress(state, 'verify local packed answer snapshot'); last = time.monotonic()
        (destination / 'SNAPSHOT_INDEX.json').write_bytes(index_data)
    progress(state, 'packed answers restored and verified on local disk')
    return destination, index


class PackedRecordStore(RecordStore):
    """Indexed old answers read locally; new answers retain original Drive saves."""
    def __init__(self, *args, snapshot, index, stats, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshot, self.index, self.stats = Path(snapshot), index, stats
        assert self.contract == index['contract']
        self.cached = RecordStore(self.snapshot / 'results_batch8_V2', self.contract['condition'],
            self.contract['model_identity'], self.contract['protocol_identity'])

    def key(self, task, source_index):
        self.path(task, source_index)  # Preserve canonical task/ID validation.
        return f'results_batch8_V2/{self.contract["condition"]}/{task}/{source_index:05}.json'

    def get(self, task, source_index, request):
        key = self.key(task, source_index)
        if key in self.index['files']:
            # Never fall back to expensive remote reads for an indexed answer.
            path = self.snapshot / key
            assert path.is_file() and digest(path) == self.index['files'][key]['sha256']
            value = self.cached.get(task, source_index, request)
            assert value is not None
            self.stats['snapshot_record_reads'] += 1
            return value
        self.stats['remote_record_checks_for_unindexed_IDs'] += 1
        return super().get(task, source_index, request)

    def put(self, task, source_index, request, response, scores, metadata):
        assert self.key(task, source_index) not in self.index['files'], 'Refusing to overwrite an indexed saved answer'
        value = super().put(task, source_index, request, response, scores, metadata)
        self.stats['new_records_saved_with_original_Drive_readback'] += 1
        return value


def configure_worker(worker, config, snapshot, index, transport_identity):
    import functools
    stats = {'snapshot_record_reads': 0, 'remote_record_checks_for_unindexed_IDs': 0,
             'new_records_saved_with_original_Drive_readback': 0}
    assert index['contract']['model_identity'] == config['model_identity']
    assert index['contract']['protocol_identity'] == config['protocol_identity']
    assert index['numerical_implementation_identity'] == config['implementation_identity']
    worker.RecordStore = functools.partial(PackedRecordStore, snapshot=snapshot, index=index, stats=stats)
    original_publish = worker.publish
    def publish_with_transport(path, value):
        if Path(path) == Path(config['result']):
            receipt = {'run_id': config['run_id'], 'snapshot_identity': identity(index),
                'transport_implementation_identity': transport_identity,
                'unchanged_numerical_implementation_identity': config['implementation_identity'],
                'record_IO': dict(stats), 'experiment_settings_changed': False}
            receipt_path = Path(config['root']) / 'logs' / (config['run_id'] + '_transport_result.json')
            original_publish(receipt_path, receipt)
            value['packed_resume_transport'] = receipt
        return original_publish(path, value)
    worker.publish = publish_with_transport
    return stats
