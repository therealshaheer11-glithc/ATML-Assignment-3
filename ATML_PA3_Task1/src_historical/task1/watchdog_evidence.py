"""Best-effort Drive evidence must never block the independent release timer."""
import json
import threading
import time
import uuid
from pathlib import Path


class Evidence:
    def __init__(self, directory=None, run_id=None):
        self.directory = Path(directory) if directory else None
        self.run_id = run_id
        if run_id and ('/' in run_id or not run_id.startswith('task1_')):
            raise ValueError('Unsafe watchdog evidence name')
        self.pending = None
        self.last_attempt = 0.

    def _write(self, suffix, value):
        try:
            # Do not create directories before Drive is mounted.
            if not self.directory or not self.directory.is_dir(): return
            path = self.directory / (self.run_id + suffix)
            temporary = path.with_name(path.name + '.pending_' + uuid.uuid4().hex)
            temporary.write_text(json.dumps(value, indent=2) + '\n')
            temporary.replace(path)
        except Exception:
            pass

    def progress(self, state):
        now = time.monotonic()
        if not self.directory or now - self.last_attempt < 30:
            return
        if self.pending is not None and self.pending.is_alive(): return
        self.last_attempt = now
        snapshot = {'watchdog_observed_wall': time.time(), 'worker_state': dict(state),
                    'physical_release_verified': False}
        self.pending = threading.Thread(target=self._write, args=('_watchdog_progress.json', snapshot), daemon=True)
        self.pending.start()

    def terminal(self, suffix, event, wait_seconds=.25):
        if not self.directory: return
        thread = threading.Thread(target=self._write, args=(suffix, dict(event)), daemon=True)
        thread.start(); thread.join(wait_seconds)


def cpu_checks(directory):
    directory = Path(directory); directory.mkdir(parents=True)
    writer = Evidence(directory, 'task1_fixture')
    writer.progress({'progress_wall': 123., 'phase': 'fixture'})
    writer.pending.join(2)
    saved = json.loads((directory / 'task1_fixture_watchdog_progress.json').read_text())
    assert saved['worker_state']['progress_wall'] == 123.
    writer.terminal('_release_intent.json', {'reason': 'fixture'})
    assert json.loads((directory / 'task1_fixture_release_intent.json').read_text())['reason'] == 'fixture'
    gate = threading.Event()
    try:
        writer._write = lambda *args: gate.wait(5)
        start = time.monotonic()
        writer.terminal('_blocked.json', {}, wait_seconds=.03)
        assert time.monotonic() - start < .5
    finally: gate.set()
    missing = directory / 'not_mounted'
    Evidence(missing, 'task1_unmounted').terminal('_x.json', {})
    assert not missing.exists()
    return {'snapshot_preserves_actual_worker_progress_time': True,
            'release_intent_written_before_release': True,
            'blocked_evidence_write_is_bounded': True,
            'unmounted_Drive_path_is_not_created': True}
