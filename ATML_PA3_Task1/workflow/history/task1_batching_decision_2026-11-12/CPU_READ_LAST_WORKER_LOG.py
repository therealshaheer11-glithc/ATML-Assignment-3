# Run in the CURRENT CPU runtime; Drive is already mounted by Block 12.
# Reads existing evidence only. No model loading, inference, or result changes.
import json, os
from pathlib import Path

assert os.environ.get('COLAB_GPU', '0').strip().lower() in ('', '0', 'false', 'none')
assert not list(Path('/dev').glob('nvidia[0-9]*')), 'Keep CPU selected.'
ROOT = Path('/content/drive/MyDrive/ATML-PA-3/Task1')
IMPLEMENTATION = 'd5c5b51f6107bd08adb362f878b528d6b1c92738e07e432405ddfeb7fc3ff56e'
sessions = [json.loads(p.read_text()) for p in (ROOT / 'logs').glob('task1_baseline_*_session_started.json')]
sessions = [s for s in sessions if s.get('implementation_identity') == IMPLEMENTATION]
assert sessions, 'Expected session not found; keep CPU selected and send this message.'
session = max(sessions, key=lambda s: s['started_wall'])
run_id = session['run_id']
print('Reading saved evidence for:', run_id, flush=True)
for suffix in ('_worker_console.txt', '_bootstrap_error.txt', '_result.json', '_session_final.json'):
    path = ROOT / 'logs' / (run_id + suffix)
    print('\nFILE:', path.name, flush=True)
    print(path.read_text()[-12000:] if path.exists() else 'NOT FOUND', flush=True)
print('\nKeep CPU selected and send this output.', flush=True)
