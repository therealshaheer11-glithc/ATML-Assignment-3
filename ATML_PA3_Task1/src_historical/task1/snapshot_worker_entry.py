"""Storage-only wrapper around the byte-identical original baseline worker."""
import json
import sys
from pathlib import Path


def main(config_path, core, snapshot, transport_identity):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(core).resolve()))
    from archive_io import digest
    from evaluation_records import identity
    from snapshot_transport import configure_worker
    import baseline_v2_worker as worker
    config = json.loads(Path(config_path).read_text())
    index = json.loads((Path(snapshot) / 'SNAPSHOT_INDEX.json').read_text())
    proof = json.loads((Path(config['job']) / 'transport_manifest.json').read_text())
    assert identity(proof['files']) == transport_identity
    for name, expected in proof['files'].items():
        assert digest(Path(__file__).parent / name) == expected
    original = json.loads((Path(core).parent / 'CORE_FILES.json').read_text())
    for name, expected in original.items(): assert digest(Path(core) / name) == expected
    assert identity({n: h for n, h in original.items() if n.endswith('.py')}) == config['implementation_identity']
    stats = configure_worker(worker, config, snapshot, index, transport_identity)
    print('Packed resume active: earlier scored answers are verified from local disk; new work saves to Drive.', flush=True)
    return worker.main(config_path)


if __name__ == '__main__': sys.exit(main(*sys.argv[1:]))
