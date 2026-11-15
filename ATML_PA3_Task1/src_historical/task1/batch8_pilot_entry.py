"""Add visible phase messages without changing the CPU-verified pilot source."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch8_pilot

original_boundary = batch8_pilot.boundary


def reported_boundary(config, phase):
    print('Pilot phase: ' + phase, flush=True)
    return original_boundary(config, phase)


batch8_pilot.boundary = reported_boundary
if __name__ == '__main__': sys.exit(batch8_pilot.main(sys.argv[1]))
