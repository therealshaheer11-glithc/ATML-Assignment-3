"""Isolated entry; explicit local path only. No model/library import before handshake."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gpu_worker import main
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bundle',required=True);p.add_argument('--config',required=True);a=p.parse_args();main(Path(a.root),Path(a.bundle),Path(a.config))
