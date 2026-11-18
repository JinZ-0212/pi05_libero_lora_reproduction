"""Runtime layout shared by the portable copies of the recorded experiment scripts."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get('PI05_WORKDIR', str(REPO / 'runtime'))).expanduser().resolve()
PILOT_ID = 'pi05-object-lora-20261118'
CONTINUATION_ID = 'pi05-object-lora-2000-20261118'

def experiment(phase):
    return ROOT / 'experiments' / (PILOT_ID if phase == 'pilot500' else CONTINUATION_ID)
