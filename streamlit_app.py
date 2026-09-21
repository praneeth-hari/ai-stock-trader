import sys
from pathlib import Path

# Add project root to sys.path so dashboard can import config and src
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dashboard.app
