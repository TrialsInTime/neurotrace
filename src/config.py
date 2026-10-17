import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]                 # .../neurotrace
DATA_ROOT = Path(os.environ.get("NEUROTRACE_DATA", PROJECT.parent / "dataset"))
CACHE = PROJECT / "cache"       # features/epochs, gitignored
RESULTS = PROJECT / "results"   # small CSVs and figures, committed
CACHE.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)