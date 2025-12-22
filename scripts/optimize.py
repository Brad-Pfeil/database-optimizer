"""Script to trigger optimization."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database_optimiser.cli.main import cli

if __name__ == "__main__":
    cli(["optimize"])
