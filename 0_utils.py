"""
0_utils.py — Path configuration and I/O helpers for BY04 APT notebooks.

Run this cell first.  You will be prompted to enter the folder where all
output files (pickle files and figures) should be saved and loaded from.
"""

import pickle
import pathlib
import matplotlib

# ── Plot style ────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    # Font
    'font.family':           'sans-serif',
    'font.sans-serif':       ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size':             11,
    'axes.titlesize':        11,
    'axes.labelsize':        11,
    'xtick.labelsize':       10,
    'ytick.labelsize':       10,
    'legend.fontsize':       10,
    # Spines — remove top and right
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    # Grid
    'axes.grid':             True,
    'grid.color':            '#e0e0e0',
    'grid.linewidth':        0.6,
    'grid.linestyle':        '-',
    # Ticks
    'xtick.direction':       'out',
    'ytick.direction':       'out',
    'xtick.major.size':      4,
    'ytick.major.size':      4,
    # Figure
    'figure.facecolor':      'white',
    'axes.facecolor':        'white',
    'figure.dpi':            120,
})

# ── Colour palette (blue / dark-green) ────────────────────────────────────────
BLUE       = '#1F5FA6'   # deep blue
DARKGREEN  = '#1A6B3C'   # dark forest green

# Prompt the user for the repo/project directory.
_raw = input("Enter the project directory path (e.g. /Users/you/BY04_APT): ").strip()
PROJECT_DIR = pathlib.Path(_raw).expanduser().resolve()
BASE_DIR = PROJECT_DIR / "output"
BASE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output directory set to: {BASE_DIR}")


def data_path(filename: str) -> pathlib.Path:
    """Return the full path for a file inside BASE_DIR."""
    return BASE_DIR / filename


def save_pkl(obj, filename: str) -> None:
    """Pickle `obj` to BASE_DIR/filename."""
    path = data_path(filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"Saved  {path}")


def load_pkl(filename: str):
    """Load and return a pickled object from BASE_DIR/filename."""
    path = data_path(filename)
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"Loaded {path}")
    return obj


def save_fig(fig, stem: str, formats=("png",), dpi: int = 150) -> None:
    """Save a matplotlib figure to BASE_DIR/<stem>.<ext> for each format."""
    for fmt in formats:
        path = data_path(f"{stem}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved  {path}")


# Register this module in sys.modules so subsequent scripts always get the
# latest version — re-running this cell refreshes the cache automatically.
import sys as _sys, types as _types
_mod = _types.ModuleType('_by04_utils')
for _k, _v in list(globals().items()):
    if not _k.startswith('__'):
        setattr(_mod, _k, _v)
_sys.modules['_by04_utils'] = _mod
