import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PLOT_DIR = Path(os.environ.get("THERMO_UTILS_PLOTS", "thermo-plots"))


def save(fig_factory, name):
    import matplotlib.pyplot as plt
    fig = fig_factory()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    path = PLOT_DIR / f"{safe}_{datetime.now():%Y%m%d-%H%M%S}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.resolve()


def open_file(path):
    if sys.platform.startswith("win"):
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
