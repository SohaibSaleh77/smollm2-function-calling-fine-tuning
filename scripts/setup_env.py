"""Environment setup (replaces the notebook's `!pip ...` cells).

`!pip install ...` is IPython/Colab magic and is a SyntaxError inside a
normal .py file, so dependency installation lives here and uses subprocess.
Dependencies are declared in setup.py, so this simply does an editable
install and then removes torchao (it can conflict with bitsandbytes/TRL
imports in some environments).

Run from anywhere:
    python scripts/setup_env.py
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd):
    cmd = [str(c) for c in cmd]
    print(">>", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    # `!pip install -q transformers peft trl ...`  ->  editable install via setup.py
    run([sys.executable, "-m", "pip", "install", "-q", "-e", "."])
    # `!pip uninstall -y torchao`
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])


if __name__ == "__main__":
    main()
