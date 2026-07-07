#!/usr/bin/env python3
"""
Cross-platform launcher for the Path Planning pipeline (Windows / Linux / macOS).

It creates a local virtual environment, installs requirements.txt into it, and
then runs pipeline.py inside that environment, forwarding all CLI arguments.

Run it with your normal Python; it re-launches itself inside the venv:

    python run.py --config paths.txt --method kmeans --k 3 --suplement nitrogen
    python run.py --config paths.txt --method all --high-resolution
    python run.py --config paths.txt --suplement water --use-elevation

(On Windows you can also just double-click run.bat.)
"""
import os
import subprocess
import sys
import venv

HERE = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(HERE, "venv")


def venv_python(venv_dir):
    """Path to the python executable inside a virtual environment."""
    if os.name == "nt":  # Windows
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def main():
    py = venv_python(VENV_DIR)

    # 1. Create the venv on first run.
    if not os.path.exists(py):
        print("Creating virtual environment...")
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)

    # 2. Install dependencies (quiet; pip skips what's already satisfied).
    print("Installing requirements.txt...")
    subprocess.check_call([py, "-m", "pip", "install", "--upgrade", "pip", "-q"])
    subprocess.check_call([py, "-m", "pip", "install", "-r",
                           os.path.join(HERE, "requirements.txt"), "-q"])

    # 3. Run the end-to-end pipeline inside the venv, forwarding all arguments.
    print("Initiate pipeline:")
    print("----------------------------------------------------")
    cmd = [py, os.path.join(HERE, "run_pipeline.py")] + sys.argv[1:]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
