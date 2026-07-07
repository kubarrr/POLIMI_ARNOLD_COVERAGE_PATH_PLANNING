from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def find_project_root(start: Path) -> Path:
    """Find the project root by walking upward until the expected folders exist."""
    for candidate in [start, *start.parents]:
        if (candidate / "scripts").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("Could not locate project root containing both 'scripts' and 'data'.")


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def validate_inputs(project_root: Path, config: dict) -> None:
    missing = []
    for key, value in config["inputs"].items():
        path = resolve_path(project_root, value)
        if not path.exists():
            missing.append((key, path))
    if missing:
        lines = ["Missing required input files:"]
        lines.extend(f"  - {key}: {path}" for key, path in missing)
        raise FileNotFoundError("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the reproducible side-wise P80 CPP pipeline."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.json"),
        help="Path to the JSON configuration file.",
    )
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve())
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = load_config(config_path)
    validate_inputs(project_root, config)

    local_code_dir = Path(__file__).resolve().parent
    if str(local_code_dir) not in sys.path:
        sys.path.insert(0, str(local_code_dir))

    from sidewise_p80_cpp_pipeline import run

    metrics = run(project_root, config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
