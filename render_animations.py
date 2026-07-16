r"""Render the documentation Manim animations from Windows.

Run this from a native Windows terminal after activating the Manim conda env:

    cd /d C:\Users\p2321038\Documents\GitHub\DelaDect
    call C:\ProgramData\miniforge3\Scripts\activate.bat my-manim-environment
    python render_animations.py

Useful variants:

    python render_animations.py --quality l --preview
    python render_animations.py --only division
    python render_animations.py --only shift rbf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANIMATIONS = ROOT / "animations"

QUALITY_FLAGS = {
    "l": "-ql",
    "m": "-qm",
    "h": "-qh",
    "k": "-qk",
}

PREP_SCRIPTS = [
    ANIMATIONS / "prepare_l1_shift_assets.py",
    ANIMATIONS / "prepare_division_assets.py",
]

SCENE_GROUPS = {
    "shift": ("scenes_l1_shift_correction.py", ["L1ShiftCorrection"]),
    "division": ("scenes_frame_division.py", ["FrameDivisionRatio", "RollingMedianReferenceRatio"]),
    "rbf": ("scenes_rbf_interpolator.py", ["RBFInterpolatorWarp"]),
}


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("\n>", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def prepare_assets() -> None:
    for script in PREP_SCRIPTS:
        run([sys.executable, str(script)])


def render_group(group: str, quality: str, preview: bool) -> None:
    scene_file, scene_names = SCENE_GROUPS[group]
    command = [sys.executable, "-m", "manim"]
    if preview:
        command.append("-p")
    command.extend([QUALITY_FLAGS[quality], "--disable_caching", scene_file, *scene_names])
    run(command, cwd=ANIMATIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render DelaDect Manim documentation animations.")
    parser.add_argument(
        "--quality",
        choices=QUALITY_FLAGS,
        default="h",
        help="Manim quality: l=low, m=medium, h=high, k=4k. Default: h.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[*SCENE_GROUPS.keys(), "all"],
        default=["all"],
        help="Animation groups to render. Default: all.",
    )
    parser.add_argument("--preview", action="store_true", help="Open each rendered video after rendering.")
    parser.add_argument("--skip-assets", action="store_true", help="Skip asset preparation scripts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = list(SCENE_GROUPS) if "all" in args.only else args.only

    if not args.skip_assets:
        prepare_assets()

    for group in groups:
        render_group(group, args.quality, args.preview)

    print("\nRendered videos are in:")
    print(ANIMATIONS / "media" / "videos")


if __name__ == "__main__":
    main()
