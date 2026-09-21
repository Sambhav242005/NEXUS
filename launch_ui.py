"""NEXUS UI launcher.

Usage:
    python launch_ui.py
    python launch_ui.py --virtual-mouse
    python launch_ui.py --workdir C:/projects --virtual-mouse
"""
import argparse
import sys

sys.path.insert(0, "src")
from nexus.ui.recorder import launch_ui


def main():
    ap = argparse.ArgumentParser(description="NEXUS task recorder UI")
    ap.add_argument("--workdir", default=".", help="working directory for recordings (default: .)")
    ap.add_argument("--virtual-mouse", action="store_true",
                    help="open the virtual mouse screen-map panel alongside the recorder")
    args = ap.parse_args()
    launch_ui(args.workdir, virtual_mouse=args.virtual_mouse)


if __name__ == "__main__":
    main()
