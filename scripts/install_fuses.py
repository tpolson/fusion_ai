#!/usr/bin/env python3
"""Install Fusion Fuse plugins to the appropriate directory."""

import sys
import shutil
from pathlib import Path
import platform

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fusion_ai.config import get_fusion_fuse_path, PACKAGE_ROOT


def install_fuses(force: bool = False) -> bool:
    """Install Fuse plugins to Fusion directory.

    Args:
        force: Force overwrite existing files

    Returns:
        True if successful
    """
    # Get Fusion fuse path
    fuse_path = get_fusion_fuse_path()

    if fuse_path is None:
        print("Error: Could not determine Fusion Fuse directory for this platform")
        return False

    # Create directory if it doesn't exist
    fuse_path.mkdir(parents=True, exist_ok=True)

    # Get source fuse files
    source_dir = PACKAGE_ROOT / "fusion_ai" / "fuses"
    fuse_files = list(source_dir.glob("*.fuse"))

    if not fuse_files:
        print("Error: No Fuse files found in source directory")
        return False

    print(f"Installing Fuse plugins to: {fuse_path}")
    print(f"Found {len(fuse_files)} Fuse files")

    installed_count = 0
    skipped_count = 0

    for fuse_file in fuse_files:
        dest_file = fuse_path / fuse_file.name

        if dest_file.exists() and not force:
            print(f"  Skipping {fuse_file.name} (already exists, use --force to overwrite)")
            skipped_count += 1
            continue

        try:
            shutil.copy2(fuse_file, dest_file)
            print(f"  Installed: {fuse_file.name}")
            installed_count += 1
        except Exception as e:
            print(f"  Error installing {fuse_file.name}: {e}")

    print(f"\nInstallation complete:")
    print(f"  Installed: {installed_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"\nRestart Fusion to see the new plugins in the 'AI Tools' category")

    return installed_count > 0


def uninstall_fuses() -> bool:
    """Uninstall Fuse plugins from Fusion directory.

    Returns:
        True if successful
    """
    fuse_path = get_fusion_fuse_path()

    if fuse_path is None:
        print("Error: Could not determine Fusion Fuse directory")
        return False

    if not fuse_path.exists():
        print("Fusion Fuse directory does not exist")
        return False

    # Find installed fuse files
    installed_fuses = [
        "DepthAnythingV2.fuse",
        "QwenEdit.fuse",
    ]

    print(f"Uninstalling Fuse plugins from: {fuse_path}")

    removed_count = 0
    for fuse_name in installed_fuses:
        fuse_file = fuse_path / fuse_name
        if fuse_file.exists():
            try:
                fuse_file.unlink()
                print(f"  Removed: {fuse_name}")
                removed_count += 1
            except Exception as e:
                print(f"  Error removing {fuse_name}: {e}")
        else:
            print(f"  Not found: {fuse_name}")

    print(f"\nUninstall complete: {removed_count} files removed")
    return removed_count > 0


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Install Fusion AI Fuse plugins")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing files"
    )
    parser.add_argument(
        "--uninstall",
        action="store_true",
        help="Uninstall Fuse plugins"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Fusion directory path without installing"
    )

    args = parser.parse_args()

    if args.check:
        fuse_path = get_fusion_fuse_path()
        if fuse_path:
            print(f"Fusion Fuse directory: {fuse_path}")
            print(f"Exists: {fuse_path.exists()}")
        else:
            print("Could not determine Fusion Fuse directory")
        return

    if args.uninstall:
        success = uninstall_fuses()
    else:
        success = install_fuses(force=args.force)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
