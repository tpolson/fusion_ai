#!/usr/bin/env python3
"""Configure Fusion fuses with correct Python interpreter path."""

import sys
import os
from pathlib import Path
import shutil
import argparse


def get_python_path():
    """Get the path to the current Python interpreter."""
    return sys.executable


def get_fusion_fuses_dir():
    """Get the Fusion Studio Fuses directory for the current platform."""
    import platform

    system = platform.system()

    if system == "Windows":
        appdata = os.getenv("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA environment variable not set")
        return Path(appdata) / "Blackmagic Design" / "Fusion" / "Fuses"

    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Blackmagic Design" / "Fusion" / "Fuses"

    elif system == "Linux":
        # Try multiple possible locations
        locations = [
            Path.home() / ".fusion" / "BlackmagicDesign" / "Fusion" / "Fuses",
            Path.home() / ".local" / "share" / "Blackmagic Design" / "Fusion" / "Fuses",
        ]
        for loc in locations:
            if loc.parent.exists():
                return loc
        return locations[0]  # Default to first

    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def get_resolve_fuses_dir():
    """Get the DaVinci Resolve Fuses directory for the current platform."""
    import platform

    system = platform.system()

    if system == "Windows":
        appdata = os.getenv("APPDATA")
        if not appdata:
            raise RuntimeError("APPDATA environment variable not set")
        return Path(appdata) / "Blackmagic Design" / "DaVinci Resolve" / "Fusion" / "Fuses"

    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Blackmagic Design" / "DaVinci Resolve" / "Fusion" / "Fuses"

    elif system == "Linux":
        # Try multiple possible locations
        locations = [
            Path.home() / ".local" / "share" / "DaVinci Resolve" / "Fusion" / "Fuses",
            Path.home() / ".local" / "share" / "Blackmagic Design" / "DaVinci Resolve" / "Fusion" / "Fuses",
        ]
        for loc in locations:
            if loc.parent.exists():
                return loc
        return locations[0]  # Default to first

    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def configure_fuse_file(fuse_path: Path, python_path: str) -> bool:
    """Configure a single fuse file with Python path.

    Args:
        fuse_path: Path to the .fuse file
        python_path: Path to Python interpreter

    Returns:
        True if file was modified, False otherwise
    """
    # Read the file
    try:
        with open(fuse_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {fuse_path}: {e}")
        return False

    # Check if already configured
    if python_path in content:
        print(f"  ✓ {fuse_path.name} already configured")
        return False

    # Add Python path configuration at the top of the file
    # Insert after the initial comment block
    lines = content.split('\n')

    # Find where to insert (after comment block, before FuRegisterClass)
    insert_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('FuRegisterClass'):
            insert_idx = i
            break

    if insert_idx == 0:
        # Couldn't find FuRegisterClass, insert after first comment block
        for i, line in enumerate(lines):
            if not line.strip().startswith('--') and line.strip() != '':
                insert_idx = i
                break

    # Create configuration lines
    config_lines = [
        "",
        "-- Python interpreter path (auto-configured by configure_fuses.py)",
        f'local PYTHON_PATH = "{python_path}"',
        "",
    ]

    # Insert configuration
    lines = lines[:insert_idx] + config_lines + lines[insert_idx:]

    # Update python command usage in the file
    new_content = '\n'.join(lines)

    # Replace python command patterns
    replacements = [
        ('python -c "from', 'PYTHON_PATH .. " -c \\"from'),
        ("python -c 'from", "PYTHON_PATH .. \" -c 'from"),
        ('python -c "', 'PYTHON_PATH .. " -c \\"'),
    ]

    for old, new in replacements:
        new_content = new_content.replace(old, new)

    # Write back
    try:
        with open(fuse_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"  ✓ {fuse_path.name} configured")
        return True
    except Exception as e:
        print(f"  ✗ Error writing {fuse_path}: {e}")
        return False


def install_to_directory(source_fuses_dir: Path, dest_fuses_dir: Path, python_path: str, app_name: str):
    """Install fuses to a specific directory.

    Args:
        source_fuses_dir: Source directory with .fuse files
        dest_fuses_dir: Destination directory
        python_path: Python interpreter path
        app_name: Application name (for display)

    Returns:
        Tuple of (installed_count, configured_count)
    """
    print(f"\n{app_name} Fuses directory: {dest_fuses_dir}")

    # Create destination directory if it doesn't exist
    dest_fuses_dir.mkdir(parents=True, exist_ok=True)

    # Find all .fuse files
    fuse_files = list(source_fuses_dir.glob("*.fuse"))

    print(f"Installing and configuring fuses:")
    print("-" * 70)

    configured_count = 0
    installed_count = 0

    for fuse_file in fuse_files:
        dest_file = dest_fuses_dir / fuse_file.name

        # Copy file to destination
        try:
            shutil.copy2(fuse_file, dest_file)
            installed_count += 1
        except Exception as e:
            print(f"  ✗ Error copying {fuse_file.name}: {e}")
            continue

        # Configure with Python path
        if configure_fuse_file(dest_file, python_path):
            configured_count += 1

    print("-" * 70)
    print(f"✓ {app_name} installation complete!")
    print(f"  • {installed_count} fuses installed")
    print(f"  • {configured_count} fuses configured")
    print(f"  • Location: {dest_fuses_dir}")

    return installed_count, configured_count


def main():
    """Main configuration function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Configure and install Fusion AI fuses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/configure_fuses.py                # Install to Fusion Studio (default)
  python scripts/configure_fuses.py --resolve      # Install to DaVinci Resolve
  python scripts/configure_fuses.py --both         # Install to both
  python scripts/configure_fuses.py --fusion       # Install to Fusion Studio (explicit)
        """
    )

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--fusion',
        action='store_true',
        help='Install to Fusion Studio only (default)'
    )
    group.add_argument(
        '--resolve',
        action='store_true',
        help='Install to DaVinci Resolve only'
    )
    group.add_argument(
        '--both',
        action='store_true',
        help='Install to both Fusion Studio and DaVinci Resolve'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("Fusion AI - Fuse Configuration Script")
    print("=" * 70)

    # Get Python path
    python_path = get_python_path()
    print(f"\nPython interpreter: {python_path}")

    # Normalize path for Lua (forward slashes)
    if os.name == 'nt':  # Windows
        python_path = python_path.replace('\\', '/')

    # Get source fuses directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    source_fuses_dir = project_root / "fusion_ai" / "fuses"

    if not source_fuses_dir.exists():
        print(f"\n✗ Error: Source fuses directory not found: {source_fuses_dir}")
        sys.exit(1)

    # Find all .fuse files
    fuse_files = list(source_fuses_dir.glob("*.fuse"))

    if not fuse_files:
        print(f"\n✗ No .fuse files found in {source_fuses_dir}")
        sys.exit(1)

    print(f"\nFound {len(fuse_files)} fuse files to install")

    # Determine installation targets
    install_fusion = args.fusion or args.both or (not args.resolve and not args.both)
    install_resolve = args.resolve or args.both

    total_installed = 0
    total_configured = 0
    install_locations = []

    # Install to Fusion Studio
    if install_fusion:
        try:
            fusion_dir = get_fusion_fuses_dir()
            installed, configured = install_to_directory(
                source_fuses_dir, fusion_dir, python_path, "Fusion Studio"
            )
            total_installed += installed
            total_configured += configured
            install_locations.append(("Fusion Studio", fusion_dir))
        except Exception as e:
            print(f"\n✗ Error installing to Fusion Studio: {e}")

    # Install to DaVinci Resolve
    if install_resolve:
        try:
            resolve_dir = get_resolve_fuses_dir()
            installed, configured = install_to_directory(
                source_fuses_dir, resolve_dir, python_path, "DaVinci Resolve"
            )
            total_installed += installed
            total_configured += configured
            install_locations.append(("DaVinci Resolve", resolve_dir))
        except Exception as e:
            print(f"\n✗ Error installing to DaVinci Resolve: {e}")

    # Print summary
    print("\n" + "=" * 70)
    print("Installation Summary")
    print("=" * 70)
    print(f"Total fuses installed: {total_installed}")
    print(f"Total fuses configured: {total_configured}")
    print(f"\nInstalled to {len(install_locations)} location(s):")
    for app_name, location in install_locations:
        print(f"  • {app_name}: {location}")

    # Print next steps
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)

    apps_to_restart = []
    if install_fusion:
        apps_to_restart.append("Fusion Studio")
    if install_resolve:
        apps_to_restart.append("DaVinci Resolve")

    print(f"1. Restart {' and '.join(apps_to_restart)}")
    print("2. Look for 'AI Tools' category in the node list")
    if install_resolve:
        print("   (In Resolve: switch to Fusion page first)")
    print("3. Add an AI node (e.g., DepthAnythingV2)")
    print("4. Connect to a Loader and render")
    print("\nIf nodes don't appear, check:")
    for app_name, location in install_locations:
        print(f"  • {app_name} fuses in: {location}")
    print(f"  • Python can import fusion_ai:")
    print(f"    {python_path} -c 'import fusion_ai'")
    print("=" * 70)


if __name__ == "__main__":
    main()
