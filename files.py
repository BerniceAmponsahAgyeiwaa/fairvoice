#!/usr/bin/env python3
import os

# Folders to ignore to avoid massive output
IGNORE_DIRS = {
    "CREMA-D", "AudioWAV", "audio", "audios", "splits",
    "raw", "processed", "metadata", ".git", ".venv",
}

# File extensions to ignore
IGNORE_EXTS = {".wav", ".mp3", ".flac", ".aiff", ".npz",".pt"}

def safe_listdir(path):
    """Return only safe items (ignore audio, LFS, large sets)."""
    items = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            if name in IGNORE_DIRS:
                continue
        else:
            # Ignore large audio/feature files
            ext = os.path.splitext(name)[1].lower()
            if ext in IGNORE_EXTS:
                continue
            # Ignore large git-lfs pointer files
            try:
                if os.path.getsize(full) > 2_000_000:  # >2MB
                    continue
            except:
                pass
        items.append(name)
    return items

def print_structure(root, indent=""):
    try:
        entries = safe_listdir(root)
    except PermissionError:
        return

    for entry in entries:
        full = os.path.join(root, entry)
        print(f"{indent}- {entry}/" if os.path.isdir(full) else f"{indent}- {entry}")
        if os.path.isdir(full):
            print_structure(full, indent + "  ")

if __name__ == "__main__":
    print("\n📁 PROJECT STRUCTURE (SAFE VIEW)\n")
    print_structure(".")
    print("\n✔ Done. Ignored large/noisy folders.\n")
