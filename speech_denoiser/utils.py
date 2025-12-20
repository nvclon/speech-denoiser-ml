from __future__ import annotations

import subprocess
from pathlib import Path


def _is_expected_data_layout(data_dir: Path) -> bool:
    return (data_dir / "train").exists() or (data_dir / "test").exists()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit_id(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except Exception:
        return None


def try_dvc_pull(repo_root: Path, targets: list[str] | None = None) -> None:
    cmd = ["dvc", "pull"]
    if targets:
        cmd.extend(targets)

    try:
        subprocess.run(cmd, cwd=str(repo_root), check=True)
    except Exception as e:
        # DVC remote may be unavailable on a clean machine.
        print(f"[WARN] dvc pull failed: {e}")


def try_gdown_folder(url: str, output_dir: Path) -> None:
    """Download a public Google Drive folder using gdown.

    This avoids interactive OAuth flows (which may be blocked for some accounts).
    """

    if not url:
        raise ValueError("Google Drive folder URL is empty")

    try:
        import gdown  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "gdown is not installed. Install it (e.g. `poetry add gdown`) and retry."
        ) from e

    ensure_dir(output_dir)

    # Try downloading as a folder first (if it's a folder link)
    # If it fails or if it's a file link, gdown might handle it differently.
    # For robustness: check if it looks like a file ID or folder.
    # But gdown.download_folder is specific.
    # Let's try to download as an archive if folder download fails or if we decide to support zips.

    # Strategy:
    # 1. Try download as file (archive). Folder links may fail or return HTML.
    # 2. If that file is a valid zip, extract it.
    # 3. If not, try download_folder.

    # However, distinguishing by URL is hard. Let's assume if the user provides a ZIP link,
    # they want it extracted.

    # Simplified approach: try to download as a file to a temp location.
    # If it's a zip, extract. If not, try folder.

    # Actually, let's just support the ZIP workflow explicitly as it's more reliable for datasets.

    import shutil
    import zipfile

    # Temporary path for potential zip download
    temp_zip = output_dir / "dataset_temp.zip"

    print(f"[INFO] Attempting to download from {url}...")

    # Try downloading as a single file (archive)
    downloaded_path = gdown.download(
        url=url,
        output=str(temp_zip),
        quiet=False,
        fuzzy=True,
    )

    downloaded_ok = bool(downloaded_path) and Path(str(downloaded_path)).exists()
    downloaded_is_zip = downloaded_ok and zipfile.is_zipfile(str(downloaded_path))

    if downloaded_is_zip:
        print(f"[INFO] Downloaded archive: {downloaded_path}. Extracting...")
        shutil.unpack_archive(str(downloaded_path), extract_dir=output_dir)
        Path(str(downloaded_path)).unlink()
    else:
        # Not a zip: try folder download (common for Drive folder links).
        if temp_zip.exists():
            temp_zip.unlink()

        print("[INFO] Not a zip archive. Trying gdown.download_folder...")
        gdown.download_folder(
            url=url,
            output=str(output_dir),
            quiet=False,
            use_cookies=False,
        )

    # If we ended up with output_dir/<single_subdir>/(train|test), move up one level.
    if _is_expected_data_layout(output_dir):
        return

    subdirs = [p for p in output_dir.iterdir() if p.is_dir()]
    if len(subdirs) != 1:
        return
    only = subdirs[0]
    if not _is_expected_data_layout(only):
        return

    for child in only.iterdir():
        child.replace(output_dir / child.name)
    try:
        only.rmdir()
    except OSError:
        # If non-empty for any reason, keep it.
        pass
