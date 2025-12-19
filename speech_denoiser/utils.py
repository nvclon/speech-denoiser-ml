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

    # gdown may download into output_dir/<folder_name>/...; we normalize below.
    gdown.download_folder(url=url, output=str(output_dir), quiet=False, use_cookies=False)

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
