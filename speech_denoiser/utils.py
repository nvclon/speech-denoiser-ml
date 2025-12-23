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


def dvc_pull_with_bootstrap(
    repo_root: Path,
    *,
    store_url: str | None,
    store_dir: Path,
    remote_name: str = "local_data",
) -> bool:
    """Run `dvc pull` and bootstrap local dvcstore from an archive if needed.

    Flow:
    1) Try `dvc pull -r <remote_name>`.
    2) If it fails, download `dvcstore.tar.gz` from store_url, extract into the
       parent of store_dir, ensure store_dir exists, then retry `dvc pull`.

    Returns:
        True if `dvc pull` eventually succeeds, else False.
    """

    def _run_pull() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["dvc", "pull", "-r", str(remote_name)],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
        )

    try:
        first = _run_pull()
    except FileNotFoundError:
        print("[WARN] dvc command not found; cannot pull.")
        return False

    if first.returncode == 0:
        return True

    if not store_url:
        # Keep it short; callers can decide what to do next.
        print("[WARN] dvc pull failed and no dvc.store_url is configured.")
        if first.stderr:
            print(first.stderr)
        return False

    # Bootstrap local store from archive
    try:
        try:
            import gdown  # type: ignore
        except Exception as e:
            print(f"[WARN] gdown is not installed: {e}")
            return False

        import shutil

        store_dir = store_dir.resolve()
        parent_dir = store_dir.parent
        archive_path = parent_dir / "dvcstore.tar.gz"

        print("[INFO] dvc pull failed; bootstrapping local dvcstore from archive...")
        print(f"[INFO] Downloading: {store_url}")
        parent_dir.mkdir(parents=True, exist_ok=True)

        downloaded_path = gdown.download(
            url=store_url,
            output=str(archive_path),
            quiet=False,
            fuzzy=True,
        )
        if not downloaded_path or not archive_path.exists():
            print("[WARN] Failed to download dvcstore.tar.gz")
            return False

        print(f"[INFO] Extracting: {archive_path} -> {parent_dir}")
        shutil.unpack_archive(str(archive_path), extract_dir=str(parent_dir))

        # Expected layout: parent_dir/dvcstore/...
        if not store_dir.exists():
            extracted_default = parent_dir / "dvcstore"
            if extracted_default.exists() and extracted_default != store_dir:
                # Move to configured store_dir
                store_dir.mkdir(parents=True, exist_ok=True)
                for child in extracted_default.iterdir():
                    shutil.move(str(child), str(store_dir / child.name))
                try:
                    extracted_default.rmdir()
                except OSError:
                    pass
            else:
                # Fallback: archive may contain `files/` directly
                files_dir = parent_dir / "files"
                if files_dir.exists():
                    store_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(files_dir), str(store_dir / "files"))

        if not store_dir.exists():
            print(f"[WARN] dvcstore not found after extraction at {store_dir}")
            return False

        second = _run_pull()
        if second.returncode == 0:
            return True
        if second.stderr:
            print(second.stderr)
        return False
    except Exception as e:
        print(f"[WARN] dvcstore bootstrap failed: {e}")
        return False
