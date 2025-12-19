from __future__ import annotations

import csv
import socket
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import CSVLogger, MLFlowLogger

from speech_denoiser.data import SpeechDataModule
from speech_denoiser.lightning_module import DenoiserLightningModule
from speech_denoiser.utils import ensure_dir, git_commit_id, try_dvc_pull, try_gdown_folder


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _maybe_pull_data(cfg: DictConfig) -> None:
    repo_root = _repo_root()
    data_dir = repo_root / str(cfg.data.data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if train_dir.exists() and test_dir.exists():
        return

    gdrive_url = getattr(cfg.data, "gdrive_folder_url", None)
    if gdrive_url:
        try:
            print("[INFO] data/ not found; downloading from Google Drive via gdown...")
            try_gdown_folder(str(gdrive_url), output_dir=data_dir)
        except Exception as e:
            print(f"[WARN] gdown download failed: {e}")

    if train_dir.exists() and test_dir.exists():
        return

    # Fall back to DVC (if remote is configured and accessible).
    try_dvc_pull(repo_root, targets=["data/train.dvc", "data/test.dvc"])

    if not (train_dir.exists() and test_dir.exists()):
        raise FileNotFoundError(
            "Dataset is missing. Download it with `poetry run speech-denoiser download_data` "
            "or configure DVC remote and run `poetry run dvc pull`."
        )


def _write_plots_from_csv(metrics_csv: Path, plots_dir: Path) -> None:
    if not metrics_csv.exists():
        return

    ensure_dir(plots_dir)

    # Keep a copy of raw metrics CSV
    raw_copy = plots_dir / "metrics.csv"
    if raw_copy.resolve() != metrics_csv.resolve():
        raw_copy.write_bytes(metrics_csv.read_bytes())

    summary_csv = plots_dir / "metrics_last_row.csv"

    last_row: dict[str, str] | None = None
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last_row = row

    if last_row is None:
        return

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(last_row.keys()))
        writer.writeheader()
        writer.writerow(last_row)

    def _extract_xy(
        rows: Iterable[dict[str, str]], x_key: str, y_key: str
    ) -> tuple[list[float], list[float]]:
        xs: list[float] = []
        ys: list[float] = []
        for r in rows:
            if y_key not in r:
                continue
            y_raw = r.get(y_key)
            x_raw = r.get(x_key)
            if y_raw is None or y_raw == "":
                continue
            if x_raw is None or x_raw == "":
                continue
            try:
                xs.append(float(x_raw))
                ys.append(float(y_raw))
            except ValueError:
                continue
        return xs, ys

    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Lightning writes step/epoch columns; plot against step by default.
    x_key = "step" if rows and "step" in rows[0] else "epoch"

    candidates = {
        "train_loss": ["train/loss_epoch", "train/loss_step", "train/loss"],
        "val_loss": ["val/loss", "val/loss_epoch"],
        "val_si_sdr": ["val/si_sdr", "val/si_sdr_epoch"],
    }

    def _first_present(keys: list[str]) -> str | None:
        if not rows:
            return None
        for key in keys:
            if key in rows[0]:
                return key
        return None

    for plot_name, metric_keys in candidates.items():
        y_key = _first_present(metric_keys)
        if y_key is None:
            continue
        xs, ys = _extract_xy(rows, x_key=x_key, y_key=y_key)
        if not xs:
            continue

        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys)
        plt.title(plot_name)
        plt.xlabel(x_key)
        plt.ylabel(y_key)
        plt.grid(True)
        out_path = plots_dir / f"{plot_name}.png"
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def _mlflow_http_reachable(tracking_uri: str, timeout_s: float = 0.5) -> bool:
    try:
        parsed = urlparse(tracking_uri)
        if parsed.scheme not in {"http", "https"}:
            return True
        host = parsed.hostname
        if host is None:
            return False
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except Exception:
        return False


def train(cfg: DictConfig) -> None:
    repo_root = _repo_root()

    _maybe_pull_data(cfg)

    pl.seed_everything(int(cfg.seed), workers=True)

    artifacts_dir = ensure_dir(repo_root / str(cfg.paths.artifacts_dir))
    checkpoints_dir = ensure_dir(repo_root / str(cfg.paths.checkpoints_dir))
    plots_dir = ensure_dir(repo_root / str(cfg.paths.plots_dir))

    mlflow_logger: MLFlowLogger | None = None
    try:
        tracking_uri = str(cfg.mlflow.tracking_uri)
        if not _mlflow_http_reachable(tracking_uri):
            raise RuntimeError(f"MLflow tracking URI not reachable: {tracking_uri}")

        mlflow_logger = MLFlowLogger(
            tracking_uri=tracking_uri,
            experiment_name=str(cfg.mlflow.experiment_name),
            run_name=str(cfg.mlflow.run_name),
        )

        commit = git_commit_id(repo_root)
        if commit is not None:
            mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "git_commit", commit)

        mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        # Keep training runnable even if MLflow server is down.
        print(f"[WARN] MLflow logger disabled: {e}")
        mlflow_logger = None

    csv_logger = CSVLogger(save_dir=str(plots_dir), name="lightning_logs")

    model_cfg = cfg.model
    module = DenoiserLightningModule(
        model_name=str(model_cfg.model_name),
        learning_rate=float(model_cfg.learning_rate),
        loss_function=str(model_cfg.loss_function),
        optimizer=str(model_cfg.optimizer),
        dae_num_layers=int(getattr(model_cfg, "num_layers", 4)),
        dae_kernel_size=int(getattr(model_cfg, "kernel_size", 5)),
        demucs_channels=int(getattr(model_cfg, "channels", 32)),
        demucs_depth=int(getattr(model_cfg, "depth", 4)),
        demucs_kernel_size=int(getattr(model_cfg, "kernel_size", 8)),
    )

    dm = SpeechDataModule(
        train_noisy_dir=repo_root / str(cfg.data.train_noisy_path),
        train_clean_dir=repo_root / str(cfg.data.train_clean_path),
        sample_rate=int(cfg.audio.sample_rate),
        segment_seconds=float(cfg.audio.segment_length),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
        seed=int(cfg.seed),
        val_speaker_fraction=0.15,
    )
    dm.setup()

    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_kwargs, dict)
    loggers = [csv_logger]
    if mlflow_logger is not None:
        loggers.insert(0, mlflow_logger)

    trainer = pl.Trainer(
        **trainer_kwargs,
        default_root_dir=str(artifacts_dir),
        logger=loggers,
        enable_checkpointing=True,
        callbacks=[],
    )

    trainer.fit(
        module, train_dataloaders=dm.train_dataloader(), val_dataloaders=dm.val_dataloader()
    )

    # Save a final checkpoint for convenience
    model_name = str(cfg.model.model_name)
    safe_model_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_name)
    final_ckpt = checkpoints_dir / f"latest_{safe_model_name}.ckpt"
    trainer.save_checkpoint(str(final_ckpt))

    # Store metrics artifacts under plots/
    metrics_csv = Path(csv_logger.log_dir) / "metrics.csv"
    _write_plots_from_csv(metrics_csv, plots_dir)
