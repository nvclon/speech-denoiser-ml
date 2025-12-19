from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset


def _speaker_id_from_name(filename: str) -> str:
    # Example: p226_001.wav -> p226
    stem = Path(filename).stem
    return stem.split("_")[0]


def _list_wavs(directory: Path) -> list[Path]:
    return sorted([p for p in directory.glob("*.wav") if p.is_file()])


@dataclass(frozen=True)
class Split:
    train_files: list[str]
    val_files: list[str]


def split_by_speaker(files: list[Path], val_speaker_fraction: float, seed: int) -> Split:
    rng = torch.Generator().manual_seed(seed)

    speakers = sorted({_speaker_id_from_name(p.name) for p in files})
    if not speakers:
        return Split(train_files=[], val_files=[])

    perm = torch.randperm(len(speakers), generator=rng).tolist()
    speakers_shuffled = [speakers[i] for i in perm]

    val_count = max(1, int(round(len(speakers) * val_speaker_fraction)))
    val_speakers = set(speakers_shuffled[:val_count])

    train_files: list[str] = []
    val_files: list[str] = []

    for path in files:
        (val_files if _speaker_id_from_name(path.name) in val_speakers else train_files).append(
            path.name
        )

    return Split(train_files=train_files, val_files=val_files)


class PairedWavDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        noisy_dir: Path,
        clean_dir: Path,
        filenames: list[str],
        sample_rate: int,
        segment_seconds: float,
        random_crop: bool,
        seed: int,
    ) -> None:
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.filenames = filenames
        self.sample_rate = sample_rate
        self.segment_seconds = float(segment_seconds)
        self.segment_samples = int(round(segment_seconds * sample_rate))
        self.random_crop = random_crop
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return len(self.filenames)

    def _to_mono(self, data_2d: np.ndarray) -> torch.Tensor:
        # soundfile returns shape [T, C]
        wav = torch.from_numpy(np.asarray(data_2d)).transpose(0, 1)  # [C, T]
        if wav.numel() == 0:
            return torch.zeros((1, 0), dtype=torch.float32)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav

    def _resample(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        if sr == self.sample_rate:
            return wav
        return torchaudio.functional.resample(wav, orig_freq=sr, new_freq=self.sample_rate)

    def _pad_or_trim_to_target(self, wav: torch.Tensor) -> torch.Tensor:
        """Ensure exact target segment length after (optional) resampling."""

        length = int(wav.shape[-1])
        if length == self.segment_samples:
            return wav
        if length < self.segment_samples:
            pad = self.segment_samples - length
            return torch.nn.functional.pad(wav, (0, pad))
        return wav[..., : self.segment_samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        filename = self.filenames[index]

        noisy_path = self.noisy_dir / filename
        clean_path = self.clean_dir / filename

        # Read noisy once: use its timing for aligned crops.
        with sf.SoundFile(str(noisy_path)) as f_noisy:
            noisy_sr = int(f_noisy.samplerate)
            noisy_frames = int(f_noisy.frames)
            seg_frames_noisy = int(round(self.segment_seconds * noisy_sr))

            if seg_frames_noisy <= 0 or noisy_frames <= seg_frames_noisy:
                start_noisy = 0
            else:
                max_start = noisy_frames - seg_frames_noisy
                if self.random_crop:
                    start_noisy = int(
                        torch.randint(0, max_start + 1, (1,), generator=self.rng).item()
                    )
                else:
                    start_noisy = max_start // 2

            start_noisy = max(0, min(start_noisy, max(0, noisy_frames)))
            seg_frames_noisy = int(min(seg_frames_noisy, max(0, noisy_frames - start_noisy)))
            f_noisy.seek(start_noisy)
            noisy_data = f_noisy.read(frames=seg_frames_noisy, dtype="float32", always_2d=True)
            noisy = self._to_mono(noisy_data)

        # Map start time to the clean file sample rate.
        start_seconds = start_noisy / float(noisy_sr) if noisy_sr > 0 else 0.0

        with sf.SoundFile(str(clean_path)) as f_clean:
            clean_sr = int(f_clean.samplerate)
            clean_frames = int(f_clean.frames)
            seg_frames_clean = int(round(self.segment_seconds * clean_sr))

            if seg_frames_clean <= 0 or clean_frames <= seg_frames_clean:
                start_clean = 0
            else:
                start_clean = int(round(start_seconds * clean_sr))
                start_clean = max(0, min(start_clean, clean_frames - seg_frames_clean))

            start_clean = max(0, min(start_clean, max(0, clean_frames)))
            seg_frames_clean = int(min(seg_frames_clean, max(0, clean_frames - start_clean)))
            f_clean.seek(start_clean)
            clean_data = f_clean.read(frames=seg_frames_clean, dtype="float32", always_2d=True)
            clean = self._to_mono(clean_data)

        noisy = self._resample(noisy, noisy_sr)
        clean = self._resample(clean, clean_sr)

        noisy = self._pad_or_trim_to_target(noisy)
        clean = self._pad_or_trim_to_target(clean)

        return noisy, clean


class SpeechDataModule:
    def __init__(
        self,
        train_noisy_dir: Path,
        train_clean_dir: Path,
        sample_rate: int,
        segment_seconds: float,
        batch_size: int,
        num_workers: int,
        seed: int,
        val_speaker_fraction: float = 0.15,
    ) -> None:
        self.train_noisy_dir = train_noisy_dir
        self.train_clean_dir = train_clean_dir
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_speaker_fraction = val_speaker_fraction

        self._train_ds: PairedWavDataset | None = None
        self._val_ds: PairedWavDataset | None = None

    def setup(self) -> None:
        noisy_files = _list_wavs(self.train_noisy_dir)
        split = split_by_speaker(noisy_files, self.val_speaker_fraction, self.seed)

        self._train_ds = PairedWavDataset(
            noisy_dir=self.train_noisy_dir,
            clean_dir=self.train_clean_dir,
            filenames=split.train_files,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            random_crop=True,
            seed=self.seed,
        )
        self._val_ds = PairedWavDataset(
            noisy_dir=self.train_noisy_dir,
            clean_dir=self.train_clean_dir,
            filenames=split.val_files,
            sample_rate=self.sample_rate,
            segment_seconds=self.segment_seconds,
            random_crop=False,
            seed=self.seed,
        )

    def train_dataloader(self) -> DataLoader:
        assert self._train_ds is not None
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )

    def val_dataloader(self) -> DataLoader:
        assert self._val_ds is not None
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
