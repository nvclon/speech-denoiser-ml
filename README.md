# Speech Denoiser (MLOps project)

Проект: **шумоподавление речи (Speech Enhancement) при помощи нейросетей**.

Автор: **Ветошкин Иван Сергеевич**

## Problem statement

Цель проекта — обучить модель шумоподавления речи, которая преобразует зашумлённый аудиосигнал в очищенный.
Применения: онлайн-звонки, голосовые ассистенты, распознавание речи (Speech-to-Text).

**Вход:** `wav` с шумом.

**Выход:** `wav` с очищенной речью.

## Metrics

Основная метрика качества: **SI-SDR** (Scale-Invariant Signal-to-Distortion Ratio).

Целевой ориентир: порядка **5–6 dB SI‑SDR** (в литературе для Demucs встречаются значения около ~6 dB).

## Data

Датасет: пары (noisy, clean) `wav`.

Разделение:

- Валидация: 15% дикторов из train
- Тест: 15% дикторов из train (отдельный test split остаётся неизменным)

Для воспроизводимости фиксируются сиды разбиения и обучения.

Хранение данных: **DVC**.

## Models

- Baseline: **DAE** (convolutional denoising autoencoder на 1D-свертках).
- Вторая модель: **Demucs** (используется реализация из https://github.com/facebookresearch/demucs через PyPI-пакет `demucs`).

## Setup

Требования:

- Python `>=3.10,<3.15`
- `poetry`
- `dvc`

Установка:

1. Установить зависимости

```bash
poetry install --with dev
```

Если планируете обучать **Demucs**, установите extra-зависимости:

```bash
poetry install --with dev -E demucs
```

2. (Опционально) Включить CUDA (GPU) для PyTorch

По умолчанию Poetry может поставить CPU-сборку `torch`. Для обучения на GPU выполните:

```bash
poetry run speech-denoiser setup_cuda
```

Если у вас Windows + Python 3.13 и после установки всё равно `cuda=False`, используйте Python 3.12 (для него CUDA wheels обычно доступны).

Проверка:

```bash
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

3. Установить git hooks

```bash
poetry run pre-commit install
poetry run pre-commit run -a
```

3. (Важно) Настроить DVC remote

В репозитории включен DVC tracking для `data/train` и `data/test`, но **remote должен быть доступен проверяющему**.
На некоторых аккаунтах Google OAuth для DVC (`gdrive`) может быть заблокирован ("Приложение заблокировано").
В этом репозитории добавлен простой способ скачать датасет **без OAuth** через `gdown`.

### Скачать датасет через gdown (рекомендуется)

Команда скачает публичную Google Drive папку в `data/` (ссылка уже прописана в `configs/data/dataset.yaml`):

```bash
poetry run speech-denoiser download_data
```

Проверка наличия данных:

```bash
python -c "from pathlib import Path; print((Path('data')/'train').exists(), (Path('data')/'test').exists())"
```

### DVC remote: Google Drive (опционально)

Если вы загрузили папку `data/` в Google Drive, самый простой вариант — сделать DVC remote типа `gdrive://<FOLDER_ID>`.

1. Получить `FOLDER_ID`
   - Откройте папку в браузере и возьмите ID из URL вида `.../folders/<FOLDER_ID>`
   - Или из share-link.

2. Добавить remote и сделать его default:

```bash
poetry run dvc remote add -d gdrive_data gdrive://<FOLDER_ID>
poetry run dvc remote modify gdrive_data gdrive_use_service_account false
```

Если при авторизации появляется ошибка Google вида «Приложение заблокировано…», используйте **Service Account** (рекомендуется для воспроизводимости на “чистой” машине):

1. В Google Cloud Console:
   - Создайте проект
   - Включите **Google Drive API**
   - Создайте **Service Account**
   - Сгенерируйте ключ **JSON** (скачайте файл)

2. В Google Drive:
   - Откройте папку `data/` (ту же, что по `FOLDER_ID`)
   - Поделитесь этой папкой с email сервисного аккаунта (вида `...@....iam.gserviceaccount.com`) с правами **Editor**

3. Локально (путь до JSON укажите свой):

```bash
poetry run dvc remote modify gdrive_data gdrive_use_service_account true
poetry run dvc remote modify gdrive_data gdrive_service_account_json_file_path path/to/service-account.json
```

Важно: `service-account.json` не коммитьте в git. Передайте файл проверяющему отдельно (или настройте свои креды на его машине).

3. Запушить данные в remote (один раз на вашей машине):

```bash
poetry run dvc push
```

4. Проверка на “чистой” машине:

```bash
poetry run dvc pull
```

Примечание: первый `dvc push/pull` может попросить авторизацию Google (через `pydrive2`) и на некоторых аккаунтах быть заблокирован.

## Train

Тренировка запускается через CLI с Hydra-конфигами из `configs/`:

```bash
poetry run speech-denoiser train
```

Тренировка Demucs (как второй вариант модели):

```bash
poetry run speech-denoiser train model=demucs
```

Переопределение параметров Hydra:

```bash
poetry run speech-denoiser train trainer.max_epochs=10 data.batch_size=8
```

Во время обучения логируются метрики и гиперпараметры в MLflow (по умолчанию ожидается `http://127.0.0.1:8080`).

## Production preparation

Экспорт в ONNX (после обучения):

```bash
poetry run speech-denoiser export_onnx export.ckpt_path=artifacts/checkpoints/latest_DAE_baseline.ckpt
```

Артефакты:

- чекпойнты: `artifacts/checkpoints/`
- onnx: `artifacts/onnx/`
- графики: `plots/<model_name>/` (например `plots/DAE_baseline/` или `plots/demucs_v3_tiny/`)

## Infer

Инференс на новом wav:

```bash
poetry run speech-denoiser infer \
	ckpt_path=artifacts/checkpoints/latest_DAE_baseline.ckpt \
	input_wav=path/to/noisy.wav \
	output_dir=artifacts/predictions
```

Формат входа: mono `wav` (если стерео — будет сведено в mono).

## Inference server (Triton) (optional, max points)

Ниже описан вариант сервинга через **NVIDIA Triton Inference Server**.

### 1) Подготовить Triton model repository из чекпойнта

Команда экспортирует ONNX и создаст Triton model repository в `artifacts/triton/<model_name>/`:

```bash
poetry run speech-denoiser prepare_triton_repo \
	model=dae export.ckpt_path=artifacts/checkpoints/latest_DAE_baseline.ckpt
```

По умолчанию имя модели для Triton: `denoiser_onnx`, путь берётся из Hydra: `paths.triton_repo_dir`.

### 2) Запустить Triton Server

Требуется Docker + NVIDIA Container Toolkit (или WSL2 с GPU).

PowerShell:

```powershell
./scripts/run_triton.ps1 -ModelRepo artifacts/triton/DAE_baseline
```

Triton поднимется на портах:

- HTTP: `8000`
- gRPC: `8001`
- metrics: `8002`

### 3) Сделать запрос к Triton (клиент)

Установить optional dependency:

```bash
poetry install -E triton
```

Примечание: `tritonclient` как правило доступен только для Linux. Если у вас Windows,
используйте WSL2/Linux для клиента (или делайте запросы из другого окружения).

Запустить инференс через Triton:

```bash
poetry run speech-denoiser triton_infer \
	triton.input_wav=path/to/noisy.wav \
	triton.url=127.0.0.1:8000
```

Выход будет записан в `artifacts/predictions/triton/`.

### (Опционально) TensorRT

Можно собрать TensorRT engine из ONNX через `trtexec` в NVIDIA контейнере:

```powershell
./scripts/build_trt_engine.ps1 -OnnxPath artifacts/onnx/denoiser.onnx -TritonRepo artifacts/triton/DAE_baseline
```

Далее нужно добавить `config.pbtxt` для TensorRT backend (`platform: tensorrt_plan`).
