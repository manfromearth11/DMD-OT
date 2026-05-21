# Local Experiment Scripts

이 폴더는 **로컬 실험 launcher 보관소**입니다.

재사용 가능한 실제 도구는 대부분 `dmd/scripts/`에 있고, 이 폴더에는 처음 실행할 때
바로 쓸 수 있는 canonical launcher만 둡니다. 날짜/설정이 박힌 개인 실험 launcher는
`run_..._0519.sh`처럼 만들어도 되지만 `.gitignore`에 걸려서 커밋되지 않습니다.

## 폴더 역할

| 위치 | 역할 | Git 추적 |
| --- | --- | --- |
| `dmd/scripts/` | 재사용 가능한 데이터 준비, cache 변환, 학습, FID 평가 도구 | 추적 |
| `scripts/README.md` | 이 안내 문서 | 추적 |
| `scripts/run_cifar10_dmd_original.sh` | 원래 DMD paired baseline launcher | 추적 |
| `scripts/run_cifar10_dmd_ot.sh` | DMD+OT launcher | 추적 |
| `scripts/run*_0519.sh`, `scripts/rerun*.sh` 등 | 날짜별/개인 실험 launcher | 무시 |

새 실험을 빨리 돌리고 싶으면 위 두 파일을 env var로 override해서 쓰거나, 개인용
`scripts/run_*.sh`를 복사해서 만들면 됩니다. 다만 재사용할 가치가 있는 기능은
`dmd/scripts/`로 올리는 것을 권장합니다.

## 원래 DMD 코드에서 바뀐 점

원래 DMD CIFAR-10 코드는 대략 다음 구조입니다.

1. EDM teacher가 만든 `(z_ref, y_ref)` distillation dataset을 읽습니다.
2. Student one-step generator가 `x_ref = G_theta(z_ref | c)`를 만듭니다.
3. Paired LPIPS regression으로 `x_ref`와 `y_ref`를 맞춥니다.
4. Full DMD에서는 distribution matching KL term과 fake denoiser training도 같이 씁니다.

이 repo의 변경점은 다음입니다.

### OT regression option

`dmd/dmd/loss.py`에 `--use-ot-reg=True` path를 추가했습니다.

기존 paired regression:

```text
z_i -> student x_i
z_i -> cached teacher y_i
LPIPS(x_i, y_i)
```

OT regression:

```text
student batch: x_1, ..., x_N
teacher batch: y_1, ..., y_N
cost C_ij = LPIPS_224(x_i, y_j)
C <- C / mean(C)
P = balanced Sinkhorn(C; eps, iters)
j*(i) = argmax_j P_ij
loss_reg = mean_i [N * P_i,j*(i) * LPIPS_224(x_i, y_j*(i))]
```

중요한 고정 사항:

- Marginal은 항상 balanced uniform입니다.
- Anchor 선택은 row-wise hard argmax입니다.
- OT weight는 항상 `N * P[i, j*]`입니다.
- 별도 `ot_weight` option은 실험 혼동을 줄이기 위해 쓰지 않습니다.
- Cost normalization은 mean으로만 합니다. Std로 나누지 않습니다.
- LPIPS는 PIQ VGG backbone을 사용하고, DMD 논문 CIFAR 설정처럼 입력을 `224x224`로 resize합니다.

### KL on/off

`--lambda-k`를 추가해서 DMD KL term을 조절합니다.

- `--lambda-k=1`: full DMD style
- `--lambda-k=0 --train-diffuser=False`: regression-only sanity test

`lambda_k=0` 실험에서는 fake denoiser를 훈련하거나 불필요하게 KL용 모델을 로드하지 않도록 했습니다.

### Cached teacher dataset

EDM teacher sample을 매번 online으로 만들지 않고, 먼저 disk cache로 저장한 뒤 HDF5로 변환해서
DMD training에서 빠르게 읽습니다.

저장되는 항목:

- `images`: teacher image, `[-1, 1]`, fp16
- `latents`: teacher input noise `z`, fp16
- `labels`: CIFAR-10 class id, conditional cache에서 사용
- `seeds`: reproducibility용 seed

### Class-batch sampler

`--class-batch=True`를 추가했습니다. Conditional CIFAR-10에서 한 training batch가 한 class만
포함하도록 샘플링합니다. OT 비교를 할 때 class mismatch를 줄이고, class-conditional 실험을
더 명확하게 보기 위한 옵션입니다.

### FID evaluation

`--fid-ref-path=/dataset/$USER/pretrained/cifar10-32x32.npz`를 넘기면 EDM reference stats를
사용해서 FID를 계산합니다. Checkpoint만 따로 다시 평가하는 wrapper도 있습니다.

## 훈련 전 환경 설정

기본 wrapper들은 `dmd/scripts/env_common.sh`를 source합니다.

기본값:

```bash
DATA_ROOT=/dataset/$USER
PYTHON=/dataset/$USER/conda-envs/edm-h100-lpips/bin/python
PYTHONPATH=<repo>/dmd:$PYTHONPATH
WANDB_DIR=/dataset/$USER/wandb
TMPDIR=/dataset/$USER/tmp
```

이미 서버에 env가 있으면 그대로 씁니다.

```bash
conda activate /dataset/$USER/conda-envs/edm-h100-lpips
wandb login
```

새 env를 만들 때는 H100/CUDA에 맞는 PyTorch를 먼저 맞추는 것이 중요합니다. 원본 DMD의
`dmd/environment.yml`은 오래된 PyTorch를 기준으로 하므로, 이 서버에서는 현재 쓰는
`edm-h100-lpips` env를 기준으로 맞추는 편이 안전합니다.

Wrapper에서 다른 Python을 쓰고 싶으면:

```bash
PYTHON=/path/to/python dmd/scripts/train_cifar10_cached_dmd.sh
```

## 데이터 준비 순서

### 1. EDM teacher와 FID ref 받기

```bash
dmd/scripts/prepare_cifar10_assets.sh
```

기본 저장 위치:

```text
/dataset/$USER/pretrained/
  edm-cifar10-32x32-cond-vp.pkl
  edm-cifar10-32x32-uncond-vp.pkl
  cifar10-32x32.npz
```

### 2. Real diffusion teacher cache 만들기

이 단계가 무겁습니다. EDM teacher를 18-step sampler로 돌려서 teacher images와 latents를
저장합니다. 내부적으로 `dmd/scripts/build_edm_teacher_cache.py`를 호출하므로 별도의
`~/edm` checkout은 필요 없습니다.

Conditional 500K, class-balanced:

```bash
TASK=cond dmd/scripts/build_cifar10_teacher_cache.sh
```

Unconditional 500K:

```bash
TASK=uncond dmd/scripts/build_cifar10_teacher_cache.sh
```

기본 출력:

```text
/dataset/$USER/edm_teacher_cache/cifar10-cond-edm18-500k/
/dataset/$USER/edm_teacher_cache/cifar10-uncond-edm18-500k/
```

이미 cache가 있으면 `RESUME=true` 기본값으로 이어서 진행합니다.

주요 override:

```bash
TASK=cond \
NUM_SAMPLES=500000 \
PER_CLASS=50000 \
BATCH=512 \
SHARD_SIZE=8192 \
dmd/scripts/build_cifar10_teacher_cache.sh
```

### 3. Teacher cache를 DMD용 HDF5로 변환

```bash
TASK=cond dmd/scripts/convert_cifar10_teacher_cache.sh
TASK=uncond dmd/scripts/convert_cifar10_teacher_cache.sh
```

기본 출력:

```text
/dataset/$USER/dmd_data/cifar10-cond-edm18-500k-array.hdf5
/dataset/$USER/dmd_data/cifar10-uncond-edm18-500k-array.hdf5
```

일부만 변환하고 싶으면:

```bash
TASK=cond MAX_SAMPLES=50000 dmd/scripts/convert_cifar10_teacher_cache.sh
```

## 학습 실행

가장 많이 쓰는 canonical launcher:

```bash
scripts/run_cifar10_dmd_original.sh
scripts/run_cifar10_dmd_ot.sh
```

두 launcher 모두 내부적으로 `dmd/scripts/train_cifar10_cached_dmd.sh`를 호출합니다.
개별 파라미터는 환경변수로 override합니다.

기본 training wrapper를 직접 호출할 수도 있습니다.

```bash
dmd/scripts/train_cifar10_cached_dmd.sh
```

### Paired regression sanity

```bash
TASK=cond \
BATCH_SIZE=128 \
LAMBDA_K=0 \
MAX_STEPS=50000 \
EVAL_EVERY=10000 \
WANDB_PROJECT=cifar10-dmd-sanity-test \
scripts/run_cifar10_dmd_original.sh
```

### OT regression sanity

```bash
TASK=cond \
BATCH_SIZE=64 \
OT_EPS=0.25 \
OT_ITERS=30 \
LAMBDA_K=0 \
MAX_STEPS=50000 \
EVAL_EVERY=10000 \
WANDB_PROJECT=cifar10-dmd-sanity-test \
scripts/run_cifar10_dmd_ot.sh
```

### Full DMD

```bash
TASK=cond \
BATCH_SIZE=128 \
LAMBDA_K=1 \
DMD_LOSS_LAMBDA=0.25 \
MAX_STEPS=50000 \
FID_NUM_SAMPLES=50000 \
WANDB_PROJECT=cifar10-dmd-full-test \
scripts/run_cifar10_dmd_original.sh
```

OT full DMD:

```bash
TASK=cond \
BATCH_SIZE=128 \
OT_EPS=0.2 \
LAMBDA_K=1 \
DMD_LOSS_LAMBDA=0.25 \
MAX_STEPS=50000 \
FID_NUM_SAMPLES=50000 \
WANDB_PROJECT=cifar10-dmd-full-test \
scripts/run_cifar10_dmd_ot.sh
```

## 주요 파라미터가 어디서 오는가

Wrapper는 대부분 환경변수로 설정을 받습니다. 환경변수를 안 주면 script 내부 default를 씁니다.
최종적으로는 `python -m dmd train ...` CLI argument로 변환됩니다.

| 환경변수 | 기본값 | 의미 |
| --- | --- | --- |
| `DATA_ROOT` | `/dataset/$USER` | 데이터, output, wandb root |
| `PYTHON` | `$DATA_ROOT/conda-envs/edm-h100-lpips/bin/python` | 실행할 Python |
| `TASK` | `cond` | `cond` 또는 `uncond` |
| `METHOD` | `paired` | `paired` 또는 `ot` |
| `BATCH_SIZE` | `128` | Training batch size |
| `MAX_STEPS` | `50000` | Optimizer step 수 |
| `EVAL_EVERY` | `10000` | FID/checkpoint 주기 |
| `FID_NUM_SAMPLES` | `50000` | FID sample 수 |
| `LAMBDA_K` | `0` | DMD KL term weight |
| `DMD_LOSS_LAMBDA` | `cond: 0.25`, `uncond: 0.5` | Regression loss weight |
| `OT_EPS` | `0.2` | Sinkhorn epsilon |
| `OT_ITERS` | `30` | Sinkhorn iteration 수 |
| `CLASS_BATCH` | `cond: true`, `uncond: false` | 한 batch를 한 class로 뽑을지 |
| `CACHE_DATA` | `true` | HDF5 array를 CPU RAM에 올릴지 |
| `LOG_WANDB` | `true` | W&B logging |
| `WANDB_PROJECT` | `cifar10-dmd-sanity-test` | W&B project |
| `NAME` | 자동 생성 | Run name |
| `OUTROOT` | `$DATA_ROOT/dmd_ot_runs/cifar10_cached_dmd` | Output root |

직접 전체 CLI를 보고 싶으면:

```bash
python -m dmd train --help
```

## Checkpoint FID 재계산

Training 중 FID가 꼬였거나, eval mode/seed를 맞춰 다시 보고 싶으면:

```bash
CHECKPOINT=/dataset/$USER/dmd_ot_runs/.../last_checkpoint.pt \
NUM=50000 \
BATCH=128 \
SEED=0 \
dmd/scripts/eval_cifar10_checkpoint_fid.sh
```

결과는 checkpoint 폴더에 `corrected_fid_*.json`으로 저장됩니다.

## Git 정책

커밋에 포함할 것:

- `dmd/dmd/*.py`의 실제 코드 변경
- `dmd/scripts/*.py`, `dmd/scripts/*.sh`의 재사용 도구
- `scripts/README.md`
- `scripts/.gitkeep`
- `scripts/run_cifar10_dmd_original.sh`
- `scripts/run_cifar10_dmd_ot.sh`

커밋하지 않을 것:

- 날짜/설정이 박힌 개인 `scripts/run*.sh`
- `scripts/rerun*.sh`
- `/dataset/$USER/...` 아래 결과물, cache, checkpoint, wandb 파일

즉, `scripts/`는 실험 노트북 같은 공간이고, `dmd/scripts/`는 다른 사람이 다시 쓸 수 있는
도구 모음입니다.
