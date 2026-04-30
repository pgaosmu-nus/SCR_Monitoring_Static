# SCR_Monitoring_Static

Static SCR monitoring and inversion experiments based on a two-stage BMN-SCR-DD pipeline.

The current mainline workflow is:

1. train a data-driven `Decoder`
2. freeze that Decoder
3. train an `Encoder` that maps sparse observations to global parameters
4. reconstruct the full SCR response through `Encoder -> Decoder`

## Overview

This repository currently contains three related tracks:

- `scr_exact_bvp_solver.py`: exact static SCR boundary-value solver
- `Decoder_DD.py`: pure data-driven Decoder training and dataset generation
- `BMN_DD.py`: Decoder-guided Encoder training and inverse inference

The active inverse workflow is not end-to-end joint training. `BMN_DD.py` assumes a pre-trained Decoder checkpoint already exists, loads it in frozen mode, and then trains only the Encoder.

## BMN Formulation

The variables are split into:

- known conditions `c = [Dx, ht]`
- unknown global parameters `mu = [Us, Ub, p]`
- arc-length coordinate `s`
- full response `y(s) = [x, z, theta, T, M]`
- sparse observation `o = [x_top, z_top, T_i, M_i, theta_i, ...]`

The current model roles are:

- Decoder: `(s, c, mu) -> y(s)`
- Encoder: `o -> mu_hat`
- Full BMN inverse reconstruction: `o -> Encoder -> mu_hat -> Decoder -> y_hat`

In `BMN_DD.py`, the Decoder is wrapped as a frozen differentiable forward model. It can provide observation-consistency gradients, but its parameters are not updated during Encoder training.

## Repository Structure

Main files:

- `Decoder_DD.py`
- `BMN_DD.py`
- `scr_exact_bvp_solver.py`
- `evaluate_decoder_test_exact_cases.py`
- `evaluate_bmn_random_exact_cases.py`
- `para_config.json`
- `physics_config.json`

Default outputs are now placed under:

- `outputs/`

Important default output subfolders:

- `outputs/BMN_SCR_DD_outputs/`: trained checkpoints, datasets, histories, metrics
- `outputs/decoder_test_exact_comparison/`: Decoder vs exact test-case figures
- `outputs/bmn_random_exact_comparison/`: BMN inverse vs exact random-case figures

## Physical Meaning

The exact solver and data-driven models use the following default physical quantities:

- geometry:
  - `x(s)`, `z(s)`
- internal response:
  - `theta(s)`, `T(s)`, `M(s)`
- known top conditions:
  - `Dx`: top horizontal offset
  - `ht`: top height parameter
- current-profile parameters:
  - `Us`: surface current velocity
  - `Ub`: bottom current velocity
  - `p`: current-profile exponent

## Configuration Files

The main configuration is now split into two files:

- `para_config.json`: ranges, dataset settings, network settings, training settings, output paths
- `physics_config.json`: physical constants shared by the exact solver, Decoder data generation, and BMN inverse workflow

Both files are standard JSON files and should be edited without comments.

`physics_config.json` stores the `physical` block only. It is loaded through `para_config.json`, then merged into the runtime `cfg.physical` object used by:

- exact-case generation in `Decoder_DD.py`
- frozen-Decoder Encoder-data generation in `BMN_DD.py`
- exact-solver-based evaluation scripts

`para_config.json` points to `physics_config.json` through:

```text
"physics_config": "physics_config.json"
```

## Default Parameter Ranges

The current default ranges in `para_config.json` and `Decoder_DD.py` are:

```text
Us in [0.5, 2.5]
Ub in [0.0, 0.8]
p  in [1/7, 1/3] = [0.142857..., 0.333333...]
Dx in [1700.0, 1900.0] m
ht in [-10.0, 10.0] m
```

Sampling constraints:

- `Us >= Ub`
- geometry admissibility must pass the built-in screen in `sample_one_case(...)`

## Default Physical Constants

Current default physical settings from `physics_config.json`:

```text
D_o = 0.4064 m
t = 0.0254 m
E_steel = 2.1e11 Pa
rho_s = 7850 kg/m^3
rho_w = 1025 kg/m^3
g = 9.81 m/s^2
C_d = 1.2
L = 2500 m
water_depth = 1000 m
x_bottom = 0.0 m
k_b = 5.0e3
```

## Default Dataset and Training Settings

Current defaults from `para_config.json`:

### Dataset

```text
decoder_n_cases = 200
encoder_n_cases = 20000
n_nodes = 256
seed = 42
train_fraction = 0.8
val_fraction = 0.1
output_vars = [x, z, theta, T, M]
observation_vars = [T, M, theta]
n_default_sensors = 6
output_dir = outputs/BMN_SCR_DD_outputs
```

If `sensor_indices` and `sensor_s` are left empty, internal sensors are generated automatically.

### Decoder Network

```text
hidden_dim = 256
num_hidden_layers = 5
activation = tanh
dropout = 0.0
```

### Decoder Training

```text
epochs = 2000
batch_size = 8192
lr = 1e-3
weight_decay = 0.0
grad_clip = 1.0
patience = 300
device = cuda
```

### Encoder Network

```text
architecture = bounded_mlp
hidden_dim = 256
num_hidden_layers = 4
activation = gelu
dropout = 0.0
use_layer_norm = true
bounded_output = true
```

### Encoder Training

```text
epochs = 1500
batch_size = 128
lr = 1e-3
weight_decay = 0.0
grad_clip = 1.0
patience = 300
lambda_observation = 0.0
lambda_order = 0.0
device = cuda
```

## Decoder Workflow

`Decoder_DD.py` is responsible for:

1. sampling valid parameter cases
2. calling the exact solver
3. generating `decoder_fullfield_dataset.npz`
4. training a pure data-supervised Decoder
5. optionally extracting an Encoder dataset from the exact-response dataset

Core mapping:

```text
[s, Dx, ht, Us, Ub, p] -> [x, z, theta, T, M]
```

The Decoder is trained on scaled pointwise data assembled from full-field cases.

## Encoder / BMN Workflow

`BMN_DD.py` currently supports two Encoder-data routes:

1. exact-dataset-derived observation data
2. frozen-Decoder-generated observation data

The active default path is:

```text
sample (c, mu)
-> frozen Decoder
-> full response y_D
-> sparse observation o_D
-> Encoder training pair (o_D, mu)
```

Encoder training loss is:

```text
L = L_mu + lambda_observation * L_obs + lambda_order * L_order
```

where:

- `L_mu`: parameter supervision loss on `mu`
- `L_obs`: observation consistency through the frozen Decoder
- `L_order`: soft penalty for non-physical `Ub > Us`

Important implementation note:

- `BMN_DD.py` does not train `Decoder_DD`
- it only reads a trained Decoder checkpoint
- if the Decoder checkpoint does not exist, training stops with an error

## Typical Usage

### 1. Train Decoder

Generate exact-solver Decoder data and train the Decoder:

```bash
python Decoder_DD.py --mode all
```

Or split into steps:

```bash
python Decoder_DD.py --mode generate
python Decoder_DD.py --mode train_decoder
```

This produces files under:

- `outputs/BMN_SCR_DD_outputs/`

especially:

- `decoder_fullfield_dataset.npz`
- `Decoder_DD_model.pth`
- `Decoder_DD_history.json`

### 2. Build Encoder Data

Using the frozen Decoder:

```bash
python BMN_DD.py --mode build_data
```

This creates:

- `outputs/BMN_SCR_DD_outputs/bmn_encoder_dataset.npz`

### 3. Train Encoder

```bash
python BMN_DD.py --mode train
```

Or build data and train in one shot:

```bash
python BMN_DD.py --mode all
```

This produces:

- `BMN_DD_encoder.pth`
- `BMN_DD_history.json`
- `BMN_DD_test_metrics.json`

all under:

- `outputs/BMN_SCR_DD_outputs/`

## Evaluation Scripts

### Decoder vs Exact on held-out test set

```bash
python evaluate_decoder_test_exact_cases.py
```

Default output:

- `outputs/decoder_test_exact_comparison/`

### BMN inverse vs Exact on random cases

```bash
python evaluate_bmn_random_exact_cases.py --n_cases 50 --device cpu
```

This script:

- samples 50 random valid cases
- solves them with the exact solver
- extracts observations using the trained Encoder's sensor layout
- runs `Encoder -> Decoder`
- saves one 4-panel figure per case
- writes `evaluation_summary.json`

Default output:

- `outputs/bmn_random_exact_comparison/`

## Config Notes

The main configuration lives in `para_config.json`, and physical constants live in `physics_config.json`.

Two dataset-size fields are intentionally separated:

- `decoder_n_cases`: exact-solver cases used for Decoder dataset generation
- `encoder_n_cases`: cases used for Encoder dataset generation

The legacy field `n_cases` is kept only for backward compatibility and should not be used as the main control anymore.

## Current Status

With the current default setup:

- Decoder and Encoder are trained in two stages
- Encoder training is stable with large enough `encoder_n_cases`
- `bounded_mlp` is active for Encoder training
- all default outputs are centralized under `outputs/`

## Limitations

- the main BMN path is still based on a frozen Decoder, not joint optimization
- exact-solver success still depends on sampled-case feasibility
- current observation layout is still configuration-driven, not automatically optimized
- robustness to observation noise is not yet a first-class training feature

## Quick Start

If you only want the shortest working path:

```bash
python Decoder_DD.py --mode all
python BMN_DD.py --mode all
python evaluate_bmn_random_exact_cases.py --n_cases 50 --device cpu
```

That will give you:

- a trained Decoder
- a trained Encoder
- random exact-case BMN-vs-exact comparison figures
