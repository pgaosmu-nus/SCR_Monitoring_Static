# BMN-SCR-DD: Static SCR Monitoring via Encoder-Decoder Pipeline

This repository implements a two-stage **BMN (Bounded-parameter Neural Network)** inverse method for static Subsea Cable Route (SCR) monitoring and inversion.

## BMN Method Overview

The BMN-SCR-DD pipeline consists of:

1. **Decoder**: A data-driven forward model mapping parameters and conditions to full-field SCR response
   - Trained on exact-solver-generated data
   - Frozen during Encoder training
2. **Encoder**: A neural network mapping sparse observations to global parameters
   - Trained with the frozen Decoder as a differentiable constraint
3. **Full Inverse**: `Observations -> Encoder -> Parameters -> Frozen Decoder -> Reconstructed Response`

**Key workflow**:
1. Train a data-driven `Decoder` on exact solver outputs
2. Freeze the Decoder 
3. Train an `Encoder` using frozen Decoder constraints
4. Reconstruct full SCR response via `Encoder -> Decoder`

The workflow is **not end-to-end joint training**. `BMN_DD.py` assumes a pre-trained Decoder checkpoint exists, loads it in frozen mode, and trains only the Encoder for improved efficiency and stability.

## BMN Formulation

### Variables

- **Known boundary conditions**: `c = [Dx, ht]`
  - `Dx`: top horizontal offset
  - `ht`: top height parameter
- **Unknown global parameters**: `μ = [Us, Ub, p]`
  - `Us`: surface current velocity
  - `Ub`: bottom current velocity  
  - `p`: current-profile exponent (power-law)
- **Arc-length coordinate**: `s`
- **Full response**: `y(s) = [x, z, θ, T, M]`
  - `x(s)`, `z(s)`: geometry
  - `θ(s)`: internal angle
  - `T(s)`: tension
  - `M(s)`: bending moment
- **Sparse observations**: `o = [x_top, z_top, T_i, M_i, θ_i, ...]`

### Architecture

- **Decoder**: Maps `(s, c, μ) → y(s)` — pure data-driven forward model
- **Encoder**: Maps `o → μ̂` — parameter estimator  
- **Full inverse**: `o → Encoder → μ̂ → Decoder → ŷ`

### Training Strategy

The Decoder is wrapped as a **frozen differentiable forward model** in `BMN_DD.py`:
- Provides observation-consistency gradients during Encoder training
- Parameters remain fixed (not updated)
- Ensures physically consistent reconstructions

## Core Files

- **[Decoder_DD.py](Decoder_DD.py)**: Decoder training pipeline
  - Samples valid parameter cases and generates exact-solver data
  - Trains pure data-driven Decoder on full-field responses
  - Generates Encoder dataset from exact responses

- **[BMN_DD.py](BMN_DD.py)**: Encoder training and inverse inference  
  - Uses frozen Decoder as forward model constraint
  - Trains Encoder to map sparse observations to parameters
  - Computes parameter-supervision and observation-consistency losses

- **[scr_exact_bvp_solver.py](scr_exact_bvp_solver.py)**: Exact static SCR boundary-value solver
  - Reference physics implementation for generating training data
  - Used for validation and test-set evaluation

- **[evaluate_bmn_random_exact_cases.py](evaluate_bmn_random_exact_cases.py)**: BMN inverse evaluation
  - Compares BMN predictions to exact solver on random cases
  - Generates comparison figures and metrics

- **[evaluate_decoder_test_exact_cases.py](evaluate_decoder_test_exact_cases.py)**: Decoder forward validation
  - Tests Decoder predictions against exact solver on held-out test set
  - Quantifies Decoder approximation error

- **[DecoderOptimizationInversion.py](DecoderOptimizationInversion.py)**: Alternative baseline method
  - Joint end-to-end optimization of Decoder and Encoder
  - Provides comparison point for BMN two-stage approach
  - Useful for benchmarking and method evaluation

- **[para_config.json](para_config.json)**: Main configuration file
  - Dataset sizes, network architectures, training hyperparameters
  - Output paths and logging settings
  - References physics constants from `physics_config.json`

- **[physics_config.json](physics_config.json)**: Physical constants
  - Pipe, water, and environmental parameters
  - Shared by exact solver, Decoder, and Encoder

## Configuration

### para_config.json

Main runtime configuration controlling:
- `decoder_n_cases`: Number of exact-solver cases for Decoder data
- `encoder_n_cases`: Number of cases for Encoder data  
- `n_nodes`: Number of arc-length discretization points
- **Network architectures**: `hidden_dim`, `num_hidden_layers`, activation functions
- **Training**: epochs, batch size, learning rate, patience, regularization
- **Device**: CUDA or CPU
- **Output paths**: `output_dir = outputs/BMN_SCR_DD_outputs`

Two dataset sizes are **intentionally separated**:
- `decoder_n_cases`: Controls Decoder training data volume
- `encoder_n_cases`: Controls Encoder training data volume

### physics_config.json

Physical constants shared across the pipeline:
- **Pipe**: diameter, thickness, steel properties
- **Water**: density, depth, drag coefficient
- **Environment**: gravity, cable length, bottom friction
- **Geometry**: bottom position reference

## Typical Workflow

### Step 1: Train Decoder

Generate exact-solver data and train the Decoder:

```bash
python Decoder_DD.py --mode all
```

Alternatively, split into steps:
```bash
python Decoder_DD.py --mode generate  # Generate exact-solver cases
python Decoder_DD.py --mode train_decoder  # Train Decoder
```

**Outputs**:
- `outputs/BMN_SCR_DD_outputs/decoder_fullfield_dataset.npz` — Decoder training data
- `outputs/BMN_SCR_DD_outputs/Decoder_DD_model.pth` — Trained Decoder weights
- `outputs/BMN_SCR_DD_outputs/Decoder_DD_history.json` — Training history

### Step 2: Build Encoder Data

Using the frozen Decoder:

```bash
python BMN_DD.py --mode build_data
```

**Output**:
- `outputs/BMN_SCR_DD_outputs/bmn_encoder_dataset.npz` — Encoder training pairs (observations, parameters)

### Step 3: Train Encoder

```bash
python BMN_DD.py --mode train
```

Or combine steps 2 and 3:
```bash
python BMN_DD.py --mode all
```

**Outputs**:
- `outputs/BMN_SCR_DD_outputs/BMN_DD_encoder.pth` — Trained Encoder weights
- `outputs/BMN_SCR_DD_outputs/BMN_DD_history.json` — Training history  
- `outputs/BMN_SCR_DD_outputs/BMN_DD_test_metrics.json` — Test metrics

### Step 4: Evaluate on Test Cases

**Decoder validation** (Decoder vs exact forward):
```bash
python evaluate_decoder_test_exact_cases.py
```
Output: `outputs/decoder_test_exact_comparison/`

**BMN validation** (Full inverse vs exact):
```bash
python evaluate_bmn_random_exact_cases.py --n_cases 50 --device cpu
```
Output: `outputs/bmn_random_exact_comparison/`

## Encoder Training Loss

The Encoder is trained with a composite loss:

$$L = L_{\mu} + \lambda_{\text{obs}} \cdot L_{\text{obs}} + \lambda_{\text{order}} \cdot L_{\text{order}}$$

where:
- $L_{\mu}$: Parameter supervision loss on ground-truth parameters
- $L_{\text{obs}}$: Observation consistency through frozen Decoder
- $L_{\text{order}}$: Soft penalty for non-physical $U_b > U_s$

## Method Comparison

This repository includes a baseline method for comparison:

### Alternative: End-to-End Joint Optimization

**File**: [DecoderOptimizationInversion.py](DecoderOptimizationInversion.py)

A baseline approach that jointly optimizes Decoder and Encoder in an end-to-end fashion:
- Trains Decoder and Encoder simultaneously
- Direct comparison point for the two-stage BMN approach
- Useful for understanding tradeoffs between joint vs. two-stage training

**Comparison perspective**:
- BMN (two-stage, frozen Decoder): Better stability, lower computational load
- Joint optimization: Potentially higher accuracy, more complex tuning

## Advantages of Two-Stage BMN Approach

1. **Efficiency**: Decoder training is decoupled from Encoder training
2. **Stability**: Frozen Decoder provides consistent physical constraints
3. **Modularity**: Can replace either component independently
4. **Feasibility**: Reduces joint optimization complexity

## Key Implementation Notes

- `BMN_DD.py` **requires** a pre-trained Decoder checkpoint
  - Will stop with error if Decoder not found
  - Ensures reproducibility of inverse results
- The Decoder is wrapped as a **differentiable but frozen** forward model
- All training outputs centralized in `outputs/BMN_SCR_DD_outputs/`

## Quick Start

Run the complete pipeline in three commands:

```bash
# 1. Train Decoder (generates exact-solver data)
python Decoder_DD.py --mode all

# 2. Build Encoder data and train Encoder
python BMN_DD.py --mode all

# 3. Validate on random cases
python evaluate_bmn_random_exact_cases.py --n_cases 50 --device cpu
```

This produces:
- Trained Decoder and Encoder checkpoints
- Comparison figures and metrics
- Full outputs in `outputs/BMN_SCR_DD_outputs/`

## Output Structure

```
outputs/BMN_SCR_DD_outputs/
├── decoder_fullfield_dataset.npz      # Decoder training data
├── Decoder_DD_model.pth               # Trained Decoder
├── Decoder_DD_history.json            # Training history
├── bmn_encoder_dataset.npz            # Encoder training pairs
├── BMN_DD_encoder.pth                 # Trained Encoder
├── BMN_DD_history.json                # Encoder history
└── BMN_DD_test_metrics.json           # Test results

outputs/decoder_test_exact_comparison/ # Decoder validation figures
outputs/bmn_random_exact_comparison/   # BMN inverse validation figures
```

## Customization

Key configuration parameters in `para_config.json`:

| Parameter | Purpose |
|-----------|---------|
| `decoder_n_cases` | Number of exact-solver training cases for Decoder |
| `encoder_n_cases` | Number of cases for Encoder training |
| `hidden_dim` | Network hidden layer dimension |
| `num_hidden_layers` | Network depth |
| `epochs` | Training epochs |
| `batch_size` | Training batch size |
| `lr` | Learning rate |

Physical parameters can be adjusted in `physics_config.json`.

## Limitations & Future Work

**Current limitations**:
- Frozen Decoder approach (no joint optimization)
- Exact solver feasibility-dependent sampling

**Future improvements**:
- Automated optimal sensor placement
- Physics informed network, reducing data dependency
- Extension to dynamic scenarios
