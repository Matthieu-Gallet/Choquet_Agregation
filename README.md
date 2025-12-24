# Choquet Aggregation for Binary Classification

A machine learning framework for binary classification using Choquet integral-based aggregation methods on ensemble classifiers, with applications to cryosphere SAR image classification.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Choquet Aggregation Methods](#choquet-aggregation-methods)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)

## Overview

This project implements a framework for binary classification using Choquet integral aggregation to combine predictions from ensemble classifiers. The Choquet integral models interactions between classifiers using fuzzy measure theory.

Designed for SAR image classification in cryosphere studies, it includes ensemble training, various Choquet variants, automated parameter sweeps, parallel execution, and analysis tools for plots and LaTeX tables. Ensures reproducibility through seed-based experiments.

## Dataset

Uses the CPAZMaL dataset (Cryosphere PAZ Machine Learning dataset):

```bibtex
@misc{matthieu_gallet_2025,
    author       = { Matthieu Gallet and Christophe Lin-Kwong-Chon and
                     Suvrat Kaushik and Emmanuel Trouvé },
    title        = { CPAZMaL (Revision 885a72f) },
    year         = 2025,
    url          = { https://huggingface.co/datasets/musmb/CPAZMaL },
    doi          = { 10.57967/hf/7306 },
    publisher    = { Hugging Face }
}
```

Contains SAR images from PAZ satellite in HDF5 format, with glacier surface classes (ABL, ACC, FOR, PLA, ROC), dual polarization, and group-based organization for proper train/test splitting.

## Project Structure

```
Choquet_Agregation/
├── README.md                      # This file
├── environment.yml                # Conda environment specification
├── requirements_minimal.txt       # Python dependencies
├── setup_env.sh                   # Automated environment setup script
├── run_pipeline.sh               # Main pipeline orchestrator
│
├── src/                          # Source code
│   ├── main.py                   # Single experiment runner
│   ├── main_sweep.py             # Sweep experiment runner
│   ├── test_imports.py           # Module import verification
│   │
│   ├── config/                   # Configuration files
│   │   ├── sweep_samples/        # Max samples sweep configs
│   │   └── sweep_noise/          # Noise sweep configs
│   │
│   ├── dataset/                  # Data loading and preprocessing
│   ├── learning/                 # Training and learning algorithms
│   ├── fuzzy_measure/            # Fuzzy measure computations
│   ├── evaluation/               # Results analysis and visualization
│   │
│   ├── results/                  # Experimental results (generated)
│   └── figures/                  # Generated plots and tables (generated)
```

## Installation

### Automated Setup

```bash
./setup_env.sh
```

Creates virtual environment, installs dependencies, and tests imports. Use uv for quick setup.

### Manual Setup

```bash
pip install -r requirements_minimal.txt
```

**Requirements:** Python 3.12+, numpy, pandas, scikit-learn, h5py, matplotlib, pyyaml, tqdm.

## Quick Start

1. Download CPAZMaL dataset from Hugging Face
2. Run interactive pipeline:
   ```bash
   ./run_pipeline.sh --data-path /path/to/CPAZMaL.hdf5
   ```
3. Select option from menu (1-5)

Results saved in `src/results/` and `src/figures/`.

## Usage

### Run Pipeline Script

```bash
./run_pipeline.sh [OPTIONS]
```

**Options:**
- `--data-path PATH`: Path to HDF5 dataset (required)
- `--verbose`: Enable detailed output

**Modes:**
1. **Samples Sweep**: Tests different max samples per class (1-1000)
2. **Noise Sweep**: Tests different data noise std (0.0-2.5)
3. **All Sweeps**: Runs both sweeps sequentially
4. **Analyze Results**: Generate plots and tables from existing results
5. **Exit**

### Command Line Examples

```bash
# Basic usage
./run_pipeline.sh --data-path /data/CPAZMaL.hdf5

# Verbose output
./run_pipeline.sh --data-path /data/CPAZMaL.hdf5 --verbose

# Programmatic run
python src/main_sweep.py --mode samples --n_jobs 10 --data-path /data/CPAZMaL.hdf5
```
The option 4. **Analyze Results** in `run_pipeline.sh` generates plots and tables from existing results without re-running experiments.

### Command Line Examples

```bash
# Basic usage with default settings
./run_pipeline.sh --data-path /data/CPAZMaL.hdf5

# Verbose output for debugging
./run_pipeline.sh --data-path /data/CPAZMaL.hdf5 --verbose

# Run programmatically (non-interactive)
python src/main_sweep.py --mode samples --n_jobs 10 --data-path /data/CPAZMaL.hdf5

## Choquet Aggregation Methods
(cf. Associative Functions: Triangular Norms and Copulas by Alsina et al. 2006):
### Classical Methods
1. **Choquet_Power**: Power-based fuzzy measure
2. **Choquet_Weight**: Weight-based fuzzy measure

### T-norm Methods
3. **Choquet_Power_TN3/TN6**: Power with T-norm (3/6)
4. **Choquet_Weight_TN3/TN6**: Weight with T-norm (3/6)

### Baseline
5. **LogisticRegression**: Standard logistic regression

All methods compared on Accuracy, F1 Score, and AUC-ROC.

## Configuration

Configuration files use YAML format with the following structure:

```yaml
dataset:
  data_path: /path/to/CPAZMaL.hdf5
  class_pair: [ABL, ACC]          # Binary classification pair
  date: '20200804'                 # Acquisition date
  window_size: 11                  # Spatial window size (11x11)
  orbit: DSC                       # Descending orbit
  polarisation: [HH, HV]           # Dual polarization
  scale_type: amplitude            # Scaling: intensity, amplitude, log10
  n_splits: 4                      # Number of K-folds
  skip_optim_offset: true
  max_samples_per_class: null      # null for sweep, int for fixed

sweep:
  param_name: max_samples_per_class  # or window_size
  values: [1, 2, 5, 10, 21, 46, 100, 215, 464, 1000]

training:
  n_seeds: 10                      # Number of random seeds
  save_scores: true

noise:
  label_noise_percentage: 0.0      # Label noise (0-100%)
  data_noise_std: 0.0              # Gaussian noise std

models:                            # Ensemble classifiers
  - name: RandomForest_100
    type: RandomForestClassifier
    params:
      n_estimators: 100
      max_depth: 10
  - name: SVM_RBF
    type: SVC
    params:
      kernel: rbf
      probability: true
  # ... more models

choquet:                           # Choquet configuration
  optimizer_weight: L-BFGS-B
  optimizer_power: L-BFGS-B
  optimizer_tnorm: GD
  niter: 1000
  stpa_weight: 0.005
  stpa_power: 0.001
```

## Results

Experimental results are provided in the `results/` directory. To regenerate plots and tables:

- **Option 4** in `run_pipeline.sh`: Analyze existing results and regenerate all plots/tables
- **Option 3** in `run_pipeline.sh`: Run complete pipeline (~6 × 40min on standard hardware)


## Advanced Usage

### Verbose Mode

Enable detailed logging with `--verbose`:

```bash
./run_pipeline.sh --data-path $DATA_PATH --verbose
```

### Custom Parallelization

Adjust `N_JOBS` in `run_pipeline.sh` for parallel processing.

### Skip Aggregation

Train only ensemble classifiers (faster):

```bash
python src/main_sweep.py --mode samples --n_jobs 10 --skip-aggregation
```

## Troubleshooting

### Import Errors

Test module imports:
```bash
cd src
python test_imports.py
```


## License

This project is available under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, commit your changes, push to the branch, and open a Pull Request.

## Contact

For questions or issues, please open an issue on the repository or contact the authors via the CPAZMaL dataset page.

