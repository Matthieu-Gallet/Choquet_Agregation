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
- [Results and Analysis](#results-and-analysis)
- [Citation](#citation)

## Overview

This project implements a comprehensive framework for binary classification using Choquet integral aggregation to combine predictions from ensemble classifiers. The Choquet integral is a powerful aggregation operator from fuzzy measure theory that can model interactions between classifiers.

The framework is designed for SAR (Synthetic Aperture Radar) image classification in cryosphere studies, particularly for glacier surface type classification. It includes ensemble training with multiple classifiers (Random Forest, SVM, MLP, etc.), various Choquet integral variants (Power-based, Weight-based, T-norm), automated parameter sweeps, parallel execution, and comprehensive analysis tools for generating plots and LaTeX tables. The system ensures reproducibility through seed-based experiments and can be applied to any binary classification problem with group-structured data.

## Dataset

This project uses the CPAZMaL dataset (Cryosphere PAZ MArine ice and Land ice dataset):

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

The dataset contains SAR images from PAZ and TerraSAR-X satellites in HDF5 format, with multiple glacier surface classes (ABL, ACC, FOR, PLA, ROC), dual polarization (HH, HV), temporal acquisitions with metadata, and group-based organization for proper train/test splitting. The dataset is available on [Hugging Face](https://huggingface.co/datasets/musmb/CPAZMaL).

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
│   │   ├── main_config.yaml      # Base configuration template
│   │   ├── sweep_samples/        # Max samples sweep configs
│   │   │   ├── ABL_ACC.yaml
│   │   │   ├── FOR_PLA.yaml
│   │   │   └── ROC_PLA.yaml
│   │   └── sweep_window/         # Window size sweep configs
│   │       ├── ABL_ACC.yaml
│   │       ├── FOR_PLA.yaml
│   │       └── ROC_PLA.yaml
│   │
│   ├── dataset/                  # Data loading and preprocessing
│   │   ├── load_dataset.py       # HDF5 dataset loader
│   │   └── dataloader.py         # Binary classification data preparation
│   │
│   ├── learning/                 # Training and learning algorithms
│   │   ├── train_ensemble.py     # Ensemble classifier training
│   │   ├── train_aggregate.py    # Choquet aggregation training
│   │   ├── choquet_learnable.py  # Choquet integral classifiers
│   │   └── optimize/             # Optimization algorithms
│   │       ├── gradient_descent.py
│   │       └── objective_functions.py
│   │
│   ├── fuzzy_measure/            # Fuzzy measure computations
│   │   ├── classical.py          # Power and Weight measures
│   │   └── tnorm.py              # T-norm based measures
│   │
│   ├── evaluation/               # Results analysis and visualization
│   │   ├── aggregate_results.py  # Results aggregation
│   │   └── analyze_sweeps.py     # Sweep analysis and plotting
│   │
│   ├── results/                  # Experimental results (generated)
│   │   └── {config_hash}/        # Results by configuration
│   │       ├── config.yaml
│   │       ├── choquet_aggregation_results.csv
│   │       ├── ensemble_scores.csv
│   │       └── sweep_*/          # Sweep subdirectories
│   │           └── seed_*/       # Per-seed results
│   │
│   └── figures/                  # Generated plots and tables (generated)
│       └── {class_pair}/
│           ├── sweep_window_size.pdf
│           ├── sweep_max_samples.pdf
│           ├── table_window_size_*.tex
│           └── table_max_samples_*.tex
```

## Installation

### Automated Setup

```bash
# Run the automated setup script
./setup_env.sh
```

This script will:
1. Install `uv` (fast Python package installer) if not present
2. Create a virtual environment named `.venv_ChoquetLearning`
3. Install all required Python packages
4. Test imports and auto-install missing dependencies

### Manual Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate Choquet_Agregation

# Or use pip
pip install -r requirements_minimal.txt
```

### Requirements

- Python 3.12+
- Main dependencies:
  - `numpy >= 1.24.0`
  - `pandas >= 2.0.0`
  - `scikit-learn >= 1.3.0`
  - `h5py >= 3.8.0`
  - `matplotlib >= 3.7.0`
  - `pyyaml >= 6.0`
  - `tqdm >= 4.65.0`

## Quick Start

### 1. Prepare Your Data

Download the CPAZMaL dataset and note its path:
```bash
# Example path
DATA_PATH="/path/to/CPAZMaL.hdf5"
```

### 2. Run the Interactive Pipeline

```bash
./run_pipeline.sh --data-path $DATA_PATH
```

Select from the menu:
```
Options:
  1. Run samples sweep (max_samples_per_class)
  2. Run window sweep (window_size)
  3. Run all sweeps (samples + window)
  4. Analyze existing results
  5. Exit
```

### 3. View Results

Results are automatically saved and analyzed:
- **CSV files**: `src/results/{hash}/choquet_aggregation_results.csv`
- **Plots**: `src/figures/{class_pair}/sweep_*.pdf`
- **LaTeX tables**: `src/figures/{class_pair}/table_*.tex`

## Usage

### Run Pipeline Script

The main entry point is `run_pipeline.sh`:

```bash
./run_pipeline.sh [OPTIONS]
```

**Options:**
- `--data-path PATH`: Path to HDF5 dataset file (required)
- `--verbose`: Enable detailed output (default: quiet mode)

**Modes:**

1. **Samples Sweep**: Tests different maximum samples per class (1 to 1000)
   - Evaluates model performance with varying training set sizes
   - Useful for understanding data efficiency

2. **Window Sweep**: Tests different spatial window sizes (1 to 56)
   - Evaluates impact of spatial context on classification
   - Critical for SAR image analysis

3. **All Sweeps**: Runs both sweeps sequentially
   - Complete parameter exploration
   - Recommended for comprehensive experiments

4. **Analyze Results**: Generate plots and tables from existing results
   - Re-analyze without re-running experiments
   - Customize visualization parameters

### Command Line Examples

```bash
# Basic usage with default settings
./run_pipeline.sh --data-path /data/CPAZMaL.hdf5

# Verbose output for debugging
./run_pipeline.sh --data-path /data/CPAZMaL.hdf5 --verbose

# Run programmatically (non-interactive)
python src/main_sweep.py --mode samples --n_jobs 10 --data-path /data/CPAZMaL.hdf5

# Analyze specific results
python src/evaluation/analyze_sweeps.py \
    --results_dir src/results \
    --figures_dir src/figures \
    --window_size 7 \
    --max_samples 46
```

### Python API

```python
from src.dataset.load_dataset import MLDatasetLoader
from src.dataset.dataloader import load_and_extract_data, prepare_train_test_split
from src.learning.train_ensemble import main as train_ensemble
from src.learning.train_aggregate import main as train_aggregate

# Load dataset
loader = MLDatasetLoader('data/CPAZMaL.hdf5')

# Extract data for binary classification
X, y, groups, class_names, group_names = load_and_extract_data(
    loader=loader,
    class_pair=('ABL', 'ACC'),
    date='20200804',
    window_size=11,
    verbose=True
)

# Prepare train/test split
data = prepare_train_test_split(
    X=X, y=y, groups=groups,
    class_names=class_names,
    group_names=group_names,
    seed=42,
    max_samples_per_class=100,
    n_splits=4,
    verbose=True
)

# Train ensemble and aggregation
train_ensemble(config_path='src/config/sweep_samples/ABL_ACC.yaml', n_jobs=10)
train_aggregate(config_path='src/config/sweep_samples/ABL_ACC.yaml', n_jobs=10)
```

## Choquet Aggregation Methods

The framework implements several Choquet integral variants:

### Classical Methods

1. **Choquet_Power**: Power-based fuzzy measure
   - Emphasizes majority agreement
   - Parameters learned via L-BFGS-B

2. **Choquet_Weight**: Weight-based fuzzy measure
   - Linear combination with interactions
   - Parameters learned via L-BFGS-B

### T-norm Methods

3. **Choquet_Power_TN3**: Power with Hamacher T-norm (c=3)
   - Non-linear interactions between sources
   - Gradient descent optimization

4. **Choquet_Power_TN6**: Power with Hamacher T-norm (c=6)
   - Stronger interactions
   - Gradient descent optimization

5. **Choquet_Weight_TN3**: Weight with Dubois-Prade T-norm (c=3)
   - Weighted interactions
   - Gradient descent optimization

6. **Choquet_Weight_TN6**: Weight with Dubois-Prade T-norm (c=6)
   - Enhanced weighted interactions
   - Gradient descent optimization

### Baseline

7. **LogisticRegression**: Standard logistic regression baseline
   - No penalty, 1000 max iterations

### Comparison

All methods are automatically compared on:
- **Accuracy**: Classification accuracy
- **F1 Score**: Weighted F1 score (primary metric)
- **AUC-ROC**: Area under ROC curve

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

## Results and Analysis

### Sweep Results

After running sweeps, results are organized as:

```
results/{config_hash}/
├── config.yaml                          # Experiment configuration
├── choquet_aggregation_results.csv      # Main results file
├── ensemble_scores.csv                  # Base classifier scores
└── sweep_{param_name}_{value}/          # Per-parameter value
    └── seed_{XX}/                       # Per-seed subdirectory
        ├── train.npy                    # Training predictions
        ├── test.npy                     # Test predictions
        └── scores.csv                   # Classifier scores
```

### Visualization

The `analyze_sweeps.py` script automatically generates:

1. **Boxplots** (PDF): F1 score distribution across parameter values
   - Grouped by aggregation method
   - Color-coded (orange: Power, blue: Weight, black: baseline)

2. **LaTeX Tables**: Comparison at specific parameter values
   - Mean ± std for each method
   - Bold formatting for best results
   - Ready for publication

### Metrics

Primary metrics reported:
- **Test F1 Score**: Primary performance metric
- **Test Accuracy**: Classification accuracy
- **Test AUC-ROC**: Area under ROC curve
- **Train metrics**: For overfitting analysis

## Advanced Usage

### Quiet Mode (Default)

By default, the pipeline runs in quiet mode showing only progress bars:

```bash
./run_pipeline.sh --data-path $DATA_PATH
```

Output:
```
Data extracted: 4663 samples
Calcul en cours...
Seeds (max_samples_per_class=21): 100%|████████| 10/10 [02:15<00:00]
```

### Verbose Mode

Enable detailed logging with `--verbose`:

```bash
./run_pipeline.sh --data-path $DATA_PATH --verbose
```

Output includes:
```
[DATA] Binary classification: ABL (0) vs ACC (1)
[DATA] Parameters: date=20200804, orbit=DSC, window=11x11
[TRAIN] Preparing split (seed=0, max_samples=21)
[SPLIT] Selected fold 2 randomly from 4 splits
[FILTER] Class 0: 1585 -> 21 samples
...
```

### Custom Parallelization

Adjust the number of parallel jobs in `run_pipeline.sh`:

```bash
# Edit line 85
N_JOBS=10  # Set to number of cores

# Or run Python directly
python src/main_sweep.py --mode all --n_jobs 20 --data-path $DATA_PATH
```

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

### Memory Issues

Reduce parallelization or window size:
```bash
# In config YAML
window_size: 7  # Instead of 11
n_splits: 3     # Instead of 4
```

### CUDA/GPU Warnings

This project uses CPU-only scikit-learn. GPU warnings can be ignored or suppressed.

## Citation

If you use this code or the CPAZMaL dataset, please cite:

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

## License

This project is available under the MIT License. See LICENSE file for details.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, commit your changes, push to the branch, and open a Pull Request.

## Contact

For questions or issues, please open an issue on the repository or contact the authors via the CPAZMaL dataset page.

## Acknowledgments

- CPAZMaL dataset provided by Hugging Face
- Choquet integral implementations based on fuzzy measure theory
- SAR imagery from PAZ and TerraSAR-X satellites
