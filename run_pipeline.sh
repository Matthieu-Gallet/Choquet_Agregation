#!/bin/bash
# Pipeline script for Choquet Aggregation sweep experiments

set -e  # Exit on error

# Parse command line arguments
DATA_PATH=""
VERBOSE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--data-path /path/to/data.hdf5] [--verbose]"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Choquet Aggregation - Sweep Pipeline"
echo "=========================================="
echo ""

if [ -n "$DATA_PATH" ]; then
    echo "Using custom data path: $DATA_PATH"
    echo ""
fi

# Run environment setup
echo "Checking environment setup..."
if [ -f "setup_env.sh" ]; then
    bash setup_env.sh
else
    echo "Warning: setup_env.sh not found, skipping automated setup"
    echo ""
fi

# Activate virtual environment
ENV_NAME="ChoquetLearning"
ENV_PATH=".venv_$ENV_NAME"

if [ -d "$ENV_PATH" ]; then
    echo "Activating environment '$ENV_NAME'..."
    source "$ENV_PATH/bin/activate"
elif [ -d ".venv" ]; then
    echo "Activating .venv environment..."
    source .venv/bin/activate
else
    echo "[✗] Error: No virtual environment found!"
    echo "Please run setup_env.sh first"
    exit 1
fi

echo ""
echo "=========================================="
echo ""
echo "=========================================="
echo "Ready to run sweep pipeline!"
echo "=========================================="
echo ""
echo "Options:"
echo "  1. Run samples sweep (max_samples_per_class)"
echo "  2. Run window sweep (window_size)"
echo "  3. Run all sweeps (samples + window)"
echo "  4. Analyze existing results"
echo "  5. Exit"
echo ""

read -p "Choose option (1-5): " choice

# Default parameters
N_JOBS=5
WINDOW_SIZE=7
MAX_SAMPLES=46

case $choice in
    1)
        echo ""
        echo "Running samples sweep..."
        if [ -n "$DATA_PATH" ]; then
            python src/main_sweep.py --mode samples --n_jobs $N_JOBS --data-path "$DATA_PATH" $VERBOSE
        else
            python src/main_sweep.py --mode samples --n_jobs $N_JOBS $VERBOSE
        fi
        
        echo ""
        echo "Analyzing results..."
        python src/evaluation/analyze_sweeps.py \
            --results_dir src/results \
            --figures_dir src/figures \
            --max_samples $MAX_SAMPLES
        ;;
    2)
        echo ""
        echo "Running window sweep..."
        if [ -n "$DATA_PATH" ]; then
            python src/main_sweep.py --mode window --n_jobs $N_JOBS --data-path "$DATA_PATH" $VERBOSE
        else
            python src/main_sweep.py --mode window --n_jobs $N_JOBS $VERBOSE
        fi
        
        echo ""
        echo "Analyzing results..."
        python src/evaluation/analyze_sweeps.py \
            --results_dir src/results \
            --figures_dir src/figures \
            --window_size $WINDOW_SIZE
        ;;
    3)
        echo ""
        echo "Running all sweeps..."
        if [ -n "$DATA_PATH" ]; then
            python src/main_sweep.py --mode all --n_jobs $N_JOBS --data-path "$DATA_PATH" $VERBOSE
        else
            python src/main_sweep.py --mode all --n_jobs $N_JOBS $VERBOSE
        fi
        
        echo ""
        echo "Analyzing results..."
        python src/evaluation/analyze_sweeps.py \
            --results_dir src/results \
            --figures_dir src/figures \
            --window_size $WINDOW_SIZE \
            --max_samples $MAX_SAMPLES
        ;;
    4)
        echo ""
        read -p "Enter window_size for tables (default: $WINDOW_SIZE): " ws
        read -p "Enter max_samples for tables (default: $MAX_SAMPLES): " ms
        
        WS=${ws:-$WINDOW_SIZE}
        MS=${ms:-$MAX_SAMPLES}
        
        echo "Analyzing results..."
        python src/evaluation/analyze_sweeps.py \
            --results_dir src/results \
            --figures_dir src/figures \
            --window_size $WS \
            --max_samples $MS
        ;;
    5)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid option!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Pipeline completed!"
echo "=========================================="
echo ""
echo "Results are in: src/results/"
echo "Figures are in: src/figures/"
echo ""
