#!/bin/bash
# Pipeline script for Choquet Aggregation sweep experiments

set -e  # Exit on error

# Parse command line arguments
DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--data-path /path/to/data.hdf5]"
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

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please create it first:"
    echo "  python -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Test imports
echo ""
echo "Testing module imports..."
cd src
python test_imports.py
TEST_RESULT=$?

if [ $TEST_RESULT -ne 0 ]; then
    echo ""
    echo "Import test failed! Please fix the issues before continuing."
    exit 1
fi

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
N_JOBS=10
WINDOW_SIZE=7
MAX_SAMPLES=46

case $choice in
    1)
        echo ""
        echo "Running samples sweep..."
        if [ -n "$DATA_PATH" ]; then
            python src/main_sweep.py --mode samples --n_jobs $N_JOBS --data-path "$DATA_PATH"
        else
            python src/main_sweep.py --mode samples --n_jobs $N_JOBS
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
            python src/main_sweep.py --mode window --n_jobs $N_JOBS --data-path "$DATA_PATH"
        else
            python src/main_sweep.py --mode window --n_jobs $N_JOBS
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
            python src/main_sweep.py --mode all --n_jobs $N_JOBS --data-path "$DATA_PATH"
        else
            python src/main_sweep.py --mode all --n_jobs $N_JOBS
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
