#!/bin/bash
# Quick start script for Choquet Aggregation pipeline

set -e  # Exit on error

echo "=========================================="
echo "Choquet Aggregation - Quick Start"
echo "=========================================="
echo ""

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
echo "Ready to run pipeline!"
echo "=========================================="
echo ""
echo "Options:"
echo "  1. Complete pipeline (ensemble + aggregation)"
echo "  2. Ensemble training only"
echo "  3. Aggregation only (use existing results)"
echo "  4. Exit"
echo ""

read -p "Choose option (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Running complete pipeline..."
        python main.py --config config/config.yaml --n_jobs -1
        ;;
    2)
        echo ""
        echo "Running ensemble training..."
        python -m learning.train_ensemble --config config/config.yaml --n_jobs 2
        ;;
    3)
        read -p "Enter results directory (or press Enter for auto-detect): " results_dir
        if [ -z "$results_dir" ]; then
            echo ""
            echo "Running aggregation (auto-detect results)..."
            python -m learning.train_aggregate --config config/config.yaml --n_jobs -1
        else
            echo ""
            echo "Running aggregation on: $results_dir"
            python -m learning.train_aggregate --config config/config.yaml --results_dir "$results_dir" --n_jobs -1
        fi
        ;;
    4)
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
echo "Results are in: ../results/"
echo ""
echo "To visualize results:"
echo "  cd src"
echo "  python -m evaluation.plot_results \\"
echo "    --csv ../results/{hash}/choquet_aggregation_results.csv \\"
echo "    --output_dir ../figures/ \\"
echo "    --metric test_f1"
echo ""
