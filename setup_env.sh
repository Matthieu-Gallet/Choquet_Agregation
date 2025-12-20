#!/bin/bash
# Complete environment setup script using uv for Choquet Aggregation project

set -e  # Exit on error

echo "=========================================="
echo "Choquet Learning - Environment Setup"
echo "=========================================="
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    echo ""
    echo "[✓] uv installed successfully"
else
    echo "[✓] uv is already installed"
fi

echo ""

# Environment name
ENV_NAME="ChoquetLearning"
ENV_PATH=".venv_$ENV_NAME"

# Check if environment exists
if [ -d "$ENV_PATH" ]; then
    echo "[✓] Virtual environment '$ENV_NAME' already exists"
else
    echo "Creating virtual environment '$ENV_NAME'..."
    uv venv "$ENV_PATH"
    echo "[✓] Virtual environment created"
fi

echo ""

# Activate environment
echo "Activating environment..."
source "$ENV_PATH/bin/activate"

echo ""

# Install base requirements
if [ -f "requirements_minimal.txt" ]; then
    echo "Installing base packages from requirements_minimal.txt..."
    uv pip install -r requirements_minimal.txt
    echo "[✓] Base packages installed"
else
    echo "Warning: requirements_minimal.txt not found"
fi

echo ""

# Test imports and install missing packages
echo "Testing imports..."
cd src

# Run test_imports.py and capture output
if python test_imports.py 2>&1 | tee /tmp/import_test.log; then
    echo "[✓] All imports successful"
else
    echo ""
    echo "Some imports failed. Analyzing missing packages..."
    echo ""
    
    # Extract missing packages from error messages
    MISSING_PACKAGES=$(grep -oP "No module named '\K[^']+(?=')" /tmp/import_test.log | sort -u || true)
    
    if [ -n "$MISSING_PACKAGES" ]; then
        echo "Missing packages detected:"
        echo "$MISSING_PACKAGES"
        echo ""
        echo "Installing missing packages..."
        
        for package in $MISSING_PACKAGES; do
            echo "  - Installing $package..."
            
            # Handle special package names
            case $package in
                sklearn)
                    uv pip install scikit-learn
                    ;;
                yaml)
                    uv pip install PyYAML
                    ;;
                *)
                    uv pip install "$package" || echo "Warning: Failed to install $package"
                    ;;
            esac
        done
        
        echo ""
        echo "Retesting imports..."
        if python test_imports.py; then
            echo "[✓] All imports now successful"
        else
            echo "[✗] Some imports still failing. Please check manually."
            cd ..
            exit 1
        fi
    fi
fi

cd ..

echo ""
echo "=========================================="
echo "Environment Setup Complete!"
echo "=========================================="
echo ""
echo "Environment: $ENV_NAME"
echo "Location: $(pwd)/$ENV_PATH"
echo ""
echo "To activate manually:"
echo "  source $ENV_PATH/bin/activate"
echo ""
echo "Installed packages:"
uv pip list

echo ""
