# BindsNET - Spiking Neural Networks for Machine Learning

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

BindsNET is a Python spiking neural network library built on PyTorch for machine learning applications. It provides tools for creating, training, and analyzing spiking neural networks with focus on biologically-plausible learning algorithms.

## Working Effectively

### Prerequisites and Environment Setup
- **CRITICAL**: Requires Python >=3.10 (currently tested with Python 3.12.3)
- **NEVER CANCEL**: All build and dependency installation commands can take 15-45 minutes. Always set timeouts to 60+ minutes.
- Install Poetry for dependency management: `pip install poetry` (takes ~2 minutes)

### Bootstrap and Install Dependencies
**Primary method (Poetry - recommended but may fail due to network issues):**
```bash
export PATH="$HOME/.local/bin:$PATH"
poetry config virtualenvs.create false  # Use system Python to avoid conflicts
poetry install  # NEVER CANCEL: Takes 15-45 minutes, set timeout to 60+ minutes
poetry run pre-commit install  # Setup code formatting hooks
```

**Alternative method (pip - use when Poetry fails):**
```bash
# Install core dependencies first
pip install torch torchvision numpy scipy matplotlib tqdm pytest

# Install additional packages as needed (may require system packages)
sudo apt update && sudo apt install -y python3-pandas python3-sklearn python3-dev
pip install tensorboardX scikit-learn opencv-python gymnasium[atari] ale-py

# Install in editable mode
pip install -e .
```

**IMPORTANT NOTES:**
- Poetry installation may fail due to network timeouts or system package conflicts
- When Poetry fails, use pip-based installation as backup
- Some dependencies (like pandas) may have numpy version conflicts - use system packages via apt when needed
- Network timeouts are common - retry operations with longer timeouts

### Testing and Validation
**Run tests:**
```bash
# With Poetry (if installed successfully)
poetry run pytest  # NEVER CANCEL: Takes 5-15 minutes

# With pip installation
export PYTHONPATH="/home/runner/work/bindsnet/bindsnet:$PYTHONPATH"
python -m pytest test/ -v  # May have import issues due to missing dependencies
```

**CRITICAL**: Many tests may fail due to missing optional dependencies (pandas, scikit-learn, opencv). This is expected in environments with incomplete dependency installation.

## Validation Scenarios

**ALWAYS test these scenarios after making changes:**

1. **Basic import test (most reliable):**
```bash
python -c "
import sys, os
print('Python version:', sys.version)
try:
    import torch, numpy as np, matplotlib
    print('✓ Core dependencies (torch, numpy, matplotlib) available') 
    print('  PyTorch version:', torch.__version__)
    print('  NumPy version:', np.__version__)
    print('✓ Basic validation complete')
except Exception as e:
    print('✗ Error:', e)
"
```

2. **Repository structure validation:**
```bash
# Verify key directories and files exist
test -d bindsnet && echo "✓ bindsnet/ directory exists" || echo "✗ bindsnet/ missing"
test -d examples && echo "✓ examples/ directory exists" || echo "✗ examples/ missing"  
test -d test && echo "✓ test/ directory exists" || echo "✗ test/ missing"
test -f pyproject.toml && echo "✓ pyproject.toml exists" || echo "✗ pyproject.toml missing"
```

3. **Example help text (limited validation):**
```bash
# These will likely fail due to import issues but show example structure
cd examples/mnist && python eth_mnist.py --help 2>&1 | head -5 || echo "Import dependencies missing (expected)"
cd examples/benchmark && python benchmark.py --help 2>&1 | head -5 || echo "Import dependencies missing (expected)"  
```

4. **Code formatting and linting (when dev tools available):**
```bash
# Only works if dev dependencies successfully installed
poetry run pre-commit run -a || echo "pre-commit tools not available"
```

## Repository Structure

### Key directories:
- `bindsnet/`: Main package code
  - `network/`: Core network components (nodes, connections, topology)
  - `learning/`: Learning algorithms (STDP, reward-modulated learning)  
  - `encoding/`: Input encoding methods (Poisson, rank-order, etc.)
  - `models/`: Pre-built network architectures
  - `analysis/`: Analysis and visualization tools
  - `datasets/`: Dataset handling utilities
  - `environment/`: RL environment interfaces

- `examples/`: Example scripts and tutorials
  - `mnist/`: MNIST classification examples with different architectures
  - `breakout/`: Atari Breakout reinforcement learning
  - `benchmark/`: Performance benchmarking tools
  - `tensorboard/`: TensorBoard integration examples

- `test/`: Unit tests (pytest-based)
- `docs/`: Documentation source files

### Important files:
- `pyproject.toml`: Poetry configuration and dependencies
- `setup.py`: Fallback setup script for pip installation
- `CONTRIBUTING.md`: Development workflow and guidelines
- `.github/workflows/`: CI/CD pipelines (python-app.yml, black.yml)

## Common Tasks and Troubleshooting

### Dependency Installation Issues:
- **Poetry fails with timeout**: Use pip-based installation method
- **numpy.dtype size error**: Install system packages via apt instead of pip
- **Module not found errors**: Export PYTHONPATH or use `pip install -e .`

### Development Workflow:
1. Create feature branch: `git branch feature-name`
2. Install pre-commit hooks: `poetry run pre-commit install`
3. Make changes and test locally
4. Format code: `poetry run pre-commit run -a`
5. Run tests: `poetry run pytest`
6. Commit and push changes

### Performance Notes:
- **CPU vs GPU**: Examples default to CPU, use `--gpu` flag for CUDA acceleration
- **Memory usage**: Large networks may require significant RAM (8GB+ recommended)  
- **Training time**: 
  - eth_mnist.py: ~1 hour on Intel i7 CPU for full training
  - batch_eth_mnist.py: ~1 minute/epoch on GPU (GTX 1080ti)
  - Benchmark tests: 5-15 minutes depending on parameters

## Known Limitations and Common Issues

1. **Dependency Installation Challenges:**
   - Poetry may fail with network timeouts (common in CI environments)
   - System package conflicts between pip and apt-installed packages
   - numpy/pandas version mismatches causing import errors
   - **SOLUTION**: Use pip-based installation as fallback, install system packages via apt

2. **Import and Module Issues:**
   - Many examples fail to import bindsnet due to missing analysis dependencies (pandas)
   - tensorboardX and other optional dependencies may not be available
   - **SOLUTION**: Set PYTHONPATH and install core dependencies only for basic functionality

3. **Development Tool Availability:**
   - pre-commit hooks may not work if dev dependencies not installed
   - black/isort formatters may not be in PATH
   - **SOLUTION**: Install formatting tools separately or skip formatting validation

4. **Example Execution:**
   - Full MNIST examples require 1+ hours and significant computing resources
   - GPU examples need CUDA-compatible PyTorch installation
   - **SOLUTION**: Use minimal parameters for testing (small epochs, reduced data)

5. **Testing Issues:**
   - pytest may fail on multiple test modules due to missing dependencies
   - Analysis and conversion tests especially prone to import errors
   - **SOLUTION**: Focus on core network functionality tests, accept partial test failures

## Example Command Reference

```bash
# Quick start with minimal MNIST example (1 epoch, reduced data)
export PYTHONPATH=".:$PYTHONPATH" 
python examples/mnist/eth_mnist.py --n_epochs 1 --n_train 100 --n_test 50 --time 100
# NEVER CANCEL: Still takes 15-30 minutes even with reduced parameters

# Benchmark network performance  
python examples/benchmark/benchmark.py --start 100 --stop 500 --step 100

# Run tests (may fail due to missing dependencies)
poetry run pytest test/ -v --tb=short || python -m pytest test/ -v

# Format code (requires dev dependencies)
poetry run pre-commit run -a
# Alternative if poetry dev deps not available:
pip install black isort && black bindsnet/ examples/ test/ && isort bindsnet/ examples/ test/
```

**TIMEOUT WARNINGS AND TIMING EXPECTATIONS:**
- **Poetry install**: 60+ minutes timeout (15-45 minutes typical, varies by network)
- **pip install torch/dependencies**: 30+ minutes timeout (10-20 minutes typical)
- **pytest execution**: 30+ minutes timeout (5-15 minutes, many tests may fail)
- **MNIST example (full)**: 90+ minutes timeout (~1 hour typical on Intel i7)
- **MNIST example (minimal params)**: 45+ minutes timeout (15-30 minutes typical)
- **pre-commit formatting**: 15+ minutes timeout (2-5 minutes typical)
- **NEVER CANCEL** any long-running operations - they are expected to take significant time

## Validated Commands

The following commands have been tested and work reliably:

```bash
# Environment check (always works)
python --version && echo "✓ Python available"

# Dependencies check  
python -c "import torch, numpy, matplotlib; print('Core deps OK')"

# Repository structure
test -d bindsnet && test -d examples && test -d test && echo "✓ Repo structure OK"

# Poetry availability
poetry --version || echo "Poetry not available, use pip method"
```

Always validate any changes by running at least the basic dependency and structure checks to ensure the environment is functional.