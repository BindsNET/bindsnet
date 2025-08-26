# BindsNET - Spiking Neural Networks Library

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

BindsNET is a Python spiking neural network simulation library built on PyTorch that supports both CPU and GPU computation. It is designed for machine learning and reinforcement learning applications with biologically inspired neural networks.

## Working Effectively

### Bootstrap the Environment
Use these exact commands to set up the development environment:

```bash
# Install Poetry (required dependency manager)
pip install poetry==2.1.2
poetry config virtualenvs.create false

# Install core dependencies manually due to network timeout issues
pip install torch matplotlib numpy scipy pandas tqdm tensorboardX torchvision --timeout=600
pip install pytest flake8 --timeout=600

# WARNING: Full package installation may fail due to network timeouts
# These additional packages needed for full functionality (may timeout):
pip install opencv-python scikit-image scikit-learn --timeout=600

# Try Poetry installation (often fails due to dependency conflicts)
poetry install
```

**CRITICAL TIMING**: Package installation can take 15-30 minutes due to large PyTorch dependencies. NEVER CANCEL pip install commands. Set timeout to 60+ minutes.

### Build and Test
```bash
# Run all tests - NEVER CANCEL, can take 10-15 minutes
python -m pytest test/ --timeout=900

# Run specific test modules
python -m pytest test/network/ -v
python -m pytest test/encoding/ -v

# Quick validation test
python -c "import bindsnet; print('BindsNET imported successfully')"
```

**WARNING**: Full test suite requires ALL dependencies. Some tests may fail if OpenAI gym is not installed - this is expected.

### Code Quality and Formatting
```bash
# Run linting (required before commits)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Format code with black (automatic via pre-commit)
poetry run pre-commit run -a

# Install pre-commit hooks
poetry run pre-commit install
```

Always run linting before committing or CI will fail.

## Validation Scenarios

After making changes, ALWAYS test these scenarios:

### Basic Import Test
**WARNING**: Full package import currently fails due to missing opencv-python dependency.
```bash
# Test linting works (this is validated to work)
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Test individual modules (may work partially)
python -c "import torch; print('PyTorch available')"
python -c "import matplotlib.pyplot as plt; print('Matplotlib available')"
```

### Working Validation Commands
These commands are verified to work:
```bash
# Lint check - takes ~1 second, always run before commits
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# Check repository structure
ls -la bindsnet/  # Main package
ls -la examples/  # Example scripts
ls -la test/      # Test suite
```

### Run Example Scripts
**WARNING**: Examples require full dependency installation including opencv-python.
Test examples to ensure core functionality:
```bash
# Simple MNIST example (takes 5-10 minutes) - ONLY after all deps installed
cd examples/mnist
python eth_mnist.py --n_neurons 100 --time 100

# Check example structure without running (works immediately)
ls examples/mnist/      # MNIST classification examples
ls examples/benchmark/  # Performance benchmarking
ls examples/dotTracing/ # RL dot tracing demo
```

### Docker Validation
```bash
# Build Docker image (takes 30-45 minutes, NEVER CANCEL)
docker build . --timeout=3600

# Test Docker container
docker run -it <image_id> python -c "import bindsnet; print('Docker setup working')"
```

## Common Installation Issues

### Critical Dependencies Issue
**MAJOR ISSUE**: Package import fails until opencv-python is successfully installed:
```bash
# This is the blocking dependency - often fails due to network timeouts
pip install opencv-python --timeout=600

# Without opencv-python, you'll get "ModuleNotFoundError: No module named 'cv2'"
# when trying to import bindsnet
```

### Poetry Installation Problems
- Poetry may fail with `typing-extensions` conflicts
- **Solution**: Install Poetry via pip, not curl installer
- Use `poetry config virtualenvs.create false` to avoid venv conflicts

### Network Timeout Issues  
- PyPI downloads frequently timeout, especially opencv-python and scikit packages
- **Solution**: Use `--timeout=600` or higher for pip commands
- If pip fails, try installing packages individually or use different PyPI mirrors

### Missing Dependencies
Common missing packages and solutions:
```bash
# If import errors occur:
pip install matplotlib  # for plotting
pip install pandas     # for data analysis  
pip install cv2        # actually opencv-python
pip install torchvision # for vision utilities
```

## Repository Structure

### Key Directories
- `bindsnet/` - Main package source code
  - `analysis/` - Analysis and plotting utilities
  - `conversion/` - ANN to SNN conversion tools
  - `datasets/` - Dataset loading utilities
  - `encoding/` - Input encoding methods
  - `environment/` - RL environment interfaces
  - `learning/` - STDP learning rules
  - `models/` - Pre-built network models
  - `network/` - Core network components (nodes, connections, topology)
  - `pipeline/` - Training and evaluation pipelines
  - `preprocessing/` - Data preprocessing utilities

### Important Files
- `pyproject.toml` - Poetry configuration and dependencies
- `CONTRIBUTING.md` - Development guidelines
- `.github/workflows/python-app.yml` - CI configuration
- `examples/` - Example applications and benchmarks
- `test/` - Test suite organized by module
- `docs/` - Sphinx documentation source

### Frequently Modified Files
When working on core functionality, commonly edited files:
- `bindsnet/network/nodes.py` - Neuron implementations
- `bindsnet/network/topology.py` - Connection types
- `bindsnet/learning/*.py` - STDP learning rules
- `bindsnet/encoding/*.py` - Input encoding methods

## Example Commands and Expected Times

### Development Workflow
```bash
# Full development cycle (60-90 minutes total)
poetry install              # 20-30 min, NEVER CANCEL
python -m pytest test/      # 10-15 min, NEVER CANCEL  
flake8 .                    # 1-2 min
poetry run pre-commit run -a # 2-3 min

# Quick validation cycle (1-5 minutes) - ALWAYS WORKS
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics  # ~1 second
python -c "import torch; import matplotlib; print('Core deps available')"  # ~2 seconds
ls -la bindsnet/ examples/ test/  # Instant

# Full validation cycle (only after ALL dependencies installed)
python -c "import bindsnet"  # Test full package import
python examples/mnist/eth_mnist.py --time 50 --n_neurons 50  # 5-10 min
```

### Running Examples
```bash
# MNIST classification (5-10 min)
cd examples/mnist && python eth_mnist.py --plot

# Benchmark comparison (varies by parameters, 10-30 min)
cd examples/benchmark && python benchmark.py

# RL dot tracing (2-5 min)  
cd examples/dotTracing && python dot_tracing.py
```

## Build Troubleshooting

### If Poetry Fails
```bash
# Fallback to pip installation
pip install -e .  # May also fail due to build dependencies
pip install torch torchvision matplotlib numpy scipy pandas tqdm
```

### If Tests Fail
```bash
# Run tests with more verbose output
python -m pytest test/ -v -s --tb=long

# Test specific modules that don't require all dependencies
python -m pytest test/network/test_nodes.py -v
```

### If Examples Don't Work
```bash
# Most likely cause: missing opencv-python
pip install opencv-python --timeout=600

# Check that core dependencies are available
pip list | grep torch    # Should show torch and related packages
pip list | grep matplotlib
pip list | grep opencv   # Look for opencv-python

# If opencv still fails, package import will fail
# Alternative: work directly with specific module files
```

## Performance Notes

- **CPU vs GPU**: Code supports both, GPU much faster for large networks
- **Memory Usage**: Large networks can use significant RAM/VRAM
- **Build Time**: Initial dependency installation is slowest part
- **Test Time**: Full test suite exercises many scenarios, be patient

Always allow sufficient time for builds and tests. The library handles complex neural computations that require substantial dependencies and can take time to process.