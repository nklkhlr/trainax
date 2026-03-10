# Installation

## Requirements

- Python 3.12+
- JAX (CPU or GPU/TPU)

## From PyPI

Install the core package:

```bash
pip install trainax
```

### Optional dependency groups

`trainax` ships optional groups for each model backend:

| Group | Installs | Required for |
|---|---|---|
| `eqx` | `equinox` | `EQXTrainer` |
| `flax` | `flax` | `NNXTrainer` |

```bash
# Equinox backend
pip install "trainax[eqx]"

# Flax NNX backend
pip install "trainax[flax]"

# Both
pip install "trainax[eqx,flax]"
```

## With uv

```bash
# Add core
uv add trainax

# Sync optional groups
uv sync --group eqx
uv sync --group flax
uv sync --group eqx --group flax
```

## From source

```bash
git clone https://github.com/nikolaik/jax-trainer
cd jax-trainer
uv sync --all-groups
```

## Verifying the install

```python
import trainax
print(trainax.__version__)
```

Or run the test suite:

```bash
uv run pytest
```
