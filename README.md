# TinyTorch

TinyTorch is a personal fork of the [MiniTorch](https://github.com/minitorch/minitorch) teaching library — a DIY deep learning framework built from scratch to teach the internals of machine learning systems.

This project is where I implement **all of the assignments**, step by step, to gain a deep understanding of:

- Core ML programming concepts
- Automatic differentiation
- Tensor operations
- GPU and parallel execution
- Foundational deep learning components

By the end, TinyTorch will be able to run real Torch-like code in pure Python.

---

## 📚 Assignments & Goals

The project is organized as a series of progressive assignments. Each one builds on the previous to implement a working, minimalist deep learning framework:

1. **ML Programming Foundations**

   - Scalars, arithmetic operations
   - Computation graph construction
   - Writing forward/backward passes manually

2. **Autodifferentiation**

   - Implementing backpropagation
   - Designing `ScalarFunction` classes like `Mul`, `Add`, `ReLU`, etc.
   - Building a flexible autograd system

3. **Tensors**

   - Extending scalars to multi-dimensional arrays
   - Broadcasting, indexing, and efficient storage

4. **GPUs & Parallel Programming**

   - Writing CUDA kernels
   - Running operations on GPUs for speedup

5. **Foundational Deep Learning**

   - Linear layers, loss functions, optimizers
   - Training a small neural network end-to-end

Example task inside the repo:

```python
class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx, a):
        # TODO: Implement for Task 1.2
        raise NotImplementedError("Need to implement for Task 1.2")

    @staticmethod
    def backward(ctx, d_output):
        # TODO: Implement for Task 1.4
        raise NotImplementedError("Need to implement for Task 1.4")
```

---

## ⚙️ Setup

TinyTorch requires **Python 3.11+**.

### 1. Check Your Python Version

```bash
python --version
python3 --version
```

### 2. Set Up a Virtual Environment

This keeps dependencies isolated:

```bash
python -m venv .venv
source .venv/bin/activate
```

You only need to run the first line once.
Run the second line whenever you open a new terminal.
Your prompt should show `(venv)` if activation worked.

See the [venv docs](https://docs.python.org/3/library/venv.html) for more info.

Absolutely! Since we’ve switched to a `pyproject.toml` with editable install and optional extras, we can simplify and modernize that section. Here’s the updated version for your TinyTorch README:

---

### 4. Clone & Install

Clone your forked TinyTorch repo:

```bash
git clone https://github.com/your-username/tinytorch.git
cd tinytorch
```

Install TinyTorch in editable mode:

```bash
# Install core TinyTorch
python -m pip install -Ue .

# Optionally, install all extras (dev tools, ML, visualization, datasets, web)
pip install -Ue ".[dev,visualization,datasets,web,ml]"
```

---

### 5. Verify Installation

Run Python and check that the library imports:

```python
import tinytorch
```

If no error appears, you are ready to start implementing TinyTorch assignments!

---

## 🧠 Philosophy

TinyTorch is designed to be:

- **Readable:** Every line is easy to understand, even for newcomers.
- **Educational:** The goal is mastery, not speed.

---

## 🎯 Goals for This Fork

✅ Re-implement every MiniTorch assignment
✅ Write clear docstrings and comments for each function
✅ Add small tests and experiments to confirm correctness
✅ Explore optional extensions like new activations or optimizers

---

## 🙌 Credits

- Original project: [MiniTorch](https://github.com/minitorch/minitorch)
- Developed for _Machine Learning Engineering_ at Cornell Tech by [David Duvenaud](https://cs.toronto.edu/~duvenaud/) and collaborators.
- Inspired by industry experience at Hugging Face.
