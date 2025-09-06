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

### 2. Create a Workspace

We recommend a single workspace folder for all assignments:

```bash
mkdir workspace && cd workspace
```

### 3. Set Up a Virtual Environment

This keeps dependencies isolated:

```bash
python -m venv .venv
source .venv/bin/activate
```

You only need to run the first line once.
Run the second line whenever you open a new terminal.
Your prompt should show `(venv)` if activation worked.

See the [venv docs](https://docs.python.org/3/library/venv.html) for more info.

### 4. Clone & Install

Clone your forked TinyTorch repo and install requirements:

```bash
git clone https://github.com/<your-username>/tinytorch.git
cd tinytorch
python -m pip install -r requirements.txt
python -m pip install -r requirements.extra.txt
python -m pip install -Ue .
```

For Anaconda users, you may also need:

```bash
conda install llvmlite
```

### 5. Verify Installation

Run Python and check that the library imports:

```python
import minitorch
```

If no error appears, you are ready to start implementing!

---

## 🧠 Philosophy

TinyTorch is designed to be:

- **Readable:** Every line is easy to understand, even for newcomers.
- **Incremental:** Each assignment builds upon the last.
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

---

Would you like me to add a **section for visualizations** (e.g., showing loss curves, computation graphs) since MiniTorch includes tools for tracking your progress?

**Q1:** Should I include a section with sample `pytest` commands for running the included unit tests?
**Q2:** Do you want me to add a short “Contributing” section in case others want to learn from your fork?
**Q3:** Would you like me to include a roadmap checklist that you can update as you finish each assignment?
