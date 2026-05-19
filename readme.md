# Autograd C++

A small automatic differentiation engine and neural network library built from scratch in C++. Inspired by Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd).

The whole point of this project was to understand how backpropagation and autograd actually work under the hood, instead of just using PyTorch as a black box.

## What it does

- **Scalar autograd engine** — A `Value` class that tracks computations and builds a dynamic computation graph as you do math operations on it.
- **Backpropagation** — Computes gradients automatically using reverse-mode autodiff with a topological sort over the graph.
- **Neural network module** — `Perceptron`, `Layer`, and `NN` classes to build and train simple feedforward networks.
- **Graph visualization** — Exports the computation graph to Graphviz `.dot` format so you can actually see what's going on.

## Supported operations

| Operation | Backward ✓ |
|-----------|-----------|
| `+` `-` `*` `/` | ✓ |
| `tanh` | ✓ |
| `relu` | ✓ |
| `leaky_relu` | ✓ |
| `exp` | ✓ |

## How it works

Every `Value` object stores its data, gradient, the operation that created it, and pointers to its children. When you call `.backward()` on the final output, it does a DFS-based topological sort of the graph and propagates gradients back through every node using the chain rule.

The memory management uses `shared_ptr` for the graph nodes and `weak_ptr` inside the backward lambdas to avoid circular reference issues.

## Quick example

```cpp
#include "Value.h"
#include "perceptron.h"

int main() {
    // create a small neural network: 2 inputs -> 4 -> 4 -> 1 output
    NN net(2, {"leaky_relu", "leaky_relu", "linear"}, {4, 4, 1});

    // some training data (inputs, expected output)
    std::vector<Value> input = {3, 4};
    Value expected(7);

    // forward pass
    std::vector<Value> output = net.forward(input);

    // compute loss
    Value loss = (output[0] - expected) * (output[0] - expected);

    // backward pass
    net.zero_grad();
    loss.backward();

    // update weights
    net.train(0.01);

    std::cout << "Output: " << output[0] << std::endl;
    std::cout << "Loss: " << loss << std::endl;
}

// For more examples, see the test/ folder.
```

## Building

Requires C++17 and `g++`. Just run:

```bash
make
./my_autograd.exe
```

To utils the computation graph (needs [Graphviz](https://graphviz.org/) installed):

```bash
make png
```

This generates `my_network.png` from the `.dot` file.

## Project structure

```
├── Value.h / Value.cpp       # Core autograd engine (Value + ValueImpl)
├── perceptron.h / .cpp        # Neural network components (Perceptron, Layer, NN)
├── utils.h / .cpp         # Computation graph visualization
├── main.cpp                   # Training example
├── test/                      # Test files
│   ├── leaky_relu_test.cpp
│   └── nn_test.cpp
└── Makefile
```

## Sample output

Training a small network on simple addition data for 300 epochs:

```
Value(data=3.59729, grad=1)
Value(data=1.67602, grad=1)
Value(data=0.494854, grad=1)
Value(data=0.158523, grad=1)
Value(data=0.10869, grad=1)
Value(data=0.0833275, grad=1)

Validation output = Value(data=6.69359, grad=0), Validation Loss = Value(data=0.306412, grad=0)
```

The loss goes down and the network learns to approximate addition, which is pretty cool for something built from scratch.

## Limitations

- Everything is scalar-based, so it's slow compared to real frameworks that use tensors and matrix operations. This is meant for learning, not for training actual models.
- Only supports simple SGD for optimization (no Adam, momentum, etc.).
- No GPU support yet(sadly).

## What I learned

- How reverse automatic differentiation works at a low level
- Managing memory in C++ with smart pointers, especially avoiding circular references between `shared_ptr` and lambda captures
- How the chain rule translates into actual code
- Building a computation graph dynamically and traversing it with topological sort

## References

- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [Karpathy's neural networks zero to hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) YouTube series