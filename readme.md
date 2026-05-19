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
./bin/my_autograd.exe
```

To utils the computation graph (needs [Graphviz](https://graphviz.org/) installed):

```bash
make png
```

This generates `my_network.png` from the `.dot` file.

## Project structure

```
├── src/                       # Source files (.cpp)
│   ├── Value.cpp              # Core autograd engine
│   ├── perceptron.cpp         # Neural network components
│   ├── utils.cpp              # Computation graph visualization
│   ├── dataset.cpp            # Dataset handling
│   └── main.cpp               # Training example
├── include/                   # Header files (.h)
│   ├── Value.h
│   ├── perceptron.h
│   ├── utils.h
│   ├── dataset.h
│   └── rapidcsv.h             # CSV parsing library
├── data/                      # Datasets
│   └── Xor_Dataset.csv
├── test/                      # Test files
│   ├── leaky_relu_test.cpp
│   └── nn_test.cpp
└── Makefile
```

## Sample output

Training a small network on XOR dataset for 100 epochs:

```
Trained on XOR_Dataset from 
url = "https://www.kaggle.com/datasets/bipinmaharjan/xor-dataset/versions/1?resource=download"

Using     
    int batch_size=32;
    int layer1 = 3;
    int layer2 = 3;
    int out_layer = 1;
    size_t epochs = 100;
    float lr = 0.01;
    float train_size = 0.90; // Portion of data used in training rest is used in testing.

Got Results
    Value(data=0.675301, grad=1)
    Value(data=0.178561, grad=1)
    Value(data=0.0450419, grad=1)
    Value(data=0.0225882, grad=1)
    Value(data=0.0145437, grad=1)
    Value(data=0.0105704, grad=1)
    Value(data=0.00824065, grad=1)
    Value(data=0.00672519, grad=1)
    Value(data=0.0056649, grad=1)
    Value(data=0.00488417, grad=1)
    Validation Loss is :- 0.00596749
```

The loss goes down and the network learns to approximate the XOR function.

## Limitations

- Everything is scalar-based, so it's slow compared to real frameworks that use tensors and matrix operations.
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