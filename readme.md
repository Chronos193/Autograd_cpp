# Micrograd in C++

This project is my attempt to implement an automatic differentiation engine and tools to make neural network in C++.

The goal of this implementation is to understand how automatic differentiation and backpropagation work under the hood by recreating the core ideas of micrograd from scratch.

## Features

- Scalar `Value` class for storing data and gradients
- Support for basic operations:
  - Addition
  - Subtraction
  - Multiplication
  - Division
  - `tanh`
  - `exp`
- Automatic computation graph construction
- Backpropagation through the graph
- Simple neural network components (work in progress)

## Why this project?

I wanted to learn the internals of deep learning frameworks by implementing a small autograd engine myself. Since micrograd is a minimal educational project, I chose to recreate it in C++ for practice with both C++ and neural network fundamentals.

## Status

This is a learning project and is still under development. The implementation may not exactly match the original micrograd and is mainly for experimentation and understanding.

## Reference

Original micrograd by :contentReference[oaicite:2]{index=2}: [GitHub repository](https://github.com/karpathy/micrograd)