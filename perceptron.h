#include "Value.h"
#include "Visualize.h"

class Perceptron
{
    private:
        std::vector<Value> weights;
        Value bias;

    public:
        Perceptron(int n) : weights(n), bias(0.0f) {}

        Value forward(const std::vector<Value>& inputs);

        void train(float lr);

        void zero_grad();

        void info();
};