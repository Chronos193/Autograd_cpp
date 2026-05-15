#include "perceptron.h"

Value Perceptron::forward(const std::vector<Value>& inputs)
{
    Value sum = bias;
    for (size_t i = 0; i < weights.size(); i++)
        sum = sum + inputs[i] * weights[i];
    return sum;
}

void Perceptron::train(float lr)
{
    float weight_data, weight_grad;
    for(auto weight : weights)
    {
        weight_data = weight.get_data();
        weight_grad = weight.get_grad();
        weight_data = weight_data - lr*weight_grad;
        weight.set_data(weight_data);
    }
    float bias_data = bias.get_data();
    float bias_grad = bias.get_grad();
    bias_data = bias_data - lr*bias_grad;
    bias.set_data(bias_data);
}

void Perceptron::zero_grad() {
    // Reset weights
    for (auto& w : weights) {
        w.set_grad(0);
    }
    // Reset bias
    bias.set_grad(0);
}

void Perceptron::info()
{
    std::cout << "Info Below:- " << std::endl;
    std::cout << "The weights are: " << std::endl;
    for (auto& w : weights) {
        std::cout << w << std::endl;
    }
    // Reset bias
    std::cout << "The bias are: " << std::endl;
    std::cout << bias << std::endl;
}