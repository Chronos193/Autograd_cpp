#include "perceptron.h"

// Perceptron
Value Perceptron::forward(const std::vector<Value>& inputs)
{
    if(inputs.size() != weights.size())
    {
        throw std::runtime_error("Input size mismatch");
    }
    Value sum = bias;
    for (size_t i = 0; i < weights.size(); i++)
        sum = sum + inputs[i] * weights[i];
    return sum;
}

void Perceptron::train(float lr)
{
    float weight_data, weight_grad;
    for(auto& weight : weights)
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
    std::cout << "The weights are: " << std::endl;
    for (auto& w : weights) {
        std::cout << w << " ";
    }
    std::cout << std::endl;
    // Reset bias
    std::cout << "The bias are: " << std::endl;
    std::cout << bias << std::endl;
}

// Layer

std::vector<Value> Layer::forward(const std::vector<Value>& inputs)
{
    std::vector<Value> outputs;
    Value temp;
    for( auto& p : p_vec)
    {
        temp = p.forward(inputs);
        if(activation == "linear")
        {
            outputs.push_back(temp);
        }
        if(activation == "tanh")
        {
            outputs.push_back(temp.tanh());
        }
        if(activation == "relu")
        {
            outputs.push_back(temp.relu());
        }
    }
    return outputs;
}

std::vector<Value> Layer::forward(const std::vector<Value>& inputs, const float alpha)
{
    std::vector<Value> outputs;
    outputs.reserve(p_vec.size());
    for( auto& p : p_vec)
    {
        Value temp = p.forward(inputs);
        if(activation == "leaky_relu")
        {
            outputs.push_back(temp.leaky_relu(alpha));
        }
        else
        {
            outputs.push_back(temp);
        }
    }
    return outputs;
}

void Layer::train(float lr)
{
    for(auto& p: p_vec)
    {
        p.train(lr);
    }
}

void Layer::zero_grad()
{
    for(auto& p: p_vec)
    {
        p.zero_grad();
    }
}

void Layer::info()
{
    std::cout << "Number of perceptron = " << p_vec.size() << std::endl;
    std::cout << "Activation function = " << activation << std::endl;
}

// Neural Network

std::vector<Value> NN::forward(const std::vector<Value>& inputs)
{
    if(l_vec.empty())
    {
        return inputs;
    }
    if (acti_vec.size() != l_vec.size())
    {
        throw std::runtime_error(
            "Mismatch between layers and activation vector in first forward instance");
    }
    std::vector<Value> temp = inputs;
    std::vector<Value> outputs;
    float alpha = 0.01;

    for(size_t i=0;i<l_vec.size();i++)
    {
        if(acti_vec[i] == "leaky_relu")
        {

            outputs = l_vec[i].forward(temp, alpha);
            temp = outputs;
        }
        else
        {
            outputs = l_vec[i].forward(temp);
            temp = outputs;
        }

    }
    return outputs;
}

std::vector<Value> NN::forward(const std::vector<Value>& inputs, const std::vector<float>& alpha_vec)
{
    if(l_vec.empty())
    {
        return inputs;
    }
    // ===============Error Handling==============
    if (acti_vec.size() != l_vec.size())
    {
        throw std::runtime_error(
            "Mismatch between layers and activation vector for second forward instance");
    }
    size_t count = 0;
    for(auto& acti: acti_vec)
    {
        if(acti == "leaky_relu")
        {
            count++;
        }
    }
    if(alpha_vec.size() != count)
    {
        throw std::runtime_error("Size mismatch between alpha and activation which needs it");
    }
    // ========================================================
    std::vector<Value> temp = inputs;
    std::vector<Value> outputs;
    int j=0;
    float alpha;

    for(size_t i=0;i<l_vec.size();i++)
    {
        if(acti_vec[i] == "leaky_relu")
        {
            alpha = alpha_vec[j++];
            outputs = l_vec[i].forward(temp, alpha);
            temp = outputs;
        }
        else
        {
            outputs = l_vec[i].forward(temp);
            temp = outputs;
        }

    }
    return outputs;
}

void NN::train(float lr)
{
    for(auto& layer: l_vec)
    {
        layer.train(lr);
    }
}

void NN::zero_grad()
{
    for(auto& layer: l_vec)
    {
        layer.zero_grad();
    }
}

void NN::info()
{
    std::cout << "Number of layer = " << l_vec.size() << std::endl;
}

