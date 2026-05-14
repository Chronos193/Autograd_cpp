#include "Value.h"
#include "visualize.h"

// std::shared_ptr<Value> perceptron()
// {
//     std::shared_ptr<Value> x1 = std::make_shared<Value>(0.3);
//     std::shared_ptr<Value> x2 = std::make_shared<Value>(0.5);
//     std::shared_ptr<Value> w1 = std::make_shared<Value>(0.4);
//     std::shared_ptr<Value> w2 = std::make_shared<Value>(0.6);
//     std::shared_ptr<Value> weighted_sum;
//     std::shared_ptr<Value> output;
//     weighted_sum = w1 * w2;
//     output = weighted_sum->tanh();    
//     return output;
// }

std::shared_ptr<Value> Multiply()
{
    Value x1(0.3), x2(0.5), w1(0.4), w2(0.6);
    Value output = x1 * w1;
    std::shared_ptr<Value> output_ptr = std::make_shared<Value>(output);
    return output_ptr;
}

int main()
{
    std::unordered_set<std::shared_ptr<Value>> visited;
    std::vector<std::shared_ptr<Value>> topo;
    std::vector<std::shared_ptr<Value>> nodes;
    // auto output = perceptron();
    auto output = Multiply();
    output->backward();
    // autograd::utils::draw_graph(output, "my_network.dot");

    autograd::utils::return_all_nodes(output, nodes);

    for(auto node: nodes)
    {
        std::cout << *node << std::endl;
    }
    return 0;
}