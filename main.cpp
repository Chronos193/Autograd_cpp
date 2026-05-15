#include "Value.h"
#include "visualize.h"

std::shared_ptr<ValueImpl> perceptron()
{
    std::shared_ptr<ValueImpl> x1 = std::make_shared<ValueImpl>(0.3);
    std::shared_ptr<ValueImpl> x2 = std::make_shared<ValueImpl>(0.5);
    std::shared_ptr<ValueImpl> w1 = std::make_shared<ValueImpl>(0.4);
    std::shared_ptr<ValueImpl> w2 = std::make_shared<ValueImpl>(0.6);
    std::shared_ptr<ValueImpl> weighted_sum;
    std::shared_ptr<ValueImpl> output;
    weighted_sum = *w1 * w2;
    output = weighted_sum->tanh();    
    return output;
}

// std::shared_ptr<ValueImpl> Multiply()
// {
//     ValueImpl x1(0.3), x2(0.5), w1(0.4), w2(0.6);
//     ValueImpl output = x1 * w1;
//     std::shared_ptr<ValueImpl> output_ptr = std::make_shared<ValueImpl>(output);
//     return output_ptr;
// }

int main() {

    std::cout << "Starting Execution" << std::endl;
    // ax+b a=10.0 b=10.0
    Value a(-2), b(3);
    
    // This is now valid, readable C++!
    Value output = a.relu();
    
    output.backward();
    
    autograd::utils::draw_graph(output);

    std::vector<Value> node_arr;

    autograd::utils::return_all_nodes(output, node_arr);

    for(auto ele:node_arr)
    {
        std::cout << ele << std::endl;
    }
    
    return 0;
}