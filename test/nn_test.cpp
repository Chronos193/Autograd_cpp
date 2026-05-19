#include "Value.h"
#include "utils.h"
#include "perceptron.h"

std::vector<std::pair<std::vector<Value>, Value>> data()
{
    std::vector<std::pair<std::vector<Value>, Value>> data_arr;

    data_arr.emplace_back(
        std::vector<Value>{2, 3},
        5
    );

    data_arr.emplace_back(
        std::vector<Value>{1, 4},
        5
    );

    data_arr.emplace_back(
        std::vector<Value>{5, 2},
        7
    );

    data_arr.emplace_back(
        std::vector<Value>{3, 1},
        4
    );

    data_arr.emplace_back(
        std::vector<Value>{0, 6},
        6
    );

    data_arr.emplace_back(
        std::vector<Value>{7, 2},
        9
    );

    return data_arr;
}

Value mse_loss(const std::vector<std::pair<Value, Value>>& arr)
{
    Value loss(0);
    for(auto ele: arr)
    {
        loss = loss + (ele.first-ele.second)*(ele.first-ele.second);
    }
    Value total_ele(arr.size());
    loss = loss/total_ele;
    return loss;
}

Value mse_loss(Value y, Value y_pred)
{
    Value loss = (y-y_pred)*(y-y_pred);
    return loss;
}



int main() {
    //Hyperparameters
    float lr = 0.01;
    int epochs = 3000;
    int input = 2;
    int layer1 = 4;
    int layer2 = 4;
    int output = 1;
    //
    NN n1(input, {"tanh", "tanh", "linear"}, {layer1, layer2, output});
    auto data_vec = data();
    for (int i=0;i<epochs;i++)
    {
        std::vector<Value> Y;
        std::vector<Value> Y_pred;  // Need to scopw it here so that the computation graph built is destroyed and memory is freed
        Y.reserve(data_vec.size());
        Y_pred.reserve(data_vec.size()); // Pre-allocate for faster use later
        // ================Forward Pass============
        for(auto& data: data_vec)
        {
            std::vector<Value> outputs;
            outputs = n1.forward(data.first);
            Y_pred.push_back(outputs[0]);
            Y.push_back(data.second);
        }
        Value loss = mse_loss(autograd::utils::zip(Y_pred, Y));
        // =====================Backward Pass==============
        n1.zero_grad();
        loss.backward();
        n1.train(lr);
        // ==============Printing===============
        if (i%1000 == 0)
        {
            std::cout << loss << std::endl;
            autograd::utils::draw_graph(loss);
        }
    }
    // Validation
    std::vector<Value> val_in = {3,4};
    Value y(7);
    std::vector<Value> val_out;
    val_out = n1.forward(val_in);
    Value val_loss = mse_loss(val_out[0], y); 
    std::cout << std::endl;
    std:: cout << "Validation output = " << val_out[0] << ", Validation Loss = " <<  val_loss << std::endl;
    return 0;
}