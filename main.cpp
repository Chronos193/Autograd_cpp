#include "Value.h"
#include "visualize.h"
#include "perceptron.h"

std::vector<std::pair<std::vector<Value>, Value>> data() 
{
    std::vector<std::pair<std::vector<Value>, Value>> data_arr;
    std::vector<std::vector<Value>> X_arr;
    std::vector<Value> X_1;
    std::vector<Value> X_2;
    X_1.push_back(2); X_arr.push_back(X_1);
    X_2.push_back(1); X_arr.push_back(X_2); // X_arr = [[2], [1]]

    data_arr.emplace_back(X_arr[0],4);
    data_arr.emplace_back(X_arr[1],2);
    return data_arr;
}

Value mse_loss(std::vector<std::pair<Value, Value>>& arr)
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
    float lr = 0.1;
    int epochs = 150;
    //
    Perceptron p1(1);
    std::vector<std::pair<std::vector<Value>, Value>> data_arr = data();
    std::vector<Value> Y, Y_pred;
    Value x_val(1), y_val(2);
    std::vector<Value> X_test;
    X_test.push_back(x_val);
    for(auto data: data_arr)
    {
        Y.push_back(data.second);
    }
    for (int epoch=0 ;epoch<epochs ; epoch++)
    {
        p1.zero_grad(); // Clear gradients from prev steps
        Y_pred.clear(); // Clear after each epoch
            for(auto data: data_arr)
            {
                Y_pred.push_back(p1.forward(data.first)); // Forward pass
            }
        auto zipped_arr = autograd::utils::zip(Y, Y_pred);
        auto loss = mse_loss(zipped_arr); // Calculate loss
        std::cout << loss << std::endl;
        loss.backward(); // Backward pass
        p1.train(lr); // update 
        if(epoch == (epochs-1))
        {
            autograd::utils::draw_graph(loss);
        }
        // Validation Prediction
        if ((epoch%10 == 1) || (epoch == (epochs-1)))
        {
            auto y_pred_val = p1.forward(X_test);
            std::cout<< "Validation pred on value 1 (should be 2): " << y_pred_val << std::endl;
            std::cout<< "Validation loss : " << mse_loss(y_val, y_pred_val) << std::endl;
            p1.info();
        }
    }
    return 0;
}