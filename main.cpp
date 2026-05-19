#include "Value.h"
#include "visualize.h"
#include "perceptron.h"
#include "dataset.h"
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


Value mae_loss(const std::vector<std::pair<Value, Value>>& arr)
{
    Value loss(0);
    for(auto ele: arr)
    {
        if(ele.first.get_data() > ele.second.get_data())
        {
            loss = loss + (ele.first-ele.second);
        }
        else
        {
            loss = loss + (ele.second-ele.first);
        }
    }
    Value total_ele(arr.size());
    loss = loss/total_ele;
    return loss;
}

Value mae_loss(Value y, Value y_pred)
{
    Value loss;
    if(y.get_data()>y_pred.get_data())
    {
        loss = (y-y_pred);
    }
    else
    {
        loss = (y_pred-y);
    }
    return loss;
}


int main() {
    CSVDataset dataset("data.csv", {0,1}, {2,3});
    dataset.info();
    std::pair<std::vector<Value>,std::vector<Value>> out = dataset.get(1);
    std::cout << "X :- " << " ";
    for(auto ele: out.first)
    {
        std::cout << ele << " ";
    }
    std::cout << std::endl;
    std::cout << "Y :- " << " ";
    for(auto ele: out.second)
    {
        std::cout << ele << " ";
    }
    return 0;
}