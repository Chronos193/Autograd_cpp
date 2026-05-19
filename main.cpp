#include "Value.h"
#include "utils.h"
#include "perceptron.h"
#include "dataset.h"

int main() {
    CSVDataset dataset("E:\\Autograd_c++\\Xor_Dataset.csv", {0,1}, {2});
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