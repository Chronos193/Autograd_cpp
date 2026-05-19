#include "Value.h"
#include "utils.h"
#include "perceptron.h"
#include "dataset.h"

int main() {
    // Hyperparams
    int batch_size=32;
    int layer1 = 3;
    int layer2 = 3;
    int out_layer = 1;
    float lr = 0.01;
    float train_size = 0.90;
    float val_size = 0.10;
    //
    CSVDataset xor_dataset("E:\\Autograd_c++\\Xor_Dataset.csv", {0,1}, {2});
    DataLoader xor_loader(xor_dataset, batch_size, true);
    NN xor_net(2, {"leaky_relu", "leaky_relu", "sigmoid"}, {layer1, layer2, out_layer});
    int num_batches = xor_loader.size();
    int train_batches = num_batches * train_size;
    for(size_t i=0;i<train_batches;i++)
    {
        xor_net.zero_grad();
        auto batch = xor_loader.get_batch(i);
        Value loss(0);
        for(const auto& data: batch)
        {
            auto Y_pred = xor_net.forward(data.first);
            auto zipped_vec = autograd::utils::zip(data.second, Y_pred);
            loss = loss + autograd::utils::bce_loss(zipped_vec);
        }
        loss.backward();
        xor_net.train(lr);
        if(train_batches%10==0)
        {
            std::cout << loss << std::endl;
        }
    }
    int val_batches = num_batches * val_size;
    Value val_loss(0);
    for(size_t i=0;i<val_batches;i++)
    {
        auto batch = xor_loader.get_batch(i);
        for(const auto& data: batch)
        {
            auto Y_pred = xor_net.forward(data.first);
            auto zipped_vec = autograd::utils::zip(data.second, Y_pred);
            val_loss = val_loss + autograd::utils::bce_loss(zipped_vec);
        }
    }
    std::cout << "Validation Loss is :- " << (val_loss.get_data()/val_batches) << std::endl;
    return 0;
}