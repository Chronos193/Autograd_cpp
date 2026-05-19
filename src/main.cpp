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
    size_t epochs = 100;
    float lr = 0.01;
    float train_size = 0.90;
    //
    CSVDataset xor_dataset("E:\\Autograd_c++\\data\\Xor_Dataset.csv", {0,1}, {2});
    DataLoader xor_loader(xor_dataset, batch_size, true);
    NN xor_net(2, {"leaky_relu", "leaky_relu", "tanh"}, {layer1, layer2, out_layer});
    int num_batches = xor_loader.size();
    int train_batches = num_batches * train_size;
    for(size_t k=0;k<epochs;k++)
    {
        Value last_loss(0);
        for(int i=0;i<train_batches;i++)
        {
            Value loss(0);
            xor_net.zero_grad();
            auto batch = xor_loader.get_batch(i);
            for(const auto& data: batch)
            {
                auto Y_pred = xor_net.forward(data.first);
                for(size_t i=0;i<Y_pred.size();i++)
                {
                    Y_pred[i] = ((Y_pred[i] + Value(1))/Value(2)).clamp(1e-7f, 1.0f - 1e-7f);
                }
                auto zipped_vec = autograd::utils::zip(data.second, Y_pred);
                loss = loss + autograd::utils::bce_loss(zipped_vec);
            }
            loss = loss/Value(batch_size);
            loss.backward();
            xor_net.train(lr);
            if(i%50==0)
            {
                last_loss = loss;
            }
        }
        if (k%10 == 0)
        {
            std::cout << last_loss << std::endl;
        }
    }
    Value val_loss(0);
    for(int i=train_batches;i<(num_batches);i++)
    {
        auto batch = xor_loader.get_batch(i);
        for(const auto& data: batch)
        {
            auto Y_pred = xor_net.forward(data.first);
            for(size_t i=0;i<Y_pred.size();i++)
            {
                Y_pred[i] = ((Y_pred[i] + Value(1))/Value(2)).clamp(1e-7f, 1.0f - 1e-7f);
            }
            auto zipped_vec = autograd::utils::zip(data.second, Y_pred);
            val_loss = val_loss + autograd::utils::bce_loss(zipped_vec);
        }
    }
    std::cout << "Validation Loss is :- " << (val_loss.get_data()/((num_batches - train_batches)*batch_size)) << std::endl;
    return 0;
}