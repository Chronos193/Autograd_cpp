#include "Value.h"
#include "utils.h"
#include "perceptron.h"
#include "dataset.h"
#include <chrono>

int main() {

    // Time bechmarking: Total time
    auto start_total = std::chrono::high_resolution_clock::now();
    // Hyperparams
    int batch_size=32;
    int layer1 = 3;
    int layer2 = 3;
    int out_layer = 1;
    int print_every = 1;
    int check_seed = 30;
    size_t epochs = 10;
    float lr = 0.01;
    float train_size = 0.90;
    float threshold = 0.2;
    //
    CSVDataset xor_dataset("E:\\Autograd_c++\\data\\Xor_Dataset.csv", {0,1}, {2});
    DataLoader xor_loader(xor_dataset, batch_size, true);
    NN xor_net(2, {"leaky_relu", "leaky_relu", "tanh"}, {layer1, layer2, out_layer});
    int num_batches = xor_loader.size();
    int train_batches = num_batches * train_size;

    // Train time
    auto start_train = std::chrono::high_resolution_clock::now();
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
            // autograd::utils::draw_graph(loss);
            xor_net.train(lr);
            if(i%50==0)
            {
                last_loss = loss;
            }
        }
        if(static_cast<int> (k) == check_seed)
        {
            if(last_loss.get_data() > threshold)
            {
                std::cout << "Bad Random Seeding Rerun: Stopping program" << std::endl;
                return 0;
            }
        }
        if (k%print_every == 0)
        {
            std::cout << last_loss << std::endl;
        }
    }

    // Ending Validation time
    auto end_train = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> train_duration = end_train - start_train;
    std::cout << "Training Time: " << train_duration.count() << " seconds\n";

    // Starting Validation Time
    auto start_val = std::chrono::high_resolution_clock::now();

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

    // Stopping validation timer
    auto end_val = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> val_duration = end_val - start_val;
    std::cout << "Validation Loss is :- " << (val_loss.get_data() / ((num_batches - train_batches) * batch_size)) << std::endl;
    std::cout << "Validation Time: " << val_duration.count() << " seconds\n";

    std::cout << "Validation Loss is :- " << (val_loss.get_data()/((num_batches - train_batches)*batch_size)) << std::endl;


    // Stopping total timer
    auto end_total = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_total - start_total;
    
    std::cout << "\n--- Summary ---" << std::endl;
    std::cout << "Total Execution Time: " << total_duration.count() << " seconds" << std::endl;

    return 0;
}