#include "Value.h"
#include "Visualize.h"
#include <random>

class Perceptron
{
    private:
        std::vector<Value> weights;
        Value bias;

    public:
        Perceptron(int n) : bias(0.0f) {
            // ============= Random Initialization of weights =====================
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

            weights.reserve(n);

            for(int i=0;i<n;i++)
            {
                weights.emplace_back(dist(gen));
            }
        }

        Value forward(const std::vector<Value>& inputs);

        void train(float lr);

        void zero_grad();

        void info();
};


class Layer
{
    private:
        std::vector<Perceptron> p_vec;
        std::string activation;
    public:
        Layer(int n, int p_size)
            : activation("linear")
        {
            for(int i = 0; i < n; i++)
            {
                p_vec.emplace_back(p_size);
            }
        }

        Layer(int n, std::string acti, int p_size)
            : activation(acti)
        {

            for(int i = 0; i < n; i++)
            {
                p_vec.emplace_back(p_size);
            }
        }

        std::vector<Value> forward(const std::vector<Value>& inputs); // Call all perceptron on same inputs vec by looping and then pass through activation

        void train(float lr); // Call on all perceptron in array

        void zero_grad(); // Call on all perceptron in array

        void info(); // Need to structure it

};


class NN
{
    private:
        std::vector<Layer> l_vec;
        std::vector<std::string> acti_vec;
    public:
        NN(int input_size,const std::vector<int>& layer_sizes)
        {
            int prev = input_size;

            for(int size : layer_sizes)
            {
                l_vec.emplace_back(size, prev);
                prev = size;
            }
        }
        NN(int input_size,const std::vector<std::string>& acti_vec,const std::vector<int>& layer_sizes)
        {
            if(acti_vec.size() != layer_sizes.size())
            {
                throw std::runtime_error("Activations size mismatch");
            }
            int prev = input_size;
            int i=0;
            for(int size : layer_sizes)
            {
                l_vec.emplace_back(size, acti_vec[i], prev);
                prev = size;
                i++;
            }
        }

        std::vector<Value> forward(const std::vector<Value>& inputs);

        void train(float lr);

        void zero_grad();

        void info();
};