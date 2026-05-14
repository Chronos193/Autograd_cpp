#ifndef VALUE_H
#define VALUE_H

#include <vector>
#include <string>
#include <iostream>
#include <memory>
#include <cmath>
#include <functional>
#include <stack>
#include <unordered_set>
#include <set>
#include <fstream>
#include <sstream>

class Value: public std::enable_shared_from_this<Value>
{
    private:
        float data;
        float grad = 0;
        std::vector<std::shared_ptr<Value>> children_arr;
        std::string op = "";
    public:
    // BackProp Function
        std::function<void()> _backward = [](){}; // An empty lambda function
    // Basic functions 
        // Constructors
        Value(float data);
        Value(float data, std::vector<std::shared_ptr<Value>> children, std::string op);
        
        // Getters
        float get_data() const;
        float get_grad() const;
        const std::vector<std::shared_ptr<Value>>&  get_childen() const;
        std::string get_op() const;

        // Setters
        void set_data(float data);
        void set_grad(float grad);

        // Representation
        friend std::ostream& operator<<(std::ostream& os, const Value& v);

        // BackPropagation Implementation by DFS Topological Sort
        void backward();

    //Arithmethic implementations
        std::shared_ptr<Value> operator+ (std::shared_ptr<Value> other);
        Value operator* (Value other);
        std::shared_ptr<Value> tanh();
};

#endif