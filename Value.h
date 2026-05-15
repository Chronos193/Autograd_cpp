#ifndef Value_H
#define Value_H

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

class ValueImpl: public std::enable_shared_from_this<ValueImpl>
{
    private:
        float data;
        float grad = 0;
        std::vector<std::shared_ptr<ValueImpl>> children_arr;
        std::string op = "";
    public:
    // BackProp Function
        std::function<void()> _backward = [](){}; // An empty lambda function
    // Basic functions 
        // Constructors
        ValueImpl(float data);
        ValueImpl(float data, std::vector<std::shared_ptr<ValueImpl>> children, std::string op);
        
        // Getters
        float get_data() const;
        float get_grad() const;
        const std::vector<std::shared_ptr<ValueImpl>>&  get_childen() const;
        std::string get_op() const;

        // Setters
        void set_data(float data);
        void set_grad(float grad);

        // Representation
        // friend std::ostream& operator<<(std::ostream& os, const ValueImpl& v);

        // BackPropagation Implementation by DFS Topological Sort
        void backward();

    //Arithmethic implementations
        std::shared_ptr<ValueImpl> operator+ (std::shared_ptr<ValueImpl> other);
        std::shared_ptr<ValueImpl> operator- (std::shared_ptr<ValueImpl> other);
        std::shared_ptr<ValueImpl> operator* (std::shared_ptr<ValueImpl> other);
        std::shared_ptr<ValueImpl> operator/ (std::shared_ptr<ValueImpl> other);
        std::shared_ptr<ValueImpl> relu();
        std::shared_ptr<ValueImpl> tanh();
        std::shared_ptr<ValueImpl> exp();
};

// Writing the wrapper around Value Implementaion
class Value
{
    private:
        std::shared_ptr<ValueImpl> impl;

    public:
        // Construuctors
        Value(float data) : impl(std::make_shared<ValueImpl>(data))
        {
        }
        Value(std::shared_ptr<ValueImpl> i) : impl(i) 
        {
        }
        Value(float data, std::vector<std::shared_ptr<ValueImpl>> children, std::string op) : impl(std::make_shared<ValueImpl>(data, children, op))
        {
        }
        // Returning internal shared pointer
        std::shared_ptr<ValueImpl> _return_shared_pointer() const
        {
            return impl;
        }
        // Helper Functions
        float get_data() const
        {
            return impl->get_data();
        }
        float get_grad() const
        {
            return impl->get_grad();
        }
        void set_data(float data)
        {
            impl->set_data(data);
        }

        // Representation
        friend std::ostream& operator<<(std::ostream& os, const Value& v)
        {
            os << "Value(data=" << v.get_data() << ", grad=" << v.get_grad() << ")";
            return os;
        }
        
        //BackPropagation
        void backward()
        {
            impl->backward();
        }

        // Operator overrinde

        Value operator+ (const Value &other)
        {
            return Value(*impl + other._return_shared_pointer());
        }
        Value operator- (const Value &other)
        {
            return Value(*impl - other._return_shared_pointer());
        }
        Value operator* (const Value &other)
        {
            return Value(*impl * other._return_shared_pointer());
        }
        Value operator/ (const Value &other)
        {
            return Value(*impl / other._return_shared_pointer());
        }
        Value relu()
        {
            return Value(impl->relu());
        }
        Value tanh ()
        {
            return Value(impl->tanh());
        }
        Value exp ()
        {
            return Value(impl->exp());
        }
};

#endif