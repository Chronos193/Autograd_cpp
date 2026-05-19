#include "Value.h"

// Constructors
ValueImpl::ValueImpl(float data): data(data)
{

}

ValueImpl::ValueImpl(float data, std::vector<std::shared_ptr<ValueImpl>> children, std::string op): data(data), children_arr(children), op(op)
{

}

// Getters
float ValueImpl::get_data() const
{
    return data;
}

float ValueImpl::get_grad() const
{
    return grad;
}

const std::vector<std::shared_ptr<ValueImpl>>&  ValueImpl::get_children() const
{
    return children_arr;
}

std::string ValueImpl::get_op() const
{
    return op;
}

// Setters
void ValueImpl::set_data(float data)
{
    this->data = data;
}

void ValueImpl::set_grad(float grad)
{
    this->grad = grad;
}

// Representation
// std::ostream& operator<<(std::ostream& os, const ValueImpl& v) {
//     os << "ValueImpl(data=" << v.data << ", grad=" << v.grad << ")";
//     return os;
// }

// BackPropagation Implementation by DFS Topological Sort
void ValueImpl::backward()
{
    std::function<void(
        std::shared_ptr<ValueImpl>,
        std::unordered_set<std::shared_ptr<ValueImpl>>&,
        std::vector<std::shared_ptr<ValueImpl>>&
    )> build_topo;
    std::unordered_set<std::shared_ptr<ValueImpl>> visited;
    std::vector<std::shared_ptr<ValueImpl>> topo;
    auto node = shared_from_this();
    build_topo = [&](std::shared_ptr<ValueImpl> node, std::unordered_set<std::shared_ptr<ValueImpl>>& visited, std::vector<std::shared_ptr<ValueImpl>>& topo){
        // If we haven't visited this node yet
        if (visited.find(node) == visited.end()) {
            visited.insert(node); // Mark as visited
            
            // Visit all children first
            for (const auto& child : node->get_children()) {
                build_topo(child, visited, topo);
            }
            
            // After children are processed, push this node
            topo.push_back(node);
        }
    };
    // ... after defining build_topo ...

    build_topo(node, visited, topo); 

    node->set_grad(1.0f);
    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        (*it)->_backward();
    }
}

//Arithmethic implementations
std::shared_ptr<ValueImpl> ValueImpl::operator+ (std::shared_ptr<ValueImpl> other)
{
    float new_data = this->data + other->get_data();
    std::vector<std::shared_ptr<ValueImpl>> children;
    std::shared_ptr<ValueImpl> child1 = shared_from_this();
    std::shared_ptr<ValueImpl> child2 = other;
    children.push_back(child1);
    children.push_back(child2);
    std::shared_ptr<ValueImpl> out = std::make_shared<ValueImpl> (new_data, children , op);
    out -> op = '+';
    std::weak_ptr<ValueImpl> out_weak = out;     // As here if we pass shared pointer to _backward the _backward will own out and out will own _backward so memory leak will happen
    out -> _backward = [child1, child2, out_weak](){
        if(auto out_ptr = out_weak.lock())
        {
            float grad1 = child1->get_grad(); 
            grad1 += 1.0f * out_ptr->get_grad();
            child1->set_grad(grad1);
            float grad2 = child2->get_grad();  
            grad2 += 1.0f * out_ptr->get_grad();
            child2->set_grad(grad2); 
        }
    };
    return out;
}

std::shared_ptr<ValueImpl> ValueImpl::operator- (std::shared_ptr<ValueImpl> other)
{
    float new_data = this->data - other->get_data();
    std::vector<std::shared_ptr<ValueImpl>> children;
    std::shared_ptr<ValueImpl> child1 = shared_from_this();
    std::shared_ptr<ValueImpl> child2 = other;
    children.push_back(child1);
    children.push_back(child2);
    std::shared_ptr<ValueImpl> out = std::make_shared<ValueImpl> (new_data, children , op);
    out -> op = '-';
    std::weak_ptr<ValueImpl> out_weak = out;     // As here if we pass shared pointer to _backward the _backward will own out and out will own _backward so memory leak will happen
    out -> _backward = [child1, child2, out_weak](){
        if(auto out_ptr = out_weak.lock())
        {
            float grad1 = child1->get_grad(); 
            grad1 += 1.0f * out_ptr->get_grad();
            child1->set_grad(grad1);
            float grad2 = child2->get_grad();  
            grad2 += (-1.0f) * out_ptr->get_grad();
            child2->set_grad(grad2); 
        }
    };
    return out;
}

std::shared_ptr<ValueImpl> ValueImpl::operator* (std::shared_ptr<ValueImpl> other)
{
    float new_data = this->data * other->get_data();
    std::vector<std::shared_ptr<ValueImpl>> children;
    std::shared_ptr<ValueImpl> child1 = shared_from_this();
    std::shared_ptr<ValueImpl> child2 = other;
    children.push_back(child1);
    children.push_back(child2);
    std::shared_ptr<ValueImpl> out = std::make_shared<ValueImpl> (new_data, children , op);
    out -> op = '*';
    std::weak_ptr<ValueImpl> out_weak = out;     // As here if we pass shared pointer to _backward the _backward will own out and out will own _backward so memory leak will happen
    out -> _backward = [child1, child2, out_weak](){
        if(auto out_ptr = out_weak.lock())
        {
            float grad1 = child1->get_grad(); 
            grad1 += child2->get_data() * out_ptr->get_grad();
            child1->set_grad(grad1);
            float grad2 = child2->get_grad(); 
            grad2 += child1->get_data() * out_ptr->get_grad();
            child2->set_grad(grad2); 
        }
    };
    return out;
}

std::shared_ptr<ValueImpl> ValueImpl::operator/ (std::shared_ptr<ValueImpl> other)
{
    float new_data = this->data / other->get_data();
    std::vector<std::shared_ptr<ValueImpl>> children;
    std::shared_ptr<ValueImpl> child1 = shared_from_this();
    std::shared_ptr<ValueImpl> child2 = other;
    children.push_back(child1);
    children.push_back(child2);
    std::shared_ptr<ValueImpl> out = std::make_shared<ValueImpl> (new_data, children , op);
    out -> op = '/';
    std::weak_ptr<ValueImpl> out_weak = out;     // As here if we pass shared pointer to _backward the _backward will own out and out will own _backward so memory leak will happen
    out -> _backward = [child1, child2, out_weak](){
        if(auto out_ptr = out_weak.lock())
        {
            float grad1 = child1->get_grad(); 
            grad1 += (1/child2->get_data()) * out_ptr->get_grad();
            child1->set_grad(grad1);
            float grad2 = child2->get_grad(); 
            grad2 += -((out_ptr->get_data())*(1/child2->get_data())) * out_ptr->get_grad();
            child2->set_grad(grad2); 
        }
    };
    return out;
}

std::shared_ptr<ValueImpl> ValueImpl::tanh()
{
    float new_data = std::tanh(this->data);
    std::vector<std::shared_ptr<ValueImpl>> children;
    std::shared_ptr<ValueImpl> child1 = shared_from_this();
    children.push_back(child1);
    std::shared_ptr<ValueImpl> out = std::make_shared<ValueImpl> (new_data, children , op);
    out -> op = "tanh";
    std::weak_ptr<ValueImpl> out_weak = out;     // As here if we pass shared pointer to _backward the _backward will own out and out will own _backward so memory leak will happen
    out -> _backward = [child1, out_weak](){
        if (auto out_ptr = out_weak.lock())
        {
            float t = (out_ptr->get_data());
            float grad1 = child1->get_grad(); 
            grad1 += (1-t*t) * out_ptr->get_grad() ;
            child1->set_grad(grad1);
        }
    };
    return out;
}

std::shared_ptr<ValueImpl> ValueImpl::relu()
{
    float new_data;
    if (this->data >= 0) {new_data = (this->data);}
    else {new_data = 0;}
    std::vector<std::shared_ptr<ValueImpl>> children;
    std::shared_ptr<ValueImpl> child1 = shared_from_this();
    children.push_back(child1);
    std::shared_ptr<ValueImpl> out = std::make_shared<ValueImpl> (new_data, children , op);
    out -> op = "relu";
    std::weak_ptr<ValueImpl> out_weak = out;     // As here if we pass shared pointer to _backward the _backward will own out and out will own _backward so memory leak will happen
    out -> _backward = [child1, out_weak](){
        if (auto out_ptr = out_weak.lock())
        {
            float grad1 = child1->get_grad(); 
            if (out_ptr->get_data() > 0) {grad1 += 1.0f * out_ptr->get_grad() ;}
            else {grad1 += 0.0f;}
            child1->set_grad(grad1);
        }
    };
    return out;
}

std::shared_ptr<ValueImpl> ValueImpl::leaky_relu(float alpha)
{
    float new_data;
    if (this->data >= 0) {new_data = (this->data);}
    else {new_data = alpha*(this->data);}
    std::vector<std::shared_ptr<ValueImpl>> children;
    std::shared_ptr<ValueImpl> child1 = shared_from_this();
    children.push_back(child1);
    std::shared_ptr<ValueImpl> out = std::make_shared<ValueImpl> (new_data, children , op);
    out -> op = "leaky_relu";
    std::weak_ptr<ValueImpl> out_weak = out;     // As here if we pass shared pointer to _backward the _backward will own out and out will own _backward so memory leak will happen
    out -> _backward = [child1, out_weak, alpha](){
        if (auto out_ptr = out_weak.lock())
        {
            float grad1 = child1->get_grad(); 
            if (out_ptr->get_data() > 0) {grad1 += 1.0f * out_ptr->get_grad() ;}
            else {grad1 += alpha * out_ptr->get_grad();}
            child1->set_grad(grad1);
        }
    };
    return out;
}

std::shared_ptr<ValueImpl> ValueImpl::exp()
{
    float new_data = std::exp(this->data);
    std::vector<std::shared_ptr<ValueImpl>> children;
    std::shared_ptr<ValueImpl> child1 = shared_from_this();
    children.push_back(child1);
    std::shared_ptr<ValueImpl> out = std::make_shared<ValueImpl> (new_data, children , op);
    out -> op = "exp";
    std::weak_ptr<ValueImpl> out_weak = out;     // As here if we pass shared pointer to _backward the _backward will own out and out will own _backward so memory leak will happen
    out -> _backward = [ child1, out_weak](){
        if (auto out_ptr = out_weak.lock())
        {
            float grad1 = child1->get_grad(); 
            grad1 += out_ptr->get_data() * out_ptr->get_grad() ;
            child1->set_grad(grad1);
        }
    };
    return out;
}
