#ifndef VISUALIZE_H
#define VISUALIZE_H

#include "Value.h"

namespace autograd::utils {
    void build_trace(std::shared_ptr<ValueImpl> root, 
                 std::set<std::shared_ptr<ValueImpl>>& nodes, 
                 std::set<std::pair<std::shared_ptr<ValueImpl>, std::shared_ptr<ValueImpl>>>& edges);

    // Function to generate the Graphviz file
    void draw_graph(Value root, const std::string& filename = "my_network.dot");

    // Function to fill an array with all nodes of the graph
    void return_all_nodes(Value node, std::vector<Value>& node_arr);

    std::vector<std::pair<Value, Value>> zip(std::vector<Value> v1, std::vector<Value> v2);
}

#endif