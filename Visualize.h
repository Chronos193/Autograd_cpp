#ifndef VISUALIZE_H
#define VISUALIZE_H

#include "Value.h"

namespace autograd::utils {
    void build_trace(std::shared_ptr<Value> root, 
                 std::set<std::shared_ptr<Value>>& nodes, 
                 std::set<std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>>& edges);

    // Function to generate the Graphviz file
    void draw_graph(std::shared_ptr<Value> root, const std::string& filename = "graph.dot");

    // Function to fill an array with all nodes of the graph
    void return_all_nodes(std::shared_ptr<Value> node, std::vector<std::shared_ptr<Value>>& node_arr);
}

#endif