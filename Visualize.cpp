#include "visualize.h"

namespace autograd::utils {
    void build_trace(std::shared_ptr<Value> root, 
                 std::set<std::shared_ptr<Value>>& nodes, 
                 std::set<std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>>& edges) 
    {
        // If we haven't seen this node yet
        if (nodes.find(root) == nodes.end()) {
            nodes.insert(root);
            for (const auto& child : root->get_childen()) {
                edges.insert({child, root}); // Record the connection
                build_trace(child, nodes, edges);
            }
        }
    }

    // Function to generate the Graphviz file
    void draw_graph(std::shared_ptr<Value> root, const std::string& filename) 
    {
        std::set<std::shared_ptr<Value>> nodes;
        std::set<std::pair<std::shared_ptr<Value>, std::shared_ptr<Value>>> edges;
        
        build_trace(root, nodes, edges);

        std::ofstream out(filename);
        out << "digraph G {\n";
        out << "  rankdir=\"LR\";\n"; // Draws the graph Left-to-Right

        // 1. Define all the nodes
        for (const auto& node : nodes) {
            // Create a label containing the Op (if any), Data, and Grad
            std::string op_text = node->get_op().empty() ? "" : (node->get_op() + " | ");
            
            out << "  \"" << node.get() << "\" [shape=record, label=\"{ " 
                << op_text
                << "data: " << node->get_data() 
                << " | grad: " << node->get_grad() 
                << " }\"];\n";
        }

        // 2. Define all the edges (connections)
        for (const auto& edge : edges) {
            out << "  \"" << edge.first.get() << "\" -> \"" << edge.second.get() << "\";\n";
        }

        out << "}\n";
        out.close();
        std::cout << "Graph saved to " << filename << std::endl;
    }
    void return_all_nodes(std::shared_ptr<Value> node, std::vector<std::shared_ptr<Value>>& node_arr)
    {
        auto children = node->get_childen();
        for(const auto& child: children)
        {
            return_all_nodes(child, node_arr);
        }
        node_arr.push_back(node);
    }
}