#include "rapidcsv.h"

int main() {
    // Load the file
    rapidcsv::Document doc("data.csv");

    // Access data by column name
    std::vector<std::string> col = doc.GetColumn<std::string>("col2");
    int col_sum = 0;
    for(auto& ele: col)
    {
        col_sum += std::stoi(ele);
    }
    std::cout << col_sum << std::endl;
    // Access data by row and column index
    int num_rows = doc.GetColumnCount();

    std::cout << num_rows << std::endl;
    
    return 0;
}