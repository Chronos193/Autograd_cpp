#include "dataset.h"
#include "Visualize.h"


std::string CSVDataset::get_filename()
{
    return filename;
}

size_t CSVDataset::size() const
{
    return doc.GetRowCount();
}

std::pair<std::vector<Value>,std::vector<Value>> CSVDataset::get(size_t idx) const
{
    std::vector<Value> X;
    std::vector<Value> Y;
    for(const auto& col_idx: x_col_indexs)
    {
        X.emplace_back(doc.GetCell<float>(col_idx, idx));
    }
    for(const auto& col_idx: y_col_indexs)
    {
        Y.emplace_back(doc.GetCell<float>(col_idx, idx));
    }
    return {X,Y};
}

void CSVDataset::info()
{
    std::cout << "Number of Columns :- " << doc.GetColumnCount() << std::endl;
    std::cout << "Number of Rows :- " << doc.GetRowCount() << std::endl;
}

// DataLoader

void DataLoader::shuffle_indices()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(index_arr.begin(), index_arr.end(), gen);
}

size_t DataLoader::size() const
{
    size_t total = dataset.size();
    return (total + batch_size - 1) / batch_size;
}

std::vector<std::pair<std::vector<Value>,std::vector<Value>>> DataLoader::get_batch(size_t batch_idx) const
{
    std::vector<std::pair<std::vector<Value>,std::vector<Value>>> batch_arr;

    size_t start = batch_idx * batch_size;
    size_t end = start + batch_size;
    if(end > dataset.size())
    {
        end = dataset.size();
    }

    for(size_t i=start;i<end;i++)
    {
        batch_arr.push_back(dataset.get(index_arr[i]));
    }
    return batch_arr;
}

void DataLoader::reset()
{
    if(shuffle)
    {
        shuffle_indices();
    }
}

void DataLoader::info()
{
    std::cout << "Dataset size :- " << dataset.size() << std::endl;
    std::cout << "Batch size :- " << batch_size << std::endl;
    std::cout << "Number of batches :- " << size() << std::endl;
    std::cout << "Shuffle :- " << (shuffle ? "true" : "false") << std::endl;
}
