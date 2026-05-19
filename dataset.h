#include<iostream>
#include<vector>
#include<string>
#include<random>
#include<algorithm>
#include"Value.h"
#include"rapidcsv.h"

class Dataset {
public:
    virtual size_t size() const = 0;

    virtual std::pair<std::vector<Value>,std::vector<Value>> get(size_t idx) const = 0;
};

class CSVDataset: public Dataset
{
    private:
        std::string filename;
        std::vector<int> x_col_indexs;
        std::vector<int> y_col_indexs;
        rapidcsv::Document doc;
    public:
        CSVDataset(std::string name, std::vector<int> x_col_indexs, std::vector<int> y_col_indexs): filename(name), x_col_indexs(x_col_indexs), y_col_indexs(y_col_indexs) 
        {
            rapidcsv::Document doc(filename);
            this->doc = doc;
        }

        std::string get_filename();

        size_t size() const;

        std::pair<std::vector<Value>,std::vector<Value>> get(size_t idx) const;

        void info();
};


class DataLoader
{
    private:
        Dataset& dataset;
        int batch_size;
        bool shuffle;
        std::vector<size_t> index_arr;

    public:
        DataLoader(Dataset& dataset, int batch_size, bool shuffle = false): dataset(dataset), batch_size(batch_size), shuffle(shuffle)
        {
            index_arr.reserve(dataset.size());
            for(size_t i=0;i<dataset.size();i++)
            {
                index_arr.push_back(i);
            }
            if(shuffle)
            {
                shuffle_indices();
            }
        }

        void shuffle_indices();

        size_t size() const; // total number of batches

        std::vector<std::pair<std::vector<Value>,std::vector<Value>>> get_batch(size_t batch_idx) const; // get a batch by index

        void reset(); // reshuffle for next epoch

        void info();
};