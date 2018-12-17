#include <gtest/gtest.h>
#include <torch/data.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;

#include "htk_reader.h"

#include <fstream>
#include <chrono>

#include <string>

using namespace torch::data;

HTKParser::HTKParser(const string &file_directory, const string &file_set_name) : file_directory_(file_directory)
{
    parse_file_set(file_set_name);
}

void HTKParser::parse_file_set(const string &file_set_name)
{
    // Read json to get file infomation.
    // TODO: should we change this to txt file or add our own json parsing logic?
    ptree doc;
    read_json(file_set_name, doc);

    // Read the file extensions that needs to parse.
    for (auto &item : doc.get_child("fileType"))
    {
        file_extensions_.push_back(std::move(item.second.get_value<std::string>()));
    }

    // Get the chunk file name and example count for each chunk.
    for (auto &item : doc.get_child("fileInfo"))
    {
        ChunkInfo info;
        info.file_name = item.second.get<std::string>("name");
        info.chunk_size = item.second.get<size_t>("count");
        chunk_info_list_.push_back(std::move(info));
    }
}

std::vector<Utterance> HTKParser::parse_chunk(size_t chunk_id)
{
    std::vector<Utterance> result;

    if (chunk_id >= chunk_info_list_.size())
    {
        std::string error_msg = "chunk with id = " + std::to_string(chunk_id) + "does not exist.";
        throw std::runtime_error(error_msg);
    }

    for (auto &extension : file_extensions_)
    {
        std::string file_full_name = file_directory_ + "chunk" + std::to_string(chunk_id) + "." + extension;
        size_t example_count_in_chunk = chunk_info_list_[chunk_id].chunk_size;
        result.resize(example_count_in_chunk);

        if (extension == "feature")
        {
            parse_data<FeatureType>(file_full_name, "Feature", example_count_in_chunk, result);
        }
        else if (extension == "lattice")
        {
            parse_data<LatticeType>(file_full_name, "Lattice", example_count_in_chunk, result);
        }
        else if (extension == "label")
        {
            parse_data<LabelType>(file_full_name, "Label", example_count_in_chunk, result);
        }
        else
        {
            std::string error_msg = "unknown field " + extension;
            throw std::runtime_error(error_msg);
        }
    }

    return result;
}

void ValidateTagMatch(const char *data, const char *header, size_t dataLen, size_t headerLen)
{
    if (dataLen != headerLen)
    {
        std::string tag(header, headerLen);
        std::string actualData(data, dataLen);
        std::string error_msg = "Tag mismatch. Expected: " + tag + ", actual: " + actualData;
        throw std::runtime_error(error_msg);
    }
    int compare_result = std::memcmp(data, header, dataLen);
    if(compare_result != 0)
    {
        std::string tag(header, headerLen);
        std::string actualData(data, dataLen);
        std::string error_msg = "Tag mismatch. Expected: " + tag + ", actual: " + actualData;
        throw std::runtime_error(error_msg);
    }
}

template <
    typename ChunkSampler = samplers::SequentialSampler,
    typename ExampleSampler = samplers::SequentialSampler>
class HTKChunkDataset : public datasets::ChunkDataSet<
                            HTKChunkDataset<ChunkSampler, ExampleSampler>,
                            std::vector<Utterance>,
                            ChunkSampler,
                            ExampleSampler> {
 public:
  using BatchType = std::vector<Utterance>;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  HTKChunkDataset(
      size_t prefetch_count,
      ChunkSamplerType chunk_sampler,
      ExampleSamplerType example_sampler,
      const string& file_directory,
      const string& file_set_name,
      const size_t feature_dimension,
      const size_t label_dimension,
      const size_t truncated_length = 0)
      : datasets::ChunkDataSet<
            HTKChunkDataset<ChunkSampler, ExampleSampler>,
            std::vector<Utterance>,
            ChunkSamplerType,
            ExampleSamplerType>(prefetch_count, false),
        chunk_sampler_(std::move(chunk_sampler)),
        example_sampler_(std::move(example_sampler)),
        feature_dimension_(feature_dimension),
        label_dimension_(label_dimension),
        truncated_length_(truncated_length) {
    my_parser = torch::make_unique<HTKParser>(file_directory, file_set_name);
    num_chunks_ = my_parser->get_chunk_count();
  }

  ChunkSamplerType get_chunk_sampler() override {
    return chunk_sampler_;
  };

  ExampleSamplerType get_example_sampler() override {
    return example_sampler_;
  };

  size_t get_chunk_count() override {
    return num_chunks_;
  }

  BatchType read_chunk(size_t chunk_index) override {
    std::vector<Utterance> result = my_parser->parse_chunk(chunk_index);
    return result;
  }

private:
  size_t num_chunks_;

  std::unique_ptr<HTKParser> my_parser;

  ChunkSamplerType chunk_sampler_;
  ExampleSamplerType example_sampler_;

  // TODO: It is unclear to me which API the pyBind tie to and how flexible this
  // binded function can be. Ideally, reshape of the feature array should be the
  // last step and happen in the binded function. And this is also what CNTK
  // does -- internally the feature is always stored as a 1D array. We need to
  // explore more in pyBind and find out the right place to perform the final
  // shape massage.
  size_t feature_dimension_;
  size_t label_dimension_;
  size_t truncated_length_;
};

// Temporary helper method to output feature data. Used for data validation
// between pyTorch, CNTK and the converter.
void writeFeature(
    size_t seq_len,
    size_t feature_dim,
    std::string output,
    std::vector<Utterance>& chunk) {
  ofstream ss;
  ss.open(output, ios::out | std::ios::ate);
  ss << "feature output\n";
  for (int i = 0; i < chunk.size(); ++i) {
    ss << "[ ";
    for (int j = 0; j < seq_len; ++j) {
      ss << "  [ ";
      for (int w = 0; w < feature_dim; ++w) {
        ss << chunk[i].feature[j * feature_dim + w] << ", ";
      }
      ss << "]\n";
    }
    ss << "]\n";
    std::cout << "actual element for " << i << "'s sequence is "
              << chunk[i].feature.size() << endl;
  }
  ss.close();
}

// Temporary helper method to output label data. Used for data validation
// between pyTorch, CNTK and the converter.
void writeLabel(
    size_t seq_len,
    std::string output,
    std::vector<Utterance>& chunk) {
  ofstream ss;
  ss.open(output, ios::out | std::ios::ate);
  ss << "feature output\n";
  for (int i = 0; i < chunk.size(); ++i) {
    ss << "[ ";
    for (int j = 0; j < seq_len; ++j) {
      ss << chunk[i].label[j] << ", ";
    }
    ss << "]\n";
  }
  ss.close();
}

// Temporary validation test for HTK parser
TEST(DataTest, HTKParser) {
  HTKParser d("/home/xuzhu/data/htk_validation/generated/", "/home/xuzhu/data/htk_validation/generated/fileSet.json");
  auto chunk = d.parse_chunk(0);
  writeFeature(16, 80, "/home/xuzhu/data/htk_validation/generated/feature.txt", chunk);
  writeLabel(16, "/home/xuzhu/data/htk_validation/generated/label.txt", chunk);
}

TEST(DataTest, HTKChunkDataset) {
  size_t batch_size = 10;
  datasets::SharedBatchDataset<
      HTKChunkDataset<samplers::SequentialSampler, samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<HTKChunkDataset<
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          1,
          samplers::SequentialSampler(0),
          samplers::SequentialSampler(0),
          "/home/xuzhu/data/pytorch_unittest/",
          "/home/xuzhu/data/pytorch_unittest/fileSet.json",
          33,
          1000);

  auto data_loader = torch::data::make_chunk_data_loader(
      dataset, DataLoaderOptions(batch_size).workers(0).chunk_loading(true));

  dataset->reset();

  auto iterator = data_loader->begin();
  for (size_t i = 0; i < 2; ++i, ++iterator) {
    std::vector<Utterance> batch = *iterator;
    ASSERT_EQ(batch.size(), batch_size);
  }
}

TEST(DataTest, HTKChunkDatasetVerifyResult) {
  size_t batch_size = 300;
  datasets::SharedBatchDataset<
      HTKChunkDataset<samplers::SequentialSampler, samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<HTKChunkDataset<
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          1,
          samplers::SequentialSampler(0),
          samplers::SequentialSampler(0),
          "/home/xuzhu/data/converter_output/smalldata/",
          "/home/xuzhu/data/converter_output/smalldata/fileSet.json",
          33,
          1000);

  auto data_loader = torch::data::make_chunk_data_loader(
      dataset, DataLoaderOptions(batch_size).workers(0).chunk_loading(true));

  dataset->reset();

  ofstream featureFile("/home/xuzhu/data/converter_output/feature_pytorch.txt", ios::out);
  ofstream labelFile("/home/xuzhu/data/converter_output/label_pytorch.txt", ios::out);

  featureFile << std::fixed;
  featureFile << std::setprecision(2);

  auto iterator = data_loader->begin();
  size_t feature_count =0;
  size_t label_count = 0;
  for (size_t i = 0; i < 4; ++i) {
    std::vector<Utterance> batch = *iterator;

    for(auto& example : batch)
    {
        featureFile<<"feature "<<feature_count<< " size: "<<example.feature.size()<<":\n";
        feature_count++;
        for(int i=0;i<example.feature.size();++i)
        {
            featureFile << example.feature[i]<<" ";
            if ((i+1)%33 == 0)
            {
                featureFile <<"\n";
            }
        }
        featureFile<<"\n";

        labelFile<<"label "<<label_count<< " size: "<<example.label.size()<<":\n";
        label_count++;
        for(int i=0;i<example.label.size();++i)
        {
            labelFile << example.label[i]<<" ";
        }
        labelFile<<"\n";
    }
    ++iterator;
  }
}


TEST(DataTest, HTKChunkDatasetLargeData) {
  int dataset_worker = 4;
  int dataloader_worker = 4;

  // 4g data, 64197 features in total
  size_t batch_size = 2048;
  size_t total_count = 64197;
  datasets::SharedBatchDataset<
      HTKChunkDataset<samplers::SequentialSampler, samplers::SequentialSampler>>
      dataset = datasets::make_shared_dataset<HTKChunkDataset<
          samplers::SequentialSampler,
          samplers::SequentialSampler>>(
          dataset_worker,
          samplers::SequentialSampler(0),
          samplers::SequentialSampler(0),
          "/home/xuzhu/data/converter_output/150/",
          "/home/xuzhu/data/converter_output/150/fileSet_4g.json",
          80,
          9404);

  auto data_loader = torch::data::make_chunk_data_loader(
      dataset, DataLoaderOptions(batch_size).workers(dataloader_worker).chunk_loading(true));

  auto start = std::chrono::high_resolution_clock::now();
  dataset->reset();

  auto iterator = data_loader->begin();
  int iterator_count = ((total_count + batch_size -1)/batch_size);
  for (size_t i = 0; i < (iterator_count+2); ++i) {
    std::vector<Utterance> batch = *iterator;
    //ASSERT_EQ(batch.size(), batch_size);
    std::cout << "iteration "<<i<<"with count " << batch.size() << "\n";
    if(i ==0 || batch.size() == (709))
    {
        std::cout << "iteration "<<i<<"with feature "<<batch[0].feature[0]<<" "<<batch[0].feature[1]<<"\n ";
        std::cout << "iteration "<<i<<"with label "<<batch[0].label[0]<<" "<<batch[0].label[1]<<"\n ";
    }
    ++iterator;
  }
  auto finish = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = finish - start;
  std::cout << "finished loading "<<iterator_count<<"batches. pyTorch Elapsed time: " << elapsed.count() << " s\n";
}