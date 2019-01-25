#include <iostream>

#include <c10/util/ArrayRef.h>
#include <torch/data.h>
#include <torch/data/ctf/ctf_chunk_dataset.h>
#include <torch/data/detail/sequencers.h>
#include <torch/serialize.h>
#include <torch/types.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <chrono>
#include <utility>
using namespace std::chrono;

int main(int argc, char* argv[]) {
  const size_t batch_size = 2048;
  const size_t total_files = 50;
  const size_t total_example = 5662308;
  const size_t total_prefetch = 5;
  const size_t total_worker = 5;
  const size_t total_batch = 1 + ((total_example - 1) / batch_size);

  std::cout << "Input parameters" << std::endl;
  std::cout << "Batch size: " << batch_size << std::endl
            << "CTF files: " << total_files << std::endl
            << "Examples: " << total_example << std::endl
            << "Prefetch: " << total_prefetch << std::endl
            << "Workers: " << total_worker << std::endl
            << "Batches: " << total_batch << std::endl;

  std::vector<torch::data::ctf::CTFConfiguration> configs;
  std::vector<torch::data::ctf::CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(0,
      "M", "M", 0, torch::data::ctf::CTFInputStreamType::Feature, torch::data::ctf::CTFDataStorage::Sparse);
  input_streams.emplace_back(1,
      "R", "R", 0, torch::data::ctf::CTFInputStreamType::Label, torch::data::ctf::CTFDataStorage::Sparse);

  for (size_t i = 0; i < total_files; ++i) {
    std::ostringstream ss;
    ss << std::setw(10) << std::setfill('0') << i;
    std::string str = ss.str();

    torch::data::ctf::CTFConfiguration config(
        std::string(argv[1] + str + ".ctf"),
        input_streams,
        torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Double));

    configs.push_back(config);
  }

  torch::data::datasets::SharedBatchDataset<torch::data::ctf::CTFChunkDataset<
      double,
      torch::data::samplers::RandomSampler,
      torch::data::samplers::RandomSampler>>
      shared_dataset = torch::data::datasets::make_shared_dataset<
          torch::data::ctf::CTFChunkDataset<
              double,
              torch::data::samplers::RandomSampler,
              torch::data::samplers::RandomSampler>>(configs, total_prefetch);

  auto data_loader = torch::data::make_chunk_data_loader(
      shared_dataset,
      torch::data::DataLoaderOptions()
          .workers(total_worker)
          .batch_size(batch_size)
          .chunk_loading(true));

  size_t count_examples = 0;
  shared_dataset->reset();

  size_t read_time1 = 0;
  auto start1 = high_resolution_clock::now();
  auto iterator = data_loader->begin();
  for (size_t i = 0; i < total_batch; ++i, ++iterator) {
    count_examples += iterator->size();
  }
  auto stop1 = high_resolution_clock::now();
  auto duration1 = duration_cast<microseconds>(stop1 - start1);
  read_time1 += duration1.count();
  std::cout << "Total chunking time: " << std::fixed << std::setw(11)
            << std::setprecision(6) << read_time1 / 1000000.0 << std::endl;

  std::cout << count_examples << " examples were loaded!" << std::endl;

  return EXIT_SUCCESS;
}