#pragma once

#include <torch/data.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <torch/data/ctf/ctf_parser.h>

#include <atomic>
#include <chrono>
#include <utility>
using namespace std::chrono;

namespace torch {
namespace data {
namespace ctf {

template <
    typename DataType = double,
    typename ChunkSampler = samplers::RandomSampler,
    typename ExampleSampler = samplers::RandomSampler>
class CTFChunkDataset
    : public datasets::ChunkDataSet<
          CTFChunkDataset<DataType, ChunkSampler, ExampleSampler>,
          std::vector<Example<
              std::vector<ctf::CTFSample<DataType>>,
              std::vector<ctf::CTFSample<DataType>>>>,
          ChunkSampler,
          ExampleSampler> {
 public:
  using BatchType = std::vector<Example<
      std::vector<ctf::CTFSample<DataType>>,
      std::vector<ctf::CTFSample<DataType>>>>;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  /// Loads multiple CTF files on multiple chunks with parallelization
  /// TODO: CTF files are not splitted, so they must fit in memory
  explicit CTFChunkDataset(
      std::vector<ctf::CTFConfigHelper> configs,
      size_t prefetch_count)
      : datasets::ChunkDataSet<
            CTFChunkDataset<DataType, ChunkSampler, ExampleSampler>,
            std::vector<Example<
                std::vector<ctf::CTFSample<DataType>>,
                std::vector<ctf::CTFSample<DataType>>>>,
            ChunkSampler,
            ExampleSampler>(prefetch_count, false),
        configs_(configs),
        chunk_sampler_(std::move(ChunkSampler(0))),
        example_sampler_(std::move(ExampleSampler(0))) {
    num_chunks_ = configs.size();
  }

  /// Loads a single CTF file on a single chunk without parallelization
  /// TODO: CTF files are not splitted, so they must fit in memory
  explicit CTFChunkDataset(ctf::CTFConfigHelper config)
      : datasets::ChunkDataSet<
            CTFChunkDataset<DataType, ChunkSampler, ExampleSampler>,
            std::vector<Example<
                std::vector<ctf::CTFSample<DataType>>,
                std::vector<ctf::CTFSample<DataType>>>>,
            ChunkSampler,
            ExampleSampler>(1, false),
        chunk_sampler_(std::move(ChunkSampler(0))),
        example_sampler_(std::move(ExampleSampler(0))) {
    num_chunks_ = 1;
    configs_.push_back(config);
  }

  std::vector<Example<
      std::vector<ctf::CTFSample<DataType>>,
      std::vector<ctf::CTFSample<DataType>>>>
  read_chunk(size_t chunk_index) override {
    static std::atomic<unsigned long int> read_time(0);
    static std::atomic<unsigned long int> convert_time(0);

    auto start1 = high_resolution_clock::now();
    ctf::CTFParser<DataType> ctf_parser(configs_[chunk_index]);
    ctf_parser.read_from_file();
    std::shared_ptr<CTFDataset<DataType>> ctf_dataset =
        ctf_parser.get_dataset();
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<microseconds>(stop1 - start1);
    read_time += duration1.count();

    // TODO: reserve maximum size for this chunk?
    std::vector<Example<
        std::vector<ctf::CTFSample<DataType>>,
        std::vector<ctf::CTFSample<DataType>>>>
        batch;
    batch.reserve(ctf_dataset->features.size());

    auto start2 = high_resolution_clock::now();
    for (auto& it_features : ctf_dataset->features) {
      //   Example<
      //       std::vector<ctf::CTFSample<DataType>>,
      //       std::vector<ctf::CTFSample<DataType>>>
      //       example;

      // std::swap(example.data, it_features.second.samples);
      // std::swap(example.target,
      // ctf_dataset->labels.at(it_features.first).samples);

      //   example.data = std::move(it_features.second.samples);
      //   example.target =
      //       std::move(ctf_dataset->labels.at(it_features.first).samples);
      //   batch.emplace_back(example);
      batch.emplace_back(
          it_features.second.samples,
          ctf_dataset->labels.at(it_features.first).samples);
    }

    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<microseconds>(stop2 - start2);
    convert_time += duration2.count();

    std::cout << "File: " << configs_[chunk_index].get_file_path() << "("
              << chunk_index << ")" << std::endl;
    std::cout << "Read from file: " << duration1.count() << " microseconds ("
              << read_time << ")" << std::endl;
    std::cout << "Convert format: " << duration2.count() << " microseconds ("
              << convert_time << ")" << std::endl;
    return batch;
  }

  ChunkSampler get_chunk_sampler() override {
    return chunk_sampler_;
  }

  ExampleSampler get_example_sampler() override {
    return example_sampler_;
  }

  size_t get_chunk_count() override {
    return num_chunks_;
  }

 private:
  std::vector<ctf::CTFConfigHelper> configs_;
  size_t num_chunks_;
  ChunkSampler chunk_sampler_;
  ExampleSampler example_sampler_;
};

} // namespace ctf
} // namespace data
} // namespace torch
