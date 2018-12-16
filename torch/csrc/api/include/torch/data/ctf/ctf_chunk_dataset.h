#pragma once

#include <torch/data.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <torch/data/ctf/ctf_parser.h>

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
          std::vector<CTFExample<DataType>>,
          ChunkSampler,
          ExampleSampler> {
 public:
  using BatchType = std::vector<CTFExample<DataType>>;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  /// Loads multiple CTF files on multiple chunks with parallelization
  /// TODO: CTF files are not splitted, so they must fit in memory
  explicit CTFChunkDataset(
      std::vector<ctf::CTFConfigHelper> configs,
      size_t prefetch_count)
      : datasets::ChunkDataSet<
            CTFChunkDataset<DataType, ChunkSampler, ExampleSampler>,
            std::vector<CTFExample<DataType>>,
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
            std::vector<CTFExample<DataType>>,
            ChunkSampler,
            ExampleSampler>(1, false),
        chunk_sampler_(std::move(ChunkSampler(0))),
        example_sampler_(std::move(ExampleSampler(0))) {
    num_chunks_ = 1;
    configs_.push_back(config);
  }

  std::vector<CTFExample<DataType>> read_chunk(size_t chunk_index) override {
    // read file (which is a full chunk)
    ctf::CTFParser<DataType> ctf_parser(configs_[chunk_index]);
    ctf_parser.read_from_file();
    std::shared_ptr<CTFDataset<DataType>> ctf_dataset =
        ctf_parser.get_dataset();

    return std::move(ctf_dataset->examples);
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
