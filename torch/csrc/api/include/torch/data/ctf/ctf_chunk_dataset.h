#pragma once

#include <torch/data.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <torch/data/ctf/ctf_parser.h>

namespace torch {
namespace data {
namespace ctf {

class CTFChunkDataset : public datasets::ChunkDataSet<
                            CTFChunkDataset,
                            std::vector<Example<
                                std::vector<ctf::CTFSample>,
                                std::vector<ctf::CTFSample>>>,
                            samplers::RandomSampler,
                            samplers::RandomSampler> {
 public:
  using BatchType = std::vector<
      Example<std::vector<ctf::CTFSample>, std::vector<ctf::CTFSample>>>;
  using BatchRequestType = size_t;

  explicit CTFChunkDataset(std::vector<ctf::CTFConfigHelper> configs, size_t prefetch_count);
  explicit CTFChunkDataset(ctf::CTFConfigHelper config);

  std::vector<Example<std::vector<ctf::CTFSample>, std::vector<ctf::CTFSample>>>
  read_chunk(size_t chunk_index) override;

  samplers::RandomSampler get_chunk_sampler() override;

  samplers::RandomSampler get_example_sampler() override;

  size_t get_chunk_count() override;

 private:
  std::vector<ctf::CTFConfigHelper> configs_;
  // TODO: is num_chunks_ needed?
  size_t num_chunks_;
  samplers::RandomSampler chunk_sampler_;
  samplers::RandomSampler example_sampler_;
};

} // namespace ctf
} // namespace data
} // namespace torch
