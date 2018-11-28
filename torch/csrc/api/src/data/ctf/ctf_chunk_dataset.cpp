#include <torch/data/ctf/ctf_chunk_dataset.h>
#include <torch/data/ctf/ctf_parser.h>

namespace torch {
namespace data {
namespace ctf {

/// Loads multiple CTF files on multiple chunks with parallelization
/// TODO: CTF files are not splitted, so they must fit in memory
CTFChunkDataset::CTFChunkDataset(
    std::vector<ctf::CTFConfigHelper> configs,
    size_t prefetch_count)
    : datasets::ChunkDataSet<
          CTFChunkDataset,
          std::vector<Example<
              std::vector<ctf::CTFSample>,
              std::vector<ctf::CTFSample>>>,
          samplers::RandomSampler,
          samplers::RandomSampler>(prefetch_count, false),
      configs_(configs),
      chunk_sampler_(std::move(samplers::RandomSampler(0))),
      example_sampler_(std::move(samplers::RandomSampler(0))) {
  num_chunks_ = configs.size();
  chunk_sampler_.reset(num_chunks_);
  example_sampler_.reset(100);
}

/// Loads a single CTF file on a single chunk without parallelization
/// TODO: CTF files are not splitted, so they must fit in memory
CTFChunkDataset::CTFChunkDataset(ctf::CTFConfigHelper config)
    : datasets::ChunkDataSet<
          CTFChunkDataset,
          std::vector<Example<
              std::vector<ctf::CTFSample>,
              std::vector<ctf::CTFSample>>>,
          samplers::RandomSampler,
          samplers::RandomSampler>(1, false),
      chunk_sampler_(std::move(samplers::RandomSampler(0))),
      example_sampler_(std::move(samplers::RandomSampler(0))) {
  num_chunks_ = 1;
  chunk_sampler_.reset(num_chunks_);
  example_sampler_.reset(100);
  configs_.push_back(config);
}

std::vector<Example<std::vector<ctf::CTFSample>, std::vector<ctf::CTFSample>>>
CTFChunkDataset::read_chunk(size_t chunk_index) {
  // TODO: reserve maximum size for this chunk?
  std::vector<Example<std::vector<ctf::CTFSample>, std::vector<ctf::CTFSample>>>
      batch;

  ctf::CTFParser ctf_parser(configs_[chunk_index]);
  ctf_parser.read_from_file();
  const ctf::CTFDataset& ctf_dataset = ctf_parser.get_dataset();

  for (auto it_features = ctf_dataset.features.begin();
       it_features != ctf_dataset.features.end();
       ++it_features) {
    Example<std::vector<ctf::CTFSample>, std::vector<ctf::CTFSample>> example(
        std::move(it_features->second.samples),
        std::move(ctf_dataset.labels.at(it_features->first).samples));
    batch.push_back(std::move(example));
  }

  return batch;
}

samplers::RandomSampler CTFChunkDataset::get_chunk_sampler() {
  return chunk_sampler_;
}

samplers::RandomSampler CTFChunkDataset::get_example_sampler() {
  return example_sampler_;
}

size_t CTFChunkDataset::get_chunk_count() {
  return num_chunks_;
}

} // namespace ctf
} // namespace data
} // namespace torch
