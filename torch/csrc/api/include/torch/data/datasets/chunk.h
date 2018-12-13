#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/utils/memory.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/data/samplers.h>
#include <torch/data/worker_exception.h>

#include <atomic>
#include <string>
#include <thread>

namespace torch {
namespace data {
namespace datasets {

/// Interface for chunk reader, which performs data chunking and reading of
/// entire chunks.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text-file split based on seek
/// positions.
template <typename Self, typename Batch = std::vector<Example<>>>
class ChunkDataReader {
 public:
  using SelfType = Self;
  using BatchType = Batch;

  /// Read an entire chunk.
  virtual BatchType read_chunk(size_t chunk_index) = 0;

  /// Returns the number of chunks available in this reader.
  virtual size_t get_chunk_count() = 0;

  /// This will clear any internal state associate with this reader.
  virtual void reset() = 0;
};

/// A class that contains a raw unwrapped batch unit. An unwrapped batch unit is
/// the raw data without 'optional' wrapper. It can be a collection of images,
/// utterances, e.t.c.
template <typename UnwrappedBatch = std::vector<Example<>>>
struct UnwrappedBatchData {
 public:
  using UnwrappedBatchType = UnwrappedBatch;

  UnwrappedBatchData(UnwrappedBatchType data) : batch_data(std::move(data)) {}

  UnwrappedBatchData(std::exception_ptr e) : exception(e) {}

  /// batch data to return
  UnwrappedBatchType batch_data;

  /// exception pointer which captures any abnormal exceptions while loading the
  /// chunk.
  std::exception_ptr exception;
};

/// BatchDataBuffer manages a queue of UnwrappedBatchData. It fetches batch data
/// for ChunkDataSet, tracks when new UnwrappedBatchData is needed, and when all
/// chunk are loaded.
template <
    typename UnwrappedBatch = std::vector<Example<>>,
    typename ExampleSampler = samplers::RandomSampler>
class BatchDataBuffer {
 public:
  using UnwrappedBatchType = UnwrappedBatch;
  using BatchRequestType = typename ExampleSampler::BatchRequestType;

  BatchDataBuffer(
      size_t num_chunks,
      size_t batch_size,
      ExampleSampler example_sampler,
      bool ignore_empty_chunk,
      size_t cache_size)
      : remaining_chunk_count_(num_chunks),
        batch_size_(batch_size),
        example_sampler_(std::move(example_sampler)),
        ignore_empty_chunk_(ignore_empty_chunk),
        queue_depth_(cache_size) {}

  /// Return batch data from the queue. Called from the ChunkDataSet main
  /// thread.
  UnwrappedBatchType get_batch(size_t batch_size) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cvr_.wait(lock, [this] {
      // wait till there is available data in the queue or if all chunks are
      // loaded (i.e. the data set is exhausted for this epoch)
      return (
          this->total_example_count_in_queue_ >= batch_size_ ||
          remaining_chunk_count_ == 0);
    });

    if (batch_queue_.empty()) {
      lock.unlock();

      AT_ASSERT(remaining_chunk_count_ == 0);

      // All chunks are loaded. Return the remaining data if there is any as
      // the last batch, and wait for a reset() to restart.
      return {};
    }

    auto batch_data = batch_queue_.front();
    batch_queue_.pop();
    if (batch_data.exception) {
      throw WorkerException(batch_data.exception);
    }

    this->total_example_count_in_queue_ -= batch_data.batch_data.size();
    lock.unlock();
    cvw_.notify_all(); // notify all writers.

    return batch_data.batch_data;
  }

  // skip one chunk
  void skip_chunk() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    AT_ASSERT(remaining_chunk_count_ > 0);
    remaining_chunk_count_--;
    lock.unlock();
    cvr_.notify_all();
  }

  /// Push preloaded chunks to chunk queue. Called from the ChunkDataSet worker
  /// threads.
  void add_chunk_data(size_t index, UnwrappedBatchType data) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cvw_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < queue_depth_;
    });

    auto data_size = data.size();
    auto remaining_size = data_size;
    example_sampler_.reset(data_size);

    if (!batch_queue_.empty()) {
      auto& batch_data = batch_queue_.back();
      size_t current_count = batch_data.batch_data.size();
      if (current_count < batch_size_) {
        auto example_count =
            std::min(remaining_size, batch_size_ - current_count);
        auto batch_example_indices = example_sampler_.next(example_count);
        AT_ASSERT(
            batch_example_indices &&
            batch_example_indices.value().size() == example_count)
        BatchRequestType indices = batch_example_indices.value();
        for (size_t i : indices) {
          batch_data.batch_data.emplace_back(data[i]);
        }
        remaining_size -= example_count;
      }
    }
    while (remaining_size > 0) {
      UnwrappedBatchType current_batch;
      auto example_count = std::min(remaining_size, batch_size_);
      current_batch.reserve(example_count);
      auto batch_example_indices = example_sampler_.next(example_count);
      AT_ASSERT(
          batch_example_indices &&
          batch_example_indices.value().size() == example_count)
      BatchRequestType indices = batch_example_indices.value();
      for (size_t i : indices) {
        current_batch.emplace_back(data[i]);
      }
      remaining_size -= example_count;
      UnwrappedBatchData<UnwrappedBatchType> chunk_data(
          std::move(current_batch));
      batch_queue_.push(std::move(chunk_data));
    }
    this->total_example_count_in_queue_ += data_size;

    // change to unloaded chunk count
    this->remaining_chunk_count_--;

    lock.unlock();
    cvr_.notify_all();
  }

  /// Push exceptions throwed during preloading into chunk queue. Called from
  /// the ChunkDataSet worker threads.
  void add_chunk_data(size_t index, std::exception_ptr e_ptr) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cvw_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < queue_depth_;
    });
    UnwrappedBatchData<UnwrappedBatchType> chunk_data(e_ptr);
    batch_queue_.push(std::move(chunk_data));

    remaining_chunk_count_--;
    lock.unlock();
    cvr_.notify_all(); // notify all readers.
  }

  size_t total_example_count_in_queue_ = 0;
  size_t remaining_chunk_count_ = 0;
  size_t batch_size_ = 0;
  std::queue<UnwrappedBatchData<UnwrappedBatchType>> batch_queue_;

  // sync batch_queue_ update.
  std::mutex queue_mutex_;

  std::condition_variable cvr_;
  std::condition_variable cvw_;

  ExampleSampler example_sampler_;
  bool ignore_empty_chunk_ = false;

  size_t queue_depth_;
};

/// A stateful dataset that support hierarchical sampling and prefetching of
/// entre chunks. dataset that supports loading an entire chunk of data.
///
/// Unlike regular dataset, chunk dataset require two samplers to operate and
/// keeps an internal state. `ChunkSampler` selects, which chunk to load next,
/// while the `ExampleSampler` determins the order of Examples that are returned
/// in each `get_batch` call. The hierarchical sampling approach used here is
/// inspired by this paper http://martin.zinkevich.org/publications/nips2010.pdf
template <
    typename ChunkReader,
    typename ChunkSampler = samplers::RandomSampler,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataSet final
    : public BatchDataset<
          ChunkDataSet<ChunkReader, ChunkSampler, ExampleSampler>,
          torch::optional<typename ChunkReader::BatchType>,
          size_t> {
 public:
  using BatchType = torch::optional<typename ChunkReader::BatchType>;

  using UnwrappedBatchType = typename ChunkReader::BatchType;

  using BatchRequestType = size_t;

  using ChunkSamplerType = ChunkSampler;

  using ExampleSamplerType = ExampleSampler;

  ChunkDataSet(
      ChunkReader chunk_reader,
      ChunkSampler chunk_sampler,
      ExampleSampler example_sampler,
      size_t preloader_count,
      size_t batch_size,
      bool ignore_empty_chunk = false,
      size_t cache_size = 500)
      : chunk_reader_(std::move(chunk_reader)),
        chunk_sampler_(std::move(chunk_sampler)),
        example_sampler_(std::move(example_sampler)),
        preloader_count_(preloader_count),
        batch_size_(batch_size),
        ignore_empty_chunk_(ignore_empty_chunk),
        cache_size_(cache_size) {
    if (preloader_count_ == 0) {
      throw std::runtime_error(
          "preloader_count is 0. At least one preloader needs to be specified.");
    }

    if (batch_size == 0) {
      throw std::runtime_error(
          "batch size is 0. A positive batch size needs to be specified.");
    }

    if (cache_size_ == 0) {
      throw std::runtime_error(
          "cache size is 0. A positive cache size needs to be specified.");
    }
  }

  ChunkDataSet(ChunkDataSet&& data_set) {
    preloader_count_ = data_set.preloader_count_;
    chunk_sampler_ = std::move(data_set.chunk_sampler_);
    quit_worker_ = data_set.quit_worker_.load();
  }

  virtual ~ChunkDataSet() {
    free_workers();
  }

  /// Default get_batch method of BatchDataSet. This method will handle the
  /// chunk loading and creating of Example batches. The implemenation is
  /// dataset agnostic and does not need overriding in different chunk data
  /// sets.
  BatchType get_batch(size_t batch_size) override {
    if (chunk_buffer_ == nullptr) {
      throw std::runtime_error(
          "Dataset needs to call reset() before calling get_batch().");
    }
    if (batch_size != batch_size_) {
      std::string error =
          "The requested batch size does not match with the initialized batch size.\n The requested batch size is " +
          std::to_string(batch_size) +
          ", while the data set is created with batch size equal to " +
          std::to_string(batch_size_);

      throw std::runtime_error(error);
    }
    return chunk_buffer_->get_batch(batch_size);
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset. It simply starts a mini dataloader.
  virtual void reset() {
    // free any created worker from previous reset.
    free_workers();
    preload_threads_.clear();

    size_t chunks_to_load = chunk_reader_.get_chunk_count();
    chunk_sampler_.reset(chunks_to_load);

    // Creates a new chunk buffer each time we reset the dataset.
    chunk_buffer_ = torch::make_unique<
        BatchDataBuffer<UnwrappedBatchType, ExampleSamplerType>>(
        chunks_to_load,
        batch_size_,
        example_sampler_,
        ignore_empty_chunk_,
        cache_size_);

    // create new workers for this new epoch.
    quit_worker_ = false;

    for (size_t i = 0; i < preloader_count_; ++i) {
      preload_threads_.emplace_back(
          [this, i]() mutable { this->preloader(i); });
    }
  }

  /// size is not used for chunk dataset.
  optional<size_t> size() const override {
    return torch::nullopt;
  }

 private:
  /// running on worker thread to preload chunk data.
  void preloader(size_t id) {
    while (!quit_worker_.load()) {
      size_t chunk_id;
      try {
        auto chunk_sampler_result = chunk_sampler_.next(1);
        if (chunk_sampler_result.has_value()) {
          chunk_id = chunk_sampler_result.value()[0];
        } else {
          break;
        }
        UnwrappedBatchType data = chunk_reader_.read_chunk(chunk_id);
        if (data.empty()) {
          if (!ignore_empty_chunk_) {
            std::string error =
                "Chunk with index " + std::to_string(chunk_id) + " is empty";
            throw std::runtime_error(error);
          } else {
            // skip adding the current chunk data and move to the next.
            chunk_buffer_->skip_chunk();
          }
        }

        else {
          chunk_buffer_->add_chunk_data(chunk_id, std::move(data));
        }
      } catch (...) {
        chunk_buffer_->add_chunk_data(chunk_id, std::current_exception());
      }
    }
  }

  /// Block the current thread until the workers finish execution and exit.
  void free_workers() {
    if (!quit_worker_.load()) {
      quit_worker_ = true;
      for (auto& worker_thread : preload_threads_) {
        worker_thread.join();
      }
    }
  }

 private:
  ChunkReader chunk_reader_;

  // chunk sampler to shuffle different chunks
  ChunkSamplerType chunk_sampler_;

  // example sampler to shuffle examples in a specific chunk
  ExampleSamplerType example_sampler_;

  // chunk data buffer which holds chunk data from preloading thread.
  std::shared_ptr<BatchDataBuffer<UnwrappedBatchType, ExampleSampler>>
      chunk_buffer_;

  // worker thread pool
  std::vector<std::thread> preload_threads_;

  // worker thread count
  size_t preloader_count_ = 0;

  size_t batch_size_ = 0;

  // if it is set to true, the dataset will quietly move to the next chunk when
  // the current one is empty. Otherwise, an exception is thrown on the empty
  // batch.
  bool ignore_empty_chunk_ = false;

  std::atomic<bool> quit_worker_{false};

  size_t cache_size_;
};
} // namespace datasets
} // namespace data
} // namespace torch