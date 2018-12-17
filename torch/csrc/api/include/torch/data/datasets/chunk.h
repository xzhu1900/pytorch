#pragma once

#include <c10/util/Exception.h>
#include <torch/csrc/utils/memory.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/data/samplers.h>
#include <torch/data/worker_exception.h>

#include <atomic>
#include <thread>
#include <string>

namespace torch {
namespace data {
namespace datasets {

/// A class that contains a chunk unit. A chunk unit can be a collection of
/// images, utterances, e.t.c. This class is agnostic of how the chunk is
/// composed and what's the chunk boundary. It is an templated class which
/// holds data from a specific chunk with associated sampler for example
/// drawing.
template <
    typename Chunk = std::vector<Example<>>,
    typename ExampleSampler = samplers::RandomSampler>
struct ChunkData {
 public:
  using ChunkType = Chunk;
  using ExampleSamplerType = ExampleSampler;

  ChunkData(
      size_t chunk_index,
      size_t chunk_size,
      ChunkType data,
      ExampleSamplerType example_sampler)
      : chunk_index_(chunk_index),
        remaining_example_count(chunk_size),
        chunk_data(std::move(data)),
        sampler(std::move(example_sampler)) {
    sampler.reset(chunk_size);
  }
  ChunkData(
      size_t chunk_index,
      ExampleSamplerType example_sampler,
      std::exception_ptr e)
      : chunk_index_(chunk_index),
        sampler(std::move(example_sampler)),
        exception(e) {}

  /// the chunk index. We will need this for ordered sequence.
  size_t chunk_index_;

  /// remaining examples in this chunk.
  size_t remaining_example_count;

  /// chunk data to return
  ChunkType chunk_data;

  /// in-chunk shuffle sampler
  ExampleSampler sampler;

  /// exception pointer which captures any abnormal exceptions while loading the
  /// chunk.
  std::exception_ptr exception;
};

/// ChunkDataBuffer manages a queue of ChunkData. It fetches batch data for
/// ChunkDataSet, tracks when new ChunkData is needed, and when all chunk are
/// loaded.
template <
    typename Batch = std::vector<Example<>>,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataBuffer {
 public:
  using BatchType = Batch;
  using BatchRequestType = typename ExampleSampler::BatchRequestType;
  ChunkDataBuffer(size_t num_chunks, ExampleSampler example_sampler, bool ignore_empty_chunk, size_t cache_size)
      : remaining_chunk_count_(num_chunks),
        example_sampler_(std::move(example_sampler)),
        ignore_empty_chunk_(ignore_empty_chunk),
        queue_depth_(cache_size) {}

  /// Return batch data from the queue. Called from the ChunkDataSet main
  /// thread.
  BatchType get_batch(size_t batch_size) {
    // lock the tread to make sure only one thread to fetch batch data at a time.
    std::lock_guard<std::mutex> batch_lock(batch_mutex_);

    BatchType result;
    size_t count = 0;

    while (count < batch_size) {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      cvr_.wait(lock, [this] {
        // wait till there is available data in the queue or if all chunks are
        // loaded (i.e. the data set is exhausted for this epoch)
        return (
            this->total_example_count_in_queue_ > 0 ||
            remaining_chunk_count_ == 0);
      });

      if (remaining_chunk_count_ == 0) {
        lock.unlock();

        // All chunks are loaded. Return the remaining data if there is any as
        // the last batch, and wait for a reset() to restart.
        return result;
      }

      while (count < batch_size && chunk_queue_.size() > 0) {
        size_t local_count = 0;
        auto& chunk_data = chunk_queue_.front();

        if(chunk_data.exception)
        {
          throw WorkerException(chunk_data.exception);
        }

        if (chunk_data.remaining_example_count > 0) {
          auto example_count =
              std::min(batch_size - count, chunk_data.remaining_example_count);
          auto batch_example_indices = chunk_data.sampler.next(example_count);
          AT_ASSERT(
              batch_example_indices &&
              batch_example_indices.value().size() == example_count)
          BatchRequestType indices = batch_example_indices.value();
          for (size_t i : indices) {
            result.emplace_back(chunk_data.chunk_data[i]);
            count++;
            local_count++;
          }
          chunk_data.remaining_example_count -= local_count;
          total_example_count_in_queue_ -= local_count;
        }
        AT_ASSERT(chunk_data.remaining_example_count >= 0);
        if (chunk_data.remaining_example_count == 0) {
          chunk_queue_.pop();
          remaining_chunk_count_--;
        }
      }
      lock.unlock();
      cvw_.notify_all(); // notify all writers.
    }
    return result;
  }

  // skip one chunk
  void skip_chunk() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    AT_ASSERT(remaining_chunk_count_ > 0);
    remaining_chunk_count_--;
    lock.unlock();
  }

  /// Push preloaded chunks to chunk queue. Called from the ChunkDataSet worker
  /// threads.
  void add_chunk_data(size_t index, BatchType data) {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    cvw_.wait(lock, [this] {
      // stop loading if we have preloaded enough data.
      return this->total_example_count_in_queue_ < queue_depth_;
    });

    size_t chunk_size = data.size();
    ChunkData<BatchType, ExampleSampler> chunk_data(
        index, chunk_size, std::move(data), example_sampler_);
    chunk_queue_.push(std::move(chunk_data));
    total_example_count_in_queue_ += chunk_size;
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
    ChunkData<BatchType, ExampleSampler> chunk_data(
        index, example_sampler_, e_ptr);
    chunk_queue_.push(std::move(chunk_data));

    // Add a dummy number to the count.
    total_example_count_in_queue_ += 1;
    lock.unlock();
    cvr_.notify_all(); // notify all readers.
  }

  size_t total_example_count_in_queue_ = 0;
  size_t remaining_chunk_count_ = 0;
  std::queue<ChunkData<BatchType, ExampleSampler>> chunk_queue_;

  // sync chunk_queue_ update.
  std::mutex queue_mutex_;

  // sync each get_batch call.
  std::mutex batch_mutex_;
  std::condition_variable cvr_;
  std::condition_variable cvw_;

  ExampleSampler example_sampler_;
  bool ignore_empty_chunk_ = false;

  size_t queue_depth_;
};

/// A stateful dataset that support hierarchical sampling and prefetching of
/// entre chunks. dataset that supports loading an entire chunk of data.
///
/// A chunk could be an entire file, such as an audio data file or an image,
/// or part of a file in the case of a large text file split based on seek
/// positions.
///
/// Unlike regular dataset, chunk dataset require two samplers to operate and
/// keeps an internal state. `ChunkSampler` selects, which chunk to load next,
/// while the `ExampleSampler` determins the order of Examples that are returned
/// in each `get_batch` call. The hierarchical sampling approach used here is
/// inspired by this paper http://martin.zinkevich.org/publications/nips2010.pdf
template <
    typename Self,
    typename Batch = std::vector<Example<>>,
    typename ChunkSampler = samplers::RandomSampler,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataSet : public BatchDataset<Self, Batch, size_t> {
 public:
  using SelfType = Self;
  using BatchType = Batch;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  ChunkDataSet(size_t preloader_count, bool ignore_empty_chunk = false, size_t cache_size = 500)
  : preloader_count_(preloader_count),
    ignore_empty_chunk_(ignore_empty_chunk),
    cache_size_(cache_size) {
    if (preloader_count_ == 0) {
      throw std::runtime_error(
          "preloader_count is 0. At least one preloader needs to be specified.");
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

  /// Read an entire chunk. A derived class needs to override this method.
  /// This is the only API, other than the constructor that
  virtual Batch read_chunk(size_t chunk_index) = 0;

  /// Returns the chunk sampler for this dataset.
  virtual ChunkSampler get_chunk_sampler() = 0;

  /// Returns the example sampler for this dataset.
  virtual ExampleSampler get_example_sampler() = 0;

  /// returns the number of chunks available in this dataset.
  virtual size_t get_chunk_count() = 0;

  /// Default get_batch method of BatchDataSet. This method will handle the
  /// chunk loading and creating of Example batches. The implemenation is
  /// dataset agnostic and does not need overriding in different chunk data
  /// sets.
  Batch get_batch(size_t batch_size) override {
    if (chunk_buffer_ == nullptr) {
      throw std::runtime_error(
          "Dataset has not been reset() before calling get_batch().");
    }
    return chunk_buffer_->get_batch(batch_size);
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset. It simply starts a mini dataloader.
  virtual void reset() {
    // free any created worker from previous reset.
    free_workers();
    preload_threads_.clear();

    size_t chunks_to_load = get_chunk_count();
    chunk_sampler_ =
        std::make_shared<samplers::ThreadSafeSampler<ChunkSamplerType>>(
            get_chunk_sampler());
    chunk_sampler_->reset(chunks_to_load);

    // Creates a new chunk buffer each time we reset the dataset.
    chunk_buffer_ =
        torch::make_unique<ChunkDataBuffer<BatchType, ExampleSamplerType>>(
            chunks_to_load, get_example_sampler(), ignore_empty_chunk_, cache_size_);

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
        auto chunk_sampler_result = chunk_sampler_->next(1);
        if (chunk_sampler_result.has_value()) {
          chunk_id = chunk_sampler_result.value()[0];
        } else {
          break;
        }
        BatchType data = this->read_chunk(chunk_id);
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

        else
        {
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
  // chunk index sampler.
  std::shared_ptr<samplers::ThreadSafeSampler<ChunkSampler>> chunk_sampler_;

  // chunk data buffer which holds chunk data from preloading thread.
  std::shared_ptr<ChunkDataBuffer<BatchType, ExampleSampler>> chunk_buffer_;

  // worker thread pool
  std::vector<std::thread> preload_threads_;

  // worker thread count
  size_t preloader_count_ = 0;

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