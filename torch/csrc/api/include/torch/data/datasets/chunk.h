#pragma once

#include <c10/util/Exception.h>
#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/data/samplers.h>
#include <torch/csrc/utils/memory.h>

#include <thread>

namespace torch {
namespace data {
namespace datasets {

template <typename Chunk = std::vector<Example<>>, typename ExampleSampler = samplers::RandomSampler>
struct ChunkData {
  public:
  using ChunkType = Chunk;
    // XXX remember rename variable name
    ChunkData(size_t chk_idx, size_t chk_size, ChunkType data)
        : chunk_index(chk_idx),
        remaining_example_count(chk_size)
        {
        //sampler = std::make_shared<samplers::ThreadSafeSampler<ChunkSamplerType>>(sampler);
        chunk_data = std::move(data);

        //std::unique_ptr<ExampleSampler> sampler = torch::make_unique<ExampleSampler>(data.size()); // in the real dataset, we need to get a copy of the

        sampler = torch::make_unique<ExampleSampler>(chunk_data.size());

    }
    ChunkData(size_t chk_idx, std::exception_ptr exception)
        : chunk_index(chk_idx), exception_(exception){
    }
    size_t chunk_index;
    size_t remaining_example_count;
    std::unique_ptr<ExampleSampler> sampler;
    ChunkType chunk_data;
    std::exception_ptr exception_;
};

/// Main class that handles the chunk data.
template <typename Batch = std::vector<Example<>>, typename ExampleSampler = samplers::RandomSampler>
class ChunkDataBuffer {
public:

    using BatchType = Batch;
    using BatchRequestType = typename ExampleSampler::BatchRequestType;
    ChunkDataBuffer(size_t num_chunks)
        : remaining_chunk_count_(num_chunks) {}

    /// Multi-reader multi writer buffer.
    BatchType get_batch(size_t batch_size) {
        BatchType res;
        size_t count = 0;

        while (count < batch_size) {
            std::unique_lock<std::mutex> lock(mutex_);
            cvr_.wait(lock, [this] { // readers wait till these two conditions.
                return (
                    this->total_example_count_in_queue_ > 0 ||
                    remaining_chunk_count_ == 0);
            });
            if (remaining_chunk_count_ == 0) {
                lock.unlock();
                // cvw_.notify_all();
                return res; // unless a reset is done, data read is already completed.
            }
            while (count < batch_size && chunk_queue_.size() > 0) {
                size_t local_count = 0;
                auto& chk_data = chunk_queue_.front();
                if (chk_data.remaining_example_count > 0) {
                  auto example_count = std::min(batch_size - count, chk_data.remaining_example_count);
                  auto batch_example_indices =
                      chk_data.sampler->next(example_count);
                  AT_ASSERT(
                      batch_example_indices &&
                      batch_example_indices.value().size() ==
                          example_count)
                  BatchRequestType indices = batch_example_indices.value();
                  for (size_t i : indices) {
                    res.emplace_back(chk_data.chunk_data[i]);
                    count++;
                    local_count++;
                  }
                  chk_data.remaining_example_count -= local_count;
                  total_example_count_in_queue_ -= local_count;
                }
                assert(chk_data.remaining_example_count >= 0);
                if (chk_data.remaining_example_count == 0) {
                    chunk_queue_.pop();
                    remaining_chunk_count_--;
                }
            }
            lock.unlock();
            cvw_.notify_all(); // notify all writers.
        }
        return res;
    }

    /// Preload threads call this method to add data.
    void add_chunk_data(size_t index, BatchType data) {
        std::unique_lock<std::mutex> lock(mutex_);
        cvw_.wait(lock, [this] { // writers wait for this condition.
            return this->total_example_count_in_queue_ < queue_depth_s;
        });

                         // sampler

        ChunkData<BatchType, ExampleSampler> chk_data(index, data.size(), data);
        chunk_queue_.push(std::move(chk_data));
        total_example_count_in_queue_ += data.size();
        lock.unlock();
        cvr_.notify_all(); // notify all readers.
    }

    /// Preload threads call this method to add data.
    void add_chunk_data(size_t index, std::exception_ptr e_ptr) {
        std::unique_lock<std::mutex> lock(mutex_);
        cvw_.wait(lock, [this] { // writers wait for this condition.
            return this->total_example_count_in_queue_ < queue_depth_s;
        });
        ChunkData<BatchType, ExampleSampler> chk_data(index, e_ptr);
        chunk_queue_.push(std::move(chk_data));
        lock.unlock();
        cvr_.notify_all(); // notify all readers.
    }

    size_t total_example_count_in_queue_{};
    size_t remaining_chunk_count_;
    std::queue<ChunkData<BatchType, ExampleSampler>> chunk_queue_;
    std::mutex mutex_;
    std::condition_variable cvr_;
    std::condition_variable cvw_;

    static const size_t queue_depth_s = 500;
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
    typename BatchRequest = size_t,
    typename ChunkSampler = samplers::RandomSampler,
    typename ExampleSampler = samplers::RandomSampler>
class ChunkDataSet : public BatchDataset<Self, Batch, BatchRequest> {
 public:

  using SelfType = Self;
  using BatchType = Batch;
  using BatchRequestType = BatchRequest;
  using ChunkSamplerType = ChunkSampler;
  using ExampleSamplerType = ExampleSampler;

  ChunkDataSet(size_t preloader_count) : preloader_count_(preloader_count) {
    if (preloader_count_ == 0) {
      throw std::runtime_error(
          "preloader_count is 0. At least one preloader needs to be specified.");
    }
    auto sampler = ChunkSamplerType(0);

    chunk_sampler_ = std::make_shared<samplers::ThreadSafeSampler<ChunkSamplerType>>(sampler);
    //mutex_ = std::make_shared<std::mutex>();
    //preload_threads_ = std::make_shared<std::vector<std::thread>>();
  }

  ChunkDataSet(const ChunkDataSet& data_set)
  {
    chunk_sampler_ = data_set.chunk_sampler_;
    chunk_buffer_ = data_set.chunk_buffer_;
    // for (const std::thread& worker_thread : data_set.preload_threads_) {
    //   preload_threads_.emplace_back(worker_thread);
    // }

    //preload_threads_ = std::move(data_set.preload_threads_);
    //mutex_ = std::move(data_set.mutex_);
    chunks_to_load_ = data_set.chunks_to_load_;
    preloader_count_ = data_set.preloader_count_;
  }

  virtual ~ChunkDataSet() {
    std::cout << "calling destrctor";

    for (auto& worker_thread : preload_threads_) {
      worker_thread.join();
    }
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
  Batch get_batch(BatchRequest indices) override {
//    AT_ASSERT(indices.size() == 1);
    if (chunk_buffer_ == nullptr) {
      throw std::runtime_error(
          "Dataset has not been reset() before calling get_batch().");
    }
    return chunk_buffer_->get_batch(indices);
  }

  /// This will clear any internal state and starts the internal prefetching
  /// mechanism for the chunk dataset. It simply starts a mini dataloader.
  void reset() override {
    chunks_to_load_ = get_chunk_count();
    chunk_sampler_->reset(chunks_to_load_);
    chunk_buffer_ = std::make_shared<ChunkDataBuffer<BatchType, ExampleSamplerType>>(
        chunks_to_load_); // Creates a new chunk buffer each time we reset the
                          // dataset.

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
  void preloader(size_t id) {
    while (true) {
      size_t chunk_id = -1;
      try {
        {
          // std::lock_guard<std::mutex> lock(
          //     *mutex_); // This is simply the mutex for generating chunk
          //              // index. We can wrap the chunk sampler using the
          //              // thread-safe sampler to achieve the same
          //              // effect.
          if (chunks_to_load_ > 0) {
            auto chunk_sampler_result = chunk_sampler_->next(1);
            AT_ASSERT(chunk_sampler_result && chunk_sampler_result.value().size() == 1); // assert the sampler is not exhausted
            chunk_id = chunk_sampler_result.value()[0];
            chunks_to_load_--;
          } else {
            break;
          }
        }
        chunk_buffer_->add_chunk_data(chunk_id, this->read_chunk(chunk_id));

      } catch (...) {
        chunk_buffer_->add_chunk_data(chunk_id, std::current_exception());
      }
    }
    std::cout << "preloader stopping :" << id << std::endl;
  }

  /*void free_thread()
  {

      for (auto& worker_thread : *preload_threads_) {
      worker_thread.join();
    }
  }*/

 private:
  std::shared_ptr<samplers::ThreadSafeSampler<ChunkSampler>> chunk_sampler_;
  std::shared_ptr<ChunkDataBuffer<BatchType, ExampleSampler>> chunk_buffer_;

  // wrap with shared pointer to make it moveble and copyable
  //std::shared_ptr<std::vector<std::thread>> preload_threads_;
  std::vector<std::thread> preload_threads_;
  // wrap the mutex with a unique ptr to make chunkDataSet movable.
  //std::shared_ptr<std::mutex> mutex_;
  size_t chunks_to_load_{};

  size_t preloader_count_;
};
} // namespace datasets
} // namespace data
} // namespace torch