#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<int16_t>s inside CTFSample are important

namespace torch {
namespace data {
namespace ctf {

TEST(DataTest, CTF_SAMPLE_LEARNING_TO_RANK_SUCCESS) {
  /// Actual data
  std::vector<CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(
      0,
      "features",
      "features",
      12,
      CTFInputStreamType::Feature,
      CTFDataStorage::Dense);
  input_streams.emplace_back(
      1,
      "rating",
      "rating",
      1,
      CTFInputStreamType::Label,
      CTFDataStorage::Dense);
  CTFConfiguration config(
      std::string(CTF_SAMPLE_DIR + "/ctf_sample_learning_to_rank.ctf"),
      input_streams,
      CTFDataType(CTFDataType::Int16));

  CTFParser<int16_t> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  CTFDataset<int16_t> dataset(CTFDataType::Int16, input_streams);
#ifdef CTF_DEBUG
  size_t sequence_id = 0;
#endif
  size_t input_stream_id = 0;

  {
    // 0
#ifdef CTF_DEBUG
    sequence_id = 0;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int16_t>*>(
          sequence[input_stream_id].get());
      // |features 23 35 0 0 0 21 2345 0 0 0 0 0
      dense_stream_ptr->data.push_back(23);
      dense_stream_ptr->data.push_back(35);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(21);
      dense_stream_ptr->data.push_back(2345);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      // |features 0 123 0 22 44 44 290 22 22 22 33 0
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(123);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(22);
      dense_stream_ptr->data.push_back(44);
      dense_stream_ptr->data.push_back(44);
      dense_stream_ptr->data.push_back(290);
      dense_stream_ptr->data.push_back(22);
      dense_stream_ptr->data.push_back(22);
      dense_stream_ptr->data.push_back(22);
      dense_stream_ptr->data.push_back(33);
      dense_stream_ptr->data.push_back(0);
      // |features 0 0 0 0 0 0 1 0 0 0 0 0
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(1);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
    }

    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int16_t>*>(
          sequence[input_stream_id].get());
      // |rating 4
      dense_stream_ptr->data.push_back(4);
      // |rating 2
      dense_stream_ptr->data.push_back(2);
      // |rating 1
      dense_stream_ptr->data.push_back(1);
    }

    dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
  }

  {
    // 1
#ifdef CTF_DEBUG
    sequence_id = 1;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int16_t>*>(
          sequence[input_stream_id].get());
      // |features 34 56 0 0 0 45 1312 0 0 0 0 0
      dense_stream_ptr->data.push_back(34);
      dense_stream_ptr->data.push_back(56);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(45);
      dense_stream_ptr->data.push_back(1312);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      // |features 45 45 0 0 0 12 335 0 0 0 0 0
      dense_stream_ptr->data.push_back(45);
      dense_stream_ptr->data.push_back(45);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(12);
      dense_stream_ptr->data.push_back(335);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
    }
    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int16_t>*>(
          sequence[input_stream_id].get());
      // |rating 1
      dense_stream_ptr->data.push_back(1);
      // |rating 0
      dense_stream_ptr->data.push_back(0);
    }
    dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
  }

  {
    // 2
#ifdef CTF_DEBUG
    sequence_id = 2;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int16_t>*>(
          sequence[input_stream_id].get());
      // |features 0 0 0 0 0 0 22 0 0 0 0 0
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(22);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(0);
    }
    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int16_t>*>(
          sequence[input_stream_id].get());
      // |rating 0
      dense_stream_ptr->data.push_back(0);
    }
    dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}

} // namespace ctf
} // namespace data
} // namespace torch