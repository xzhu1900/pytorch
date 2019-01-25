#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<int16_t>s inside CTFSample are important

namespace torch {
namespace data {
namespace ctf {

TEST(DataTest, CTF_SAMPLE_CLASSIFICATION_SUCCESS) {
  /// Actual data
  std::vector<CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(
      0,
      "features",
      "features",
      5,
      CTFInputStreamType::Feature,
      CTFDataStorage::Dense);
  input_streams.emplace_back(
      1,
      "class",
      "class",
      0,
      CTFInputStreamType::Label,
      CTFDataStorage::Sparse);
  CTFConfiguration config(
      std::string(CTF_SAMPLE_DIR + "/ctf_sample_classification.ctf"),
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
    // 0 (implicit)
#ifdef CTF_DEBUG
    sequence_id = 0;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 1;
      auto sparse_stream_ptr =
          static_cast<CTFSparseInputStreamData<int16_t>*>(
              sequence[input_stream_id].get());
      // |class 23:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(23);
    }
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int16_t>*>(
          sequence[input_stream_id].get());
      // |features 2 3 4 5 6
      dense_stream_ptr->data.push_back(2);
      dense_stream_ptr->data.push_back(3);
      dense_stream_ptr->data.push_back(4);
      dense_stream_ptr->data.push_back(5);
      dense_stream_ptr->data.push_back(6);
    }
    dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
  }

  {
    // 1 (implicit)
#ifdef CTF_DEBUG
    sequence_id = 1;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));

    {
      input_stream_id = 1;
      auto sparse_stream_ptr =
          static_cast<CTFSparseInputStreamData<int16_t>*>(
              sequence[input_stream_id].get());
      // |class 13:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(13);
    }
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int16_t>*>(
          sequence[input_stream_id].get());
      // |features 1 2 0 2 3
      dense_stream_ptr->data.push_back(1);
      dense_stream_ptr->data.push_back(2);
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(2);
      dense_stream_ptr->data.push_back(3);
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