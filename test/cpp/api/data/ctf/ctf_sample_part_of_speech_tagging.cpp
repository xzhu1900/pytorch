#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

namespace torch {
namespace data {
namespace ctf {

TEST(DataTest, CTF_SAMPLE_PART_OF_SPEECH_TAGGING_SUCCESS) {
  /// Actual data
  std::vector<CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(
      0,
      "word",
      "word",
      0,
      CTFInputStreamType::Feature,
      CTFDataStorage::Sparse);
  input_streams.emplace_back(
      1, "tag", "tag", 0, CTFInputStreamType::Label, CTFDataStorage::Sparse);
  CTFConfiguration config(
      std::string(CTF_SAMPLE_DIR + "/ctf_sample_part_of_speech_tagging.ctf"),
      input_streams,
      CTFDataType(CTFDataType::Double));

  CTFParser<double> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  CTFDataset<double> dataset(CTFDataType::Double, input_streams);
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
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<double>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<double>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<double>*>(
          sequence[input_stream_id].get());
      // |word 234:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(234);
      // |word 123:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(123);
      // |word 123:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(123);
    }
    {
      input_stream_id = 1;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<double>*>(
          sequence[input_stream_id].get());
      // |tag 12:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(12);
      // |tag 10:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(10);
      // |tag 13:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(13);
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
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<double>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<double>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<double>*>(
          sequence[input_stream_id].get());
      // |word 234:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(234);
      // |word 123:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(123);
    }
    {
      input_stream_id = 1;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<double>*>(
          sequence[input_stream_id].get());
      // |tag 12:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(12);
      // |tag 10:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(10);
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