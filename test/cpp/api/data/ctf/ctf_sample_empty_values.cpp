#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>
#include <torch/data/ctf/utils.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

namespace torch {
namespace data {
namespace ctf {

TEST(DataTest, CTF_SAMPLE_EMPTY_VALUES_SUCCESS) {
  /// Actual data
  std::vector<CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(
      0, "F0", "F0", 0, CTFInputStreamType::Feature, CTFDataStorage::Sparse);
  input_streams.emplace_back(
      1, "F1", "F1", 1, CTFInputStreamType::Label, CTFDataStorage::Dense);
  input_streams.emplace_back(
      2, "F2", "F2", 1, CTFInputStreamType::Label, CTFDataStorage::Dense);
  CTFConfiguration config(
      std::string(CTF_SAMPLE_DIR + "/ctf_sample_empty_values.ctf"),
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
    input_stream_id = 2;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      // |F0
    }
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
    dataset.sequences.push_back(sequence);
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
    input_stream_id = 2;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int16_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      // |F0 |F1 |F2
    }
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
    dataset.sequences.push_back(sequence);
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}

} // namespace ctf
} // namespace data
} // namespace torch