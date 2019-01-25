#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

namespace torch {
namespace data {
namespace ctf {

TEST(DataTest, CTF_SAMPLE_DSSN_SUCCESS) {
  /// Actual data
  std::vector<CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(
      0, "src", "src", 0, CTFInputStreamType::Feature, CTFDataStorage::Sparse);
  input_streams.emplace_back(
      1, "tgt", "tgt", 0, CTFInputStreamType::Label, CTFDataStorage::Sparse);
  CTFConfiguration config(
      std::string(CTF_SAMPLE_DIR + "/ctf_sample_dssm.ctf"),
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
      // |src 12:1 23:1 345:2 45001:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(12);
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(23);
      sparse_stream_ptr->data.push_back(2);
      sparse_stream_ptr->indices.push_back(345);
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(45001);
    }
    {
      input_stream_id = 1;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<double>*>(
          sequence[input_stream_id].get());
      // |tgt 233:1 766:2 234:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(233);
      sparse_stream_ptr->data.push_back(2);
      sparse_stream_ptr->indices.push_back(766);
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(234);
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
      // |src 123:1 56:1 10324:1 18001:3
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(123);
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(56);
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(10324);
      sparse_stream_ptr->data.push_back(3);
      sparse_stream_ptr->indices.push_back(18001);
    }
    {
      input_stream_id = 1;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<double>*>(
          sequence[input_stream_id].get());
      // |tgt 233:1 2344:2 8889:1 2234:1 253434:1
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(233);
      sparse_stream_ptr->data.push_back(2);
      sparse_stream_ptr->indices.push_back(2344);
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(8889);
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(2234);
      sparse_stream_ptr->data.push_back(1);
      sparse_stream_ptr->indices.push_back(253434);
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