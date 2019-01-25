#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

namespace torch {
namespace data {
namespace ctf {

TEST(DataTest, CTF_SAMPLE_EXPONENT_VALUE_SUCCESS) {
  /// Actual data
  std::vector<CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(
      0, "F0", "F0", 0, CTFInputStreamType::Feature, CTFDataStorage::Sparse);
  input_streams.emplace_back(
      1, "T0", "T0", 1, CTFInputStreamType::Label, CTFDataStorage::Dense);
  CTFConfiguration config(
      std::string(CTF_SAMPLE_DIR + "/ctf_sample_exponent_values.ctf"),
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
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<double>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<double>*>(
          sequence[input_stream_id].get());
      // |F0 0:0.421826 1:1.42167 2:-4.13626e-000123 5:-1.83832 7:-0.000114865
      // 9:-36288.6 11:113.553 13:4.25123e+009 16:-1.78095e-005 18:-0.00162638
      // 19:-1.07109
      sparse_stream_ptr->indices.push_back(0);
      sparse_stream_ptr->data.push_back(0.421826);
      sparse_stream_ptr->indices.push_back(1);
      sparse_stream_ptr->data.push_back(1.42167);
      sparse_stream_ptr->indices.push_back(2);
      sparse_stream_ptr->data.push_back(-4.13626e-000123);
      sparse_stream_ptr->indices.push_back(5);
      sparse_stream_ptr->data.push_back(-1.83832);
      sparse_stream_ptr->indices.push_back(7);
      sparse_stream_ptr->data.push_back(-0.000114865);
      sparse_stream_ptr->indices.push_back(9);
      sparse_stream_ptr->data.push_back(-36288.6);
      sparse_stream_ptr->indices.push_back(11);
      sparse_stream_ptr->data.push_back(113.553);
      sparse_stream_ptr->indices.push_back(13);
      sparse_stream_ptr->data.push_back(4.25123e+009);
      sparse_stream_ptr->indices.push_back(16);
      sparse_stream_ptr->data.push_back(-1.78095e-005);
      sparse_stream_ptr->indices.push_back(18);
      sparse_stream_ptr->data.push_back(-0.00162638);
      sparse_stream_ptr->indices.push_back(19);
      sparse_stream_ptr->data.push_back(-1.07109);
    }
    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<double>*>(
          sequence[input_stream_id].get());
      // |T0 1
      dense_stream_ptr->data.push_back(1);
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