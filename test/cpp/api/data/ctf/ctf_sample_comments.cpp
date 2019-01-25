#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<float>s inside CTFSample are important

namespace torch {
namespace data {
namespace ctf {

TEST(DataTest, CTF_SAMPLE_COMMENTS_SUCCESS) {
  /// Actual data
  std::vector<CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(
      0, "A", "A", 5, CTFInputStreamType::Feature, CTFDataStorage::Dense);
  input_streams.emplace_back(
      1, "B", "B", 0, CTFInputStreamType::Feature, CTFDataStorage::Sparse);
  input_streams.emplace_back(
      2, "C", "C", 1, CTFInputStreamType::Label, CTFDataStorage::Dense);
  CTFConfiguration config(
      std::string(CTF_SAMPLE_DIR + "/ctf_sample_comments.ctf"),
      input_streams,
      CTFDataType(CTFDataType::Float));

  CTFParser<float> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  CTFDataset<float> dataset(CTFDataType::Float, input_streams);
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
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 2;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 1;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |B 100:3 123:4
      sparse_stream_ptr->data.push_back(3);
      sparse_stream_ptr->indices.push_back(100);
      sparse_stream_ptr->data.push_back(4);
      sparse_stream_ptr->indices.push_back(123);
    }
    {
      input_stream_id = 2;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |C 8
      dense_stream_ptr->data.push_back(8);
    }
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |A 0 1 2 3 4
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(1);
      dense_stream_ptr->data.push_back(2);
      dense_stream_ptr->data.push_back(3);
      dense_stream_ptr->data.push_back(4);
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
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 2;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |A 0 1.1 22 0.3 54
      dense_stream_ptr->data.push_back(0);
      dense_stream_ptr->data.push_back(1.1);
      dense_stream_ptr->data.push_back(22);
      dense_stream_ptr->data.push_back(0.3);
      dense_stream_ptr->data.push_back(54);
    }
    {
      input_stream_id = 2;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |C 123917
      dense_stream_ptr->data.push_back(123917);
    }
    {
      input_stream_id = 1;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |B 1134:1.911 13331:0.014
      sparse_stream_ptr->data.push_back(1.911);
      sparse_stream_ptr->indices.push_back(1134);
      sparse_stream_ptr->data.push_back(0.014);
      sparse_stream_ptr->indices.push_back(13331);
    }
    dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
  }

  {
    // 2 (implicit)
#ifdef CTF_DEBUG
    sequence_id = 2;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFSparseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 2;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<float>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 2;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |C -0.001
      dense_stream_ptr->data.push_back(-0.001);
    }
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |A 3.9 1.11 121.2 99.13 0.04
      dense_stream_ptr->data.push_back(3.9);
      dense_stream_ptr->data.push_back(1.11);
      dense_stream_ptr->data.push_back(121.2);
      dense_stream_ptr->data.push_back(99.13);
      dense_stream_ptr->data.push_back(0.04);
    }
    {
      input_stream_id = 1;
      auto sparse_stream_ptr = static_cast<CTFSparseInputStreamData<float>*>(
          sequence[input_stream_id].get());
      // |B 999:0.001 918918:-9.19
      sparse_stream_ptr->data.push_back(0.001);
      sparse_stream_ptr->indices.push_back(999);
      sparse_stream_ptr->data.push_back(-9.19);
      sparse_stream_ptr->indices.push_back(918918);
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