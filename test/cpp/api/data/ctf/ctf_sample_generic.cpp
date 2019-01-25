#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<int32_t>s inside CTFSample are important

namespace torch {
namespace data {
namespace ctf {

TEST(DataTest, CTF_SAMPLE_GENERIC_SUCCESS) {
  /// Actual data
  std::vector<CTFInputStreamInformation> input_streams;
  input_streams.emplace_back(
      0, "a", "a", 3, CTFInputStreamType::Feature, CTFDataStorage::Dense);
  input_streams.emplace_back(
      1, "b", "b", 2, CTFInputStreamType::Label, CTFDataStorage::Dense);
  CTFConfiguration config(
      std::string(CTF_SAMPLE_DIR + "/ctf_sample_generic.ctf"),
      input_streams,
      CTFDataType(CTFDataType::Int32));

  CTFParser<int32_t> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  CTFDataset<int32_t> dataset(CTFDataType::Int32, input_streams);
#ifdef CTF_DEBUG
  size_t sequence_id = 0;
#endif
  size_t input_stream_id = 0;
  {
    // 100
#ifdef CTF_DEBUG
    sequence_id = 100;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());
      // |a 1 2 3
      dense_stream_ptr->data.push_back(1);
      dense_stream_ptr->data.push_back(2);
      dense_stream_ptr->data.push_back(3);
      // a 4 5 6
      dense_stream_ptr->data.push_back(4);
      dense_stream_ptr->data.push_back(5);
      dense_stream_ptr->data.push_back(6);
      // |a 7 8 9
      dense_stream_ptr->data.push_back(7);
      dense_stream_ptr->data.push_back(8);
      dense_stream_ptr->data.push_back(9);
      // |a 7 8 9
      dense_stream_ptr->data.push_back(7);
      dense_stream_ptr->data.push_back(8);
      dense_stream_ptr->data.push_back(9);
    }

    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());
      // |b 100 200
      dense_stream_ptr->data.push_back(100);
      dense_stream_ptr->data.push_back(200);
      // |b 101 201
      dense_stream_ptr->data.push_back(101);
      dense_stream_ptr->data.push_back(201);
      // |b 102983 14532
      dense_stream_ptr->data.push_back(102983);
      dense_stream_ptr->data.push_back(14532);
    }

    dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
  }

  {
    // 200
#ifdef CTF_DEBUG
    sequence_id = 200;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());
      // |b 300 400
      dense_stream_ptr->data.push_back(300);
      dense_stream_ptr->data.push_back(400);
    }
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());
      // |a 10 20 30
      dense_stream_ptr->data.push_back(10);
      dense_stream_ptr->data.push_back(20);
      dense_stream_ptr->data.push_back(30);
    }
    dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
  }

  {
    // 333
#ifdef CTF_DEBUG
    sequence_id = 333;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());

      // |b 500 100
      dense_stream_ptr->data.push_back(500);
      dense_stream_ptr->data.push_back(100);
      // |b 600 -900
      dense_stream_ptr->data.push_back(600);
      dense_stream_ptr->data.push_back(-900);

      dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
      dataset.sequences_id.push_back(sequence_id);
#endif
    }
  }

  {
    // 400
#ifdef CTF_DEBUG
    sequence_id = 400;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());
      // |a 1 2 3
      dense_stream_ptr->data.push_back(1);
      dense_stream_ptr->data.push_back(2);
      dense_stream_ptr->data.push_back(3);
      // |a 4 5 6
      dense_stream_ptr->data.push_back(4);
      dense_stream_ptr->data.push_back(5);
      dense_stream_ptr->data.push_back(6);
      // |a 4 5 6 TODO: repeated lines should be considered invalid
      dense_stream_ptr->data.push_back(4);
      dense_stream_ptr->data.push_back(5);
      dense_stream_ptr->data.push_back(6);
    }

    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());
      // |b 100 200
      dense_stream_ptr->data.push_back(100);
      dense_stream_ptr->data.push_back(200);
      // |b 101 201
      dense_stream_ptr->data.push_back(101);
      dense_stream_ptr->data.push_back(201);
      // |b 101 201 TODO: repeated lines should be considered invalid
      dense_stream_ptr->data.push_back(101);
      dense_stream_ptr->data.push_back(201);
    }
    dataset.sequences.push_back(sequence);
#ifdef CTF_DEBUG
    dataset.sequences_id.push_back(sequence_id);
#endif
  }

  {
    // 500
#ifdef CTF_DEBUG
    sequence_id = 500;
#endif
    CTFSequenceData sequence;
    input_stream_id = 0;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    input_stream_id = 1;
    sequence.emplace_back(std::make_shared<CTFDenseInputStreamData<int32_t>>(
        input_stream_id, input_streams[input_stream_id].dimension));
    {
      input_stream_id = 0;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());
      // |a 1 2 3
      dense_stream_ptr->data.push_back(1);
      dense_stream_ptr->data.push_back(2);
      dense_stream_ptr->data.push_back(3);
    }

    {
      input_stream_id = 1;
      auto dense_stream_ptr = static_cast<CTFDenseInputStreamData<int32_t>*>(
          sequence[input_stream_id].get());
      // |b 100 200
      dense_stream_ptr->data.push_back(100);
      dense_stream_ptr->data.push_back(200);
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