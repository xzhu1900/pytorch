#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_EMPTY_VALUES_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "F0", "F0", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["features"].emplace_back(
      "F1", "F1", 1, torch::data::ctf::CTFValueFormat::Dense);
  stream_defs["labels"].emplace_back(
      "F2", "F2", 1, torch::data::ctf::CTFValueFormat::Dense);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_empty_values.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Int16));

  torch::data::ctf::CTFParser<double> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset<double> dataset(
      torch::data::ctf::CTFDataType::Double);
  {
    // 1
    torch::data::ctf::CTFSequenceID seq_id = 1;
    torch::data::ctf::CTFExample<double> example(seq_id, stream_defs);

    { // |F0
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("F0"));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 2
    torch::data::ctf::CTFSequenceID seq_id = 2;
    torch::data::ctf::CTFExample<double> example(seq_id, stream_defs);

    { // |F0
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("F0"));
      example.features.push_back(sample);
    }
    { // |F1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("F1"));
      example.features.push_back(sample);
    }
    { // |F2
      torch::data::ctf::CTFSample<double> sample(seq_id, "F2");
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}
