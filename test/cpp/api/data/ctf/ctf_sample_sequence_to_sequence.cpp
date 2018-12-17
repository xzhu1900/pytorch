#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_SEQUENCE_TO_SEQUENCE_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "sourceWord", "sourceWord", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["labels"].emplace_back(
      "targetWord", "targetWord", 0, torch::data::ctf::CTFValueFormat::Sparse);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR +
          "/ctf_sample_sequence_to_sequence.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Double));

  torch::data::ctf::CTFParser<double> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset<double> dataset(
      torch::data::ctf::CTFDataType::Double);
  {
    // 0
    torch::data::ctf::CTFSequenceID seq_id = 0;
    torch::data::ctf::CTFExample<double> example(seq_id, stream_defs);
    { // |sourceWord 234:1
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("sourceWord"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 234));
      example.features.push_back(sample);
    }

    {
      // |targetWord 344:1
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("targetWord"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 344));
      example.labels.push_back(sample);
    }

    { // |sourceWord 123:1
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("sourceWord"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 123));
      example.features.push_back(sample);
    }

    {
      // |targetWord 456:1
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("targetWord"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 456));
      example.labels.push_back(sample);
    }

    { // |sourceWord 123:1
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("sourceWord"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 123));
      example.features.push_back(sample);
    }

    {
      // |targetWord 2222:1
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("targetWord"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 2222));
      example.labels.push_back(sample);
    }

    { // |sourceWord 11:1
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("sourceWord"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 11));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 1
    torch::data::ctf::CTFSequenceID seq_id = 1;
    torch::data::ctf::CTFExample<double> example(seq_id, stream_defs);
    {
      // |sourceWord 123:1
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("sourceWord"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 123));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}
