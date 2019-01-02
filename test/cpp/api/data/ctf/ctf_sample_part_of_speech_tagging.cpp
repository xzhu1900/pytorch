#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_PART_OF_SPEECH_TAGGING_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "word", "word", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["labels"].emplace_back(
      "tag", "tag", 0, torch::data::ctf::CTFValueFormat::Sparse);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR +
          "/ctf_sample_part_of_speech_tagging.ctf"),
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
    { // |word 234:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("word"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 234));
      example.features.push_back(sample);
    }

    {
      // |tag 12:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("tag"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 12));
      example.labels.push_back(sample);
    }

    { // |word 123:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("word"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 123));
      example.features.push_back(sample);
    }

    {
      // |tag 10:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("tag"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 10));
      example.labels.push_back(sample);
    }

    {
      // |word 123:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("word"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 123));
      example.features.push_back(sample);
    }

    {
      // |tag 13:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("tag"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 13));
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 1
    torch::data::ctf::CTFSequenceID seq_id = 1;
    torch::data::ctf::CTFExample<double> example(seq_id, stream_defs);
    { // |word 234:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("word"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 234));
      example.features.push_back(sample);
    }

    {
      // |tag 12:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("tag"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 12));
      example.labels.push_back(sample);
    }

    {
      // |word 123:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("word"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 123));
      example.features.push_back(sample);
    }

    {
      // |tag 10:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("tag"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 10));
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }
  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}
