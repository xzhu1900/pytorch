#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_COMMENTS_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "A", "A", 5, torch::data::ctf::CTFValueFormat::Dense);
  stream_defs["features"].emplace_back(
      "B", "B", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["labels"].emplace_back(
      "C", "C", 1, torch::data::ctf::CTFValueFormat::Dense);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_comments.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Double));

  torch::data::ctf::CTFParser<double> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset<double> dataset(
      torch::data::ctf::CTFDataType::Double);
  {
    // 0 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 0;
    torch::data::ctf::CTFExample<double> example(seq_id, stream_defs);

    {
      // |B 100:3 123:4
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("B"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3, 100));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(4, 123));
      example.features.push_back(sample);
    }

    {
      // |C 8
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("C"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(8));
      example.labels.push_back(sample);
    }

    {
      // |A 0 1 2 3 4
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("A"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(0));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(4));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 1 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 1;
    torch::data::ctf::CTFExample<double> example(seq_id, stream_defs);
    {
      // |A 0 1.1 22 0.3 54
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("A"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(0));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1.1));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(22));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(0.3));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(54));
      example.features.push_back(sample);
    }

    {
      // |C 123917
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("C"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(123917));
      example.labels.push_back(sample);
    }

    {
      // |B 1134:1.911 13331:0.014
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("B"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1.911, 1134));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(0.014, 13331));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 2 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 2;
    torch::data::ctf::CTFExample<double> example(seq_id, stream_defs);
    {
      // |C -0.001
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("C"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(-0.001));
      example.labels.push_back(sample);
    }

    {
      // |A 3.9 1.11 121.2 99.13 0.04
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("A"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3.9));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1.11));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(121.2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(99.13));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(0.04));
      example.features.push_back(sample);
    }

    {
      // |B 999:0.001 918918:-9.19
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("B"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(0.001, 999));
      sample.values.push_back(
          torch::data::ctf::CTFValue<double>(-9.19, 918918));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}
