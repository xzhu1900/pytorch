#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_CLASSIFICATION_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "features", "features", 5, torch::data::ctf::CTFValueFormat::Dense);
  stream_defs["labels"].emplace_back(
      "class", "class", 0, torch::data::ctf::CTFValueFormat::Sparse);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_classification.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Int16));

  torch::data::ctf::CTFParser<double> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset<double> dataset(
      torch::data::ctf::CTFDataType::Double);
  {
    // 0 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 0;
    torch::data::ctf::CTFExample<double> example(seq_id);

    { // |class 23:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("class"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 23));
      example.labels.push_back(sample);
    }
    { // |features 2 3 4 5 6
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("features"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(4));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(5));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(6));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 1 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 1;
    torch::data::ctf::CTFExample<double> example(seq_id);

    { // |class 13:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("class"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 13));
      example.labels.push_back(sample);
    }
    {
      // |features 2 3 4 5 6
      torch::data::ctf::CTFSample<double> sample(
          seq_id, std::string("features"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(0));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}
