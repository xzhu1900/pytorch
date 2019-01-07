#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_GENERIC_SUCCESS) {
  /// Actual data
  std::vector<torch::data::ctf::CTFStreamInformation> features_info;
  std::vector<torch::data::ctf::CTFStreamInformation> labels_info;
  features_info.emplace_back(
      "a", "a", 3, torch::data::ctf::CTFValueFormat::Dense);
  labels_info.emplace_back(
      "b", "b", 2, torch::data::ctf::CTFValueFormat::Dense);
  torch::data::ctf::CTFConfigHelper config(
      std::string(torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_generic.ctf"),
      features_info,
      labels_info,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Int16));

  torch::data::ctf::CTFParser<double> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset<double> dataset(
      torch::data::ctf::CTFDataType::Double, 5);
  {
    // 100
    torch::data::ctf::CTFSequenceID seq_id = 100;
    torch::data::ctf::CTFExample<double> example(
        seq_id, features_info.size(), labels_info.size());
    { // |a 1 2 3
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3));
      example.features.push_back(sample);
    }

    {
      // |b 100 200
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(100));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(200));
      example.labels.push_back(sample);
    }

    { // |a 4 5 6
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(4));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(5));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(6));
      example.features.push_back(sample);
    }

    {
      // |b 101 201
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(101));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(201));
      example.labels.push_back(sample);
    }

    { // |b 102983 14532
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(102983));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(14532));
      example.labels.push_back(sample);
    }

    {
      // |a 7 8 9
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(7));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(8));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(9));
      example.features.push_back(sample);
    }

    {
      // |a 7 8 9
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(7));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(8));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(9));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 200
    torch::data::ctf::CTFSequenceID seq_id = 200;
    torch::data::ctf::CTFExample<double> example(
        seq_id, features_info.size(), labels_info.size());
    {
      // |b 300 400
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(300));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(400));
      example.labels.push_back(sample);
    }
    { // |a 10 20 30
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(10));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(20));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(30));
      example.features.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 333
    torch::data::ctf::CTFSequenceID seq_id = 333;
    torch::data::ctf::CTFExample<double> example(
        seq_id, features_info.size(), labels_info.size());
    { // |b 500 100
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(500));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(100));
      example.labels.push_back(sample);
    }

    {
      // |b 600 -900
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(600));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(-900));
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 400
    torch::data::ctf::CTFSequenceID seq_id = 400;
    torch::data::ctf::CTFExample<double> example(
        seq_id, features_info.size(), labels_info.size());
    { // |a 1 2 3
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3));
      example.features.push_back(sample);
    }

    {
      // |b 100 200
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(100));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(200));
      example.labels.push_back(sample);
    }

    { // |a 4 5 6
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(4));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(5));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(6));
      example.features.push_back(sample);
    }

    {
      // |b 101 201
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(101));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(201));
      example.labels.push_back(sample);
    }

    { // |a 4 5 6 TODO: repeated lines should be considered invalid. Add
      // this validation
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(4));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(5));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(6));
      example.features.push_back(sample);
    }

    {
      // |b 101 201
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(101));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(201));
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  {
    // 500
    torch::data::ctf::CTFSequenceID seq_id = 500;
    torch::data::ctf::CTFExample<double> example(
        seq_id, features_info.size(), labels_info.size());

    { // |a 1 2 3
      torch::data::ctf::CTFSample<double> sample(std::string("a"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3));
      example.features.push_back(sample);
    }

    {
      // |b 100 200
      torch::data::ctf::CTFSample<double> sample(std::string("b"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(100));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(200));
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}
