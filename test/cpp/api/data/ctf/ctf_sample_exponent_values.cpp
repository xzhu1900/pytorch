#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_EXPONENT_VALUE_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "F0", "F0", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["labels"].emplace_back(
      "T0", "T0", 1, torch::data::ctf::CTFValueFormat::Dense);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_exponent_values.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Int16));

  torch::data::ctf::CTFParser<double> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset<double> dataset(
      torch::data::ctf::CTFDataType::Double);
  {
    // 0
    torch::data::ctf::CTFSequenceID seq_id = 0;
    torch::data::ctf::CTFExample<double> example(seq_id);

    { // |F0 0:0.421826 1:1.42167 2:-4.13626e-000123 5:-1.83832 7:-0.000114865
      // 9:-36288.6 11:113.553 13:4.25123e+009 16:-1.78095e-005 18:-0.00162638
      // 19:-1.07109
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("F0"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(0.421826, 0));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1.42167, 1));
      sample.values.push_back(
          torch::data::ctf::CTFValue<double>(-4.13626e-000123, 2));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(-1.83832, 5));
      sample.values.push_back(
          torch::data::ctf::CTFValue<double>(-0.000114865, 7));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(-36288.6, 9));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(113.553, 11));
      sample.values.push_back(
          torch::data::ctf::CTFValue<double>(4.25123e+009, 13));
      sample.values.push_back(
          torch::data::ctf::CTFValue<double>(-1.78095e-005, 16));
      sample.values.push_back(
          torch::data::ctf::CTFValue<double>(-0.00162638, 18));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(-1.07109, 19));
      example.features.push_back(sample);
    }
    { // |T0 1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("T0"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1));
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}
