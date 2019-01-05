#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValue<double>s inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_DSSM_SUCCESS) {
  /// Actual data
  std::vector<torch::data::ctf::CTFStreamInformation> features_info;
  std::vector<torch::data::ctf::CTFStreamInformation> labels_info;
  features_info.emplace_back(
      "src", "src", 0, torch::data::ctf::CTFValueFormat::Sparse);
  labels_info.emplace_back(
      "tgt", "tgt", 0, torch::data::ctf::CTFValueFormat::Sparse);
  torch::data::ctf::CTFConfigHelper config(
      std::string(torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_dssm.ctf"),
      features_info, labels_info,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Double));

  torch::data::ctf::CTFParser<double> ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset<double> dataset(
      torch::data::ctf::CTFDataType::Double);
  {
    // 0 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 0;
    torch::data::ctf::CTFExample<double> example(
        seq_id, features_info.size(), labels_info.size());
    {
      // |src 12:1 23:1 345:2 45001:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("src"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 12));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 23));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2, 345));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 45001));
      example.features.push_back(sample);
    }
    {
      // |tgt 233:1 766:2 234:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("tgt"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 233));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2, 766));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 234));
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }
  {
    // 1 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 1;
    torch::data::ctf::CTFExample<double> example(
        seq_id, features_info.size(), labels_info.size());
    {
      // |src 123:1 56:1 10324:1 18001:3
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("src"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 123));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 56));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 10324));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(3, 18001));
      example.features.push_back(sample);
    }
    {
      // |tgt 233:1 2344:2 8889:1 2234:1 253434:1
      torch::data::ctf::CTFSample<double> sample(seq_id, std::string("tgt"));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 233));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(2, 2344));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 8889));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 2234));
      sample.values.push_back(torch::data::ctf::CTFValue<double>(1, 253434));
      example.labels.push_back(sample);
    }
    dataset.examples.push_back(example);
  }

  EXPECT_TRUE(*ctf_parser.get_dataset() == dataset);
}

