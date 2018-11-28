#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValues inside CTFSample are important

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

  torch::data::ctf::CTFParser ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset dataset(torch::data::ctf::CTFDataType::Double);
  {
    // 1 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 1;
    dataset.features[seq_id].sequence_id = seq_id;
    dataset.labels[seq_id].sequence_id = seq_id;
    {
      {
        { // |class 23:1
          torch::data::ctf::CTFSample sample(seq_id, "class");
          sample.values.push_back(torch::data::ctf::CTFValue(1, 23));
          dataset.labels[seq_id].samples.push_back(sample);
        }
        // |features 2 3 4 5 6
        torch::data::ctf::CTFSample sample(seq_id, "features");
        sample.values.push_back(torch::data::ctf::CTFValue(2));
        sample.values.push_back(torch::data::ctf::CTFValue(3));
        sample.values.push_back(torch::data::ctf::CTFValue(4));
        sample.values.push_back(torch::data::ctf::CTFValue(5));
        sample.values.push_back(torch::data::ctf::CTFValue(6));
        dataset.features[seq_id].samples.push_back(sample);
      }
    }
  }

  {
    // 2 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 2;
    dataset.features[seq_id].sequence_id = seq_id;
    dataset.labels[seq_id].sequence_id = seq_id;
    {
      { // |class 13:1
        torch::data::ctf::CTFSample sample(seq_id, "class");
        sample.values.push_back(torch::data::ctf::CTFValue(1, 13));
        dataset.labels[seq_id].samples.push_back(sample);
      }
      {
        // |features 2 3 4 5 6
        torch::data::ctf::CTFSample sample(seq_id, "features");
        sample.values.push_back(torch::data::ctf::CTFValue(1));
        sample.values.push_back(torch::data::ctf::CTFValue(2));
        sample.values.push_back(torch::data::ctf::CTFValue(0));
        sample.values.push_back(torch::data::ctf::CTFValue(2));
        sample.values.push_back(torch::data::ctf::CTFValue(3));
        dataset.features[seq_id].samples.push_back(sample);
      }
    }
  }

  EXPECT_TRUE(ctf_parser.get_dataset() == dataset);
}
