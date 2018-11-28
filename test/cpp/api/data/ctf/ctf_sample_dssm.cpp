#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValues inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_DSSM_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "src", "src", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["labels"].emplace_back(
      "tgt", "tgt", 0, torch::data::ctf::CTFValueFormat::Sparse);
  torch::data::ctf::CTFConfigHelper config(
      std::string(torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_dssm.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Double));

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
        // |src 12:1 23:1 345:2 45001:1
        torch::data::ctf::CTFSample sample(seq_id, "src");
        sample.values.push_back(torch::data::ctf::CTFValue(1, 12));
        sample.values.push_back(torch::data::ctf::CTFValue(1, 23));
        sample.values.push_back(torch::data::ctf::CTFValue(2, 345));
        sample.values.push_back(torch::data::ctf::CTFValue(1, 45001));
        dataset.features[seq_id].samples.push_back(sample);
      }

      {
        // |tgt 233:1 766:2 234:1
        torch::data::ctf::CTFSample sample(seq_id, "tgt");
        sample.values.push_back(torch::data::ctf::CTFValue(1, 233));
        sample.values.push_back(torch::data::ctf::CTFValue(2, 766));
        sample.values.push_back(torch::data::ctf::CTFValue(1, 234));
        dataset.labels[seq_id].samples.push_back(sample);
      }
    }
  }

  {
    // 2 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 2;
            dataset.features[seq_id].sequence_id = seq_id;
        dataset.labels[seq_id].sequence_id = seq_id;
    {
      {
        // |src 123:1 56:1 10324:1 18001:3
        torch::data::ctf::CTFSample sample(seq_id, "src");
        sample.values.push_back(torch::data::ctf::CTFValue(1, 123));
        sample.values.push_back(torch::data::ctf::CTFValue(1, 56));
        sample.values.push_back(torch::data::ctf::CTFValue(1, 10324));
        sample.values.push_back(torch::data::ctf::CTFValue(3, 18001));
        dataset.features[seq_id].samples.push_back(sample);
      }

      {
        // |tgt 233:1 2344:2 8889:1 2234:1 253434:1
        torch::data::ctf::CTFSample sample(seq_id, "tgt");
        sample.values.push_back(torch::data::ctf::CTFValue(1, 233));
        sample.values.push_back(torch::data::ctf::CTFValue(2, 2344));
        sample.values.push_back(torch::data::ctf::CTFValue(1, 8889));
        sample.values.push_back(torch::data::ctf::CTFValue(1, 2234));
        sample.values.push_back(torch::data::ctf::CTFValue(1, 253434));
        dataset.labels[seq_id].samples.push_back(sample);
      }
    }
  }

  EXPECT_TRUE(ctf_parser.get_dataset() == dataset);
}
