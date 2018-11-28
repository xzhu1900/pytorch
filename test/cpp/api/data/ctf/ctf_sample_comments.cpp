#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValues inside CTFSample are important

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
      std::string(torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_comments.ctf"),
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
        // |B 100:3 123:4
        torch::data::ctf::CTFSample sample(seq_id, "B");
        sample.values.push_back(torch::data::ctf::CTFValue(3, 100));
        sample.values.push_back(torch::data::ctf::CTFValue(4, 123));
        dataset.features[seq_id].samples.push_back(sample);
      }

      {
        // |C 8
        torch::data::ctf::CTFSample sample(seq_id, "C");
        sample.values.push_back(torch::data::ctf::CTFValue(8));
        dataset.labels[seq_id].samples.push_back(sample);
      }

      // |A 0 1 2 3 4
      torch::data::ctf::CTFSample sample(seq_id, "A");
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(1));
      sample.values.push_back(torch::data::ctf::CTFValue(2));
      sample.values.push_back(torch::data::ctf::CTFValue(3));
      sample.values.push_back(torch::data::ctf::CTFValue(4));
      dataset.features[seq_id].samples.push_back(sample);
    }
  }

  {
    // 2 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 2;
            dataset.features[seq_id].sequence_id = seq_id;
        dataset.labels[seq_id].sequence_id = seq_id;
    {
      // |A 0 1.1 22 0.3 54
      torch::data::ctf::CTFSample sample(seq_id, "A");
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(1.1));
      sample.values.push_back(torch::data::ctf::CTFValue(22));
      sample.values.push_back(torch::data::ctf::CTFValue(0.3));
      sample.values.push_back(torch::data::ctf::CTFValue(54));
      dataset.features[seq_id].samples.push_back(sample);
    }

    {
      // |C 123917
      torch::data::ctf::CTFSample sample(seq_id, "C");
      sample.values.push_back(torch::data::ctf::CTFValue(123917));
      dataset.labels[seq_id].samples.push_back(sample);
    }

    {
      // |B 1134:1.911 13331:0.014
      torch::data::ctf::CTFSample sample(seq_id, "B");
      sample.values.push_back(torch::data::ctf::CTFValue(1.911, 1134));
      sample.values.push_back(torch::data::ctf::CTFValue(0.014, 13331));
      dataset.features[seq_id].samples.push_back(sample);
    }
  }

  {
    // 3 (implicit)
    torch::data::ctf::CTFSequenceID seq_id = 3;
        dataset.features[seq_id].sequence_id = seq_id;
        dataset.labels[seq_id].sequence_id = seq_id;
    {
      // |C -0.001
      torch::data::ctf::CTFSample sample(seq_id, "C");
      sample.values.push_back(torch::data::ctf::CTFValue(-0.001));
      dataset.labels[seq_id].samples.push_back(sample);
    }

    {
      // |A 3.9 1.11 121.2 99.13 0.04
      torch::data::ctf::CTFSample sample(seq_id, "A");
      sample.values.push_back(torch::data::ctf::CTFValue(3.9));
      sample.values.push_back(torch::data::ctf::CTFValue(1.11));
      sample.values.push_back(torch::data::ctf::CTFValue(121.2));
      sample.values.push_back(torch::data::ctf::CTFValue(99.13));
      sample.values.push_back(torch::data::ctf::CTFValue(0.04));
      dataset.features[seq_id].samples.push_back(sample);
    }

    {
      // |B 999:0.001 918918:-9.19
      torch::data::ctf::CTFSample sample(seq_id, "B");
      sample.values.push_back(torch::data::ctf::CTFValue(0.001, 999));
      sample.values.push_back(torch::data::ctf::CTFValue(-9.19, 918918));
      dataset.features[seq_id].samples.push_back(sample);
    }
  }

  EXPECT_TRUE(ctf_parser.get_dataset() == dataset);
}
