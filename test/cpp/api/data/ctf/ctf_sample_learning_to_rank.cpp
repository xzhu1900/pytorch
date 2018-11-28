#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValues inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_LEARNING_TO_RANK_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "rating", "rating", 1, torch::data::ctf::CTFValueFormat::Dense);
  stream_defs["labels"].emplace_back(
      "features", "features", 12, torch::data::ctf::CTFValueFormat::Dense);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR +
          "/ctf_sample_learning_to_rank.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Int16));

  torch::data::ctf::CTFParser ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset dataset(torch::data::ctf::CTFDataType::Double);
  {
    // 0
    torch::data::ctf::CTFSequenceID seq_id = 0;
    dataset.features[seq_id].sequence_id = seq_id;
    dataset.labels[seq_id].sequence_id = seq_id;
    {{// |rating 4
      torch::data::ctf::CTFSample sample(seq_id, "rating");
    sample.values.push_back(torch::data::ctf::CTFValue(4));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |features 23 35 0 0 0 21 2345 0 0 0 0 0
    torch::data::ctf::CTFSample sample(seq_id, "features");
    sample.values.push_back(torch::data::ctf::CTFValue(23));
    sample.values.push_back(torch::data::ctf::CTFValue(35));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(21));
    sample.values.push_back(torch::data::ctf::CTFValue(2345));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
{{// |rating 2
  torch::data::ctf::CTFSample sample(seq_id, "rating");
sample.values.push_back(torch::data::ctf::CTFValue(2));
dataset.features[seq_id].samples.push_back(sample);
}

{
  // |features 0 123 0 22 44 44 290 22 22 22 33 0
  torch::data::ctf::CTFSample sample(seq_id, "features");
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(123));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(22));
  sample.values.push_back(torch::data::ctf::CTFValue(44));
  sample.values.push_back(torch::data::ctf::CTFValue(44));
  sample.values.push_back(torch::data::ctf::CTFValue(290));
  sample.values.push_back(torch::data::ctf::CTFValue(22));
  sample.values.push_back(torch::data::ctf::CTFValue(22));
  sample.values.push_back(torch::data::ctf::CTFValue(22));
  sample.values.push_back(torch::data::ctf::CTFValue(33));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  dataset.labels[seq_id].samples.push_back(sample);
}
}
{
  { // |rating 1
    torch::data::ctf::CTFSample sample(seq_id, "rating");
    sample.values.push_back(torch::data::ctf::CTFValue(1));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |features 0 0 0 0 0 0 1 0 0 0 0 0
    torch::data::ctf::CTFSample sample(seq_id, "features");
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(1));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
}

{
  // 1
  torch::data::ctf::CTFSequenceID seq_id = 1;
  dataset.features[seq_id].sequence_id = seq_id;
  dataset.labels[seq_id].sequence_id = seq_id;
  {{// |rating 1
    torch::data::ctf::CTFSample sample(seq_id, "rating");
  sample.values.push_back(torch::data::ctf::CTFValue(1));
  dataset.features[seq_id].samples.push_back(sample);
}

{
  // |features 34 56 0 0 0 45 1312 0 0 0 0 0
  torch::data::ctf::CTFSample sample(seq_id, "features");
  sample.values.push_back(torch::data::ctf::CTFValue(34));
  sample.values.push_back(torch::data::ctf::CTFValue(56));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(45));
  sample.values.push_back(torch::data::ctf::CTFValue(1312));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  sample.values.push_back(torch::data::ctf::CTFValue(0));
  dataset.labels[seq_id].samples.push_back(sample);
}
}

{
  { // |rating 0
    torch::data::ctf::CTFSample sample(seq_id, "rating");
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |features 45 45 0 0 0 12 335 0 0 0 0 0
    torch::data::ctf::CTFSample sample(seq_id, "features");
    sample.values.push_back(torch::data::ctf::CTFValue(45));
    sample.values.push_back(torch::data::ctf::CTFValue(45));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(12));
    sample.values.push_back(torch::data::ctf::CTFValue(335));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    sample.values.push_back(torch::data::ctf::CTFValue(0));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
}

{
  // 2
  torch::data::ctf::CTFSequenceID seq_id = 2;
  dataset.features[seq_id].sequence_id = seq_id;
  dataset.labels[seq_id].sequence_id = seq_id;
  {
    { // |rating 0
      torch::data::ctf::CTFSample sample(seq_id, "rating");
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      dataset.features[seq_id].samples.push_back(sample);
    }

    {
      // |features 0 0 0 0 0 0 22 0 0 0 0 0
      torch::data::ctf::CTFSample sample(seq_id, "features");
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(22));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      sample.values.push_back(torch::data::ctf::CTFValue(0));
      dataset.labels[seq_id].samples.push_back(sample);
    }
  }
}

EXPECT_TRUE(ctf_parser.get_dataset() == dataset);
}
