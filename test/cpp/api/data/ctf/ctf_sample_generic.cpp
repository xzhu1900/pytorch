#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValues inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_GENERIC_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "a", "a", 3, torch::data::ctf::CTFValueFormat::Dense);
  stream_defs["labels"].emplace_back(
      "b", "b", 2, torch::data::ctf::CTFValueFormat::Dense);
  torch::data::ctf::CTFConfigHelper config(
      std::string(torch::data::ctf::CTF_SAMPLE_DIR + "/ctf_sample_generic.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Int16));

  torch::data::ctf::CTFParser ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset dataset(torch::data::ctf::CTFDataType::Double);
  {
    // 100
    torch::data::ctf::CTFSequenceID seq_id = 100;
    dataset.features[seq_id].sequence_id = seq_id;
    dataset.labels[seq_id].sequence_id = seq_id;
    {{// |a 1 2 3
      torch::data::ctf::CTFSample sample(seq_id, "a");
    sample.values.push_back(torch::data::ctf::CTFValue(1));
    sample.values.push_back(torch::data::ctf::CTFValue(2));
    sample.values.push_back(torch::data::ctf::CTFValue(3));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |b 100 200
    torch::data::ctf::CTFSample sample(seq_id, "b");
    sample.values.push_back(torch::data::ctf::CTFValue(100));
    sample.values.push_back(torch::data::ctf::CTFValue(200));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}

{{// |a 4 5 6
  torch::data::ctf::CTFSample sample(seq_id, "a");
sample.values.push_back(torch::data::ctf::CTFValue(4));
sample.values.push_back(torch::data::ctf::CTFValue(5));
sample.values.push_back(torch::data::ctf::CTFValue(6));
dataset.features[seq_id].samples.push_back(sample);
}

{
  // |b 101 201
  torch::data::ctf::CTFSample sample(seq_id, "b");
  sample.values.push_back(torch::data::ctf::CTFValue(101));
  sample.values.push_back(torch::data::ctf::CTFValue(201));
  dataset.labels[seq_id].samples.push_back(sample);
}
}

{{// |b 102983 14532
  torch::data::ctf::CTFSample sample(seq_id, "b");
sample.values.push_back(torch::data::ctf::CTFValue(102983));
sample.values.push_back(torch::data::ctf::CTFValue(14532));
dataset.labels[seq_id].samples.push_back(sample);
}

{
  // |a 7 8 9
  torch::data::ctf::CTFSample sample(seq_id, "a");
  sample.values.push_back(torch::data::ctf::CTFValue(7));
  sample.values.push_back(torch::data::ctf::CTFValue(8));
  sample.values.push_back(torch::data::ctf::CTFValue(9));
  dataset.features[seq_id].samples.push_back(sample);
}
}

{
  {
    // |a 7 8 9
    torch::data::ctf::CTFSample sample(seq_id, "a");
    sample.values.push_back(torch::data::ctf::CTFValue(7));
    sample.values.push_back(torch::data::ctf::CTFValue(8));
    sample.values.push_back(torch::data::ctf::CTFValue(9));
    dataset.features[seq_id].samples.push_back(sample);
  }
}
}

{
  // 200
  torch::data::ctf::CTFSequenceID seq_id = 200;
  dataset.features[seq_id].sequence_id = seq_id;
  dataset.labels[seq_id].sequence_id = seq_id;
  {
    { // |b 300 400
      dataset.features[seq_id].sequence_id = seq_id;
      dataset.labels[seq_id].sequence_id = seq_id;
      torch::data::ctf::CTFSample sample(seq_id, "b");
      sample.values.push_back(torch::data::ctf::CTFValue(300));
      sample.values.push_back(torch::data::ctf::CTFValue(400));
      dataset.labels[seq_id].samples.push_back(sample);
    }
    { // |a 10 20 30
      torch::data::ctf::CTFSample sample(seq_id, "a");
      sample.values.push_back(torch::data::ctf::CTFValue(10));
      sample.values.push_back(torch::data::ctf::CTFValue(20));
      sample.values.push_back(torch::data::ctf::CTFValue(30));
      dataset.features[seq_id].samples.push_back(sample);
    }
  }
}

{
  // 333
  torch::data::ctf::CTFSequenceID seq_id = 333;
  dataset.labels[seq_id].sequence_id = seq_id;
  {{// |b 500 100
    torch::data::ctf::CTFSample sample(seq_id, "b");
  sample.values.push_back(torch::data::ctf::CTFValue(500));
  sample.values.push_back(torch::data::ctf::CTFValue(100));
  dataset.labels[seq_id].samples.push_back(sample);
}
}
{
  {
    // |b 600 -900
    torch::data::ctf::CTFSample sample(seq_id, "b");
    sample.values.push_back(torch::data::ctf::CTFValue(600));
    sample.values.push_back(torch::data::ctf::CTFValue(-900));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
}

{
  // 400
  torch::data::ctf::CTFSequenceID seq_id = 400;
  dataset.features[seq_id].sequence_id = seq_id;
  dataset.labels[seq_id].sequence_id = seq_id;
  {{// |a 1 2 3
    torch::data::ctf::CTFSample sample(seq_id, "a");
  sample.values.push_back(torch::data::ctf::CTFValue(1));
  sample.values.push_back(torch::data::ctf::CTFValue(2));
  sample.values.push_back(torch::data::ctf::CTFValue(3));
  dataset.features[seq_id].samples.push_back(sample);
}

{
  // |b 100 200
  torch::data::ctf::CTFSample sample(seq_id, "b");
  sample.values.push_back(torch::data::ctf::CTFValue(100));
  sample.values.push_back(torch::data::ctf::CTFValue(200));
  dataset.labels[seq_id].samples.push_back(sample);
}
}

{{// |a 4 5 6
  torch::data::ctf::CTFSample sample(seq_id, "a");
sample.values.push_back(torch::data::ctf::CTFValue(4));
sample.values.push_back(torch::data::ctf::CTFValue(5));
sample.values.push_back(torch::data::ctf::CTFValue(6));
dataset.features[seq_id].samples.push_back(sample);
}

{
  // |b 101 201
  torch::data::ctf::CTFSample sample(seq_id, "b");
  sample.values.push_back(torch::data::ctf::CTFValue(101));
  sample.values.push_back(torch::data::ctf::CTFValue(201));
  dataset.labels[seq_id].samples.push_back(sample);
}
}

{
  { // |a 4 5 6 TODO: repeated lines should be considered invalid. Add
    // this validation
    torch::data::ctf::CTFSample sample(seq_id, "a");
    sample.values.push_back(torch::data::ctf::CTFValue(4));
    sample.values.push_back(torch::data::ctf::CTFValue(5));
    sample.values.push_back(torch::data::ctf::CTFValue(6));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |b 101 201
    torch::data::ctf::CTFSample sample(seq_id, "b");
    sample.values.push_back(torch::data::ctf::CTFValue(101));
    sample.values.push_back(torch::data::ctf::CTFValue(201));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
}

{
  // 500
  torch::data::ctf::CTFSequenceID seq_id = 500;
  dataset.features[seq_id].sequence_id = seq_id;
  dataset.labels[seq_id].sequence_id = seq_id;
  {
    { // |a 1 2 3
      torch::data::ctf::CTFSample sample(seq_id, "a");
      sample.values.push_back(torch::data::ctf::CTFValue(1));
      sample.values.push_back(torch::data::ctf::CTFValue(2));
      sample.values.push_back(torch::data::ctf::CTFValue(3));
      dataset.features[seq_id].samples.push_back(sample);
    }

    {
      // |b 100 200
      torch::data::ctf::CTFSample sample(seq_id, "b");
      sample.values.push_back(torch::data::ctf::CTFValue(100));
      sample.values.push_back(torch::data::ctf::CTFValue(200));
      dataset.labels[seq_id].samples.push_back(sample);
    }
  }
}

EXPECT_TRUE(ctf_parser.get_dataset() == dataset);
}
