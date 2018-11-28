#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValues inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_SEQUENCE_CLASSIFICATION_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "word", "word", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["labels"].emplace_back(
      "class", "class", 0, torch::data::ctf::CTFValueFormat::Sparse);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR +
          "/ctf_sample_sequence_classification.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Double));

  torch::data::ctf::CTFParser ctf_parser(config);
  ctf_parser.read_from_file();

  /// Expected data
  torch::data::ctf::CTFDataset dataset(torch::data::ctf::CTFDataType::Double);
  {
    // 0
    torch::data::ctf::CTFSequenceID seq_id = 0;
    dataset.features[seq_id].sequence_id = seq_id;
    dataset.labels[seq_id].sequence_id = seq_id;
    {{// |word 234:1
      torch::data::ctf::CTFSample sample(seq_id, "word");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 234));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |class 3:1
    torch::data::ctf::CTFSample sample(seq_id, "class");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 3));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
{{// |word 123:1
  torch::data::ctf::CTFSample sample(seq_id, "word");
sample.values.push_back(torch::data::ctf::CTFValue(1, 123));
dataset.features[seq_id].samples.push_back(sample);
}
}
{
  { // |word 890:1
    torch::data::ctf::CTFSample sample(seq_id, "word");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 890));
    dataset.features[seq_id].samples.push_back(sample);
  }
}
}

{
  // 1
  torch::data::ctf::CTFSequenceID seq_id = 1;
  dataset.features[seq_id].sequence_id = seq_id;
  dataset.labels[seq_id].sequence_id = seq_id;
  {{// |word 11:1
    torch::data::ctf::CTFSample sample(seq_id, "word");
  sample.values.push_back(torch::data::ctf::CTFValue(1, 11));
  dataset.features[seq_id].samples.push_back(sample);
}
{
  // |class 2:1
  torch::data::ctf::CTFSample sample(seq_id, "class");
  sample.values.push_back(torch::data::ctf::CTFValue(1, 2));
  dataset.labels[seq_id].samples.push_back(sample);
}
}
{
  { // |word 344:1
    torch::data::ctf::CTFSample sample(seq_id, "word");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 344));
    dataset.features[seq_id].samples.push_back(sample);
  }
}
}

EXPECT_TRUE(ctf_parser.get_dataset() == dataset);
}
