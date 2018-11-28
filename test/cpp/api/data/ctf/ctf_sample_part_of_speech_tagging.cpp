#include <gtest/gtest.h>

#include <test/cpp/api/data/ctf/samples/ctf_samples.h>
#include <torch/data/ctf/ctf_parser.h>

/// Tests must be executed from root directory of the repo
/// Order of CTFValues inside CTFSample are important

TEST(DataTest, CTF_SAMPLE_PART_OF_SPEECH_TAGGING_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "word", "word", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["labels"].emplace_back(
      "tag", "tag", 0, torch::data::ctf::CTFValueFormat::Sparse);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR +
          "/ctf_sample_part_of_speech_tagging.ctf"),
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
    // |tag 12:1
    torch::data::ctf::CTFSample sample(seq_id, "tag");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 12));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
{{// |word 123:1
  torch::data::ctf::CTFSample sample(seq_id, "word");
sample.values.push_back(torch::data::ctf::CTFValue(1, 123));
dataset.features[seq_id].samples.push_back(sample);
}

{
  // |tag 10:1
  torch::data::ctf::CTFSample sample(seq_id, "tag");
  sample.values.push_back(torch::data::ctf::CTFValue(1, 10));
  dataset.labels[seq_id].samples.push_back(sample);
}
}
{
  {
    // |word 123:1
    torch::data::ctf::CTFSample sample(seq_id, "word");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 123));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |tag 13:1
    torch::data::ctf::CTFSample sample(seq_id, "tag");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 13));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
}

{
  // 1
  torch::data::ctf::CTFSequenceID seq_id = 1;
  dataset.features[seq_id].sequence_id = seq_id;
  dataset.labels[seq_id].sequence_id = seq_id;
  {{// |word 234:1
    torch::data::ctf::CTFSample sample(seq_id, "word");
  sample.values.push_back(torch::data::ctf::CTFValue(1, 234));
  dataset.features[seq_id].samples.push_back(sample);
}

{
  // |tag 12:1
  torch::data::ctf::CTFSample sample(seq_id, "tag");
  sample.values.push_back(torch::data::ctf::CTFValue(1, 12));
  dataset.labels[seq_id].samples.push_back(sample);
}
}

{
  {
    // |word 123:1
    torch::data::ctf::CTFSample sample(seq_id, "word");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 123));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |tag 10:1
    torch::data::ctf::CTFSample sample(seq_id, "tag");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 10));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
}
EXPECT_TRUE(ctf_parser.get_dataset() == dataset);
}

TEST(DataTest, CTF_SAMPLE_PART_OF_SPEECH_TAGGING_WITH_SEEK_SUCCESS) {
  /// Actual data
  torch::data::ctf::CTFStreamDefinitions stream_defs;
  stream_defs["features"].emplace_back(
      "word", "word", 0, torch::data::ctf::CTFValueFormat::Sparse);
  stream_defs["labels"].emplace_back(
      "tag", "tag", 0, torch::data::ctf::CTFValueFormat::Sparse);
  torch::data::ctf::CTFConfigHelper config(
      std::string(
          torch::data::ctf::CTF_SAMPLE_DIR +
          "/ctf_sample_part_of_speech_tagging.ctf"),
      stream_defs,
      torch::data::ctf::CTFDataType(torch::data::ctf::CTFDataType::Double));

  torch::data::ctf::CTFParser ctf_parser(config);
  ctf_parser.read_from_file(72);

  /// Expected data
  torch::data::ctf::CTFDataset dataset(torch::data::ctf::CTFDataType::Double);
  {
    // 1
    torch::data::ctf::CTFSequenceID seq_id = 1;
    dataset.features[seq_id].sequence_id = seq_id;
    dataset.labels[seq_id].sequence_id = seq_id;
    {{// |word 234:1
      torch::data::ctf::CTFSample sample(seq_id, "word");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 234));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |tag 12:1
    torch::data::ctf::CTFSample sample(seq_id, "tag");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 12));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}

{
  {
    // |word 123:1
    torch::data::ctf::CTFSample sample(seq_id, "word");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 123));
    dataset.features[seq_id].samples.push_back(sample);
  }

  {
    // |tag 10:1
    torch::data::ctf::CTFSample sample(seq_id, "tag");
    sample.values.push_back(torch::data::ctf::CTFValue(1, 10));
    dataset.labels[seq_id].samples.push_back(sample);
  }
}
}

EXPECT_TRUE(ctf_parser.get_dataset() == dataset);
}
