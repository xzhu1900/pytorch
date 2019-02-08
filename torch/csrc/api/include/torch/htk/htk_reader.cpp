#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "htk_reader.h"

#include <chrono>
#include <fstream>

#include <string>

namespace torch {
namespace data {
namespace htk {

using namespace torch::data;
using boost::property_tree::ptree;
using boost::property_tree::read_json;

HTKChunkDataReader::HTKChunkDataReader(
    const string& file_directory,
    const string& file_set_name)
    : file_directory_(file_directory) {
  assert(!file_directory_.empty());

  if (*file_directory_.rbegin() != '/' || *file_directory_.rbegin() != '\\') {
    file_directory_ += '/';
  }

  assert(!file_set_name.empty());
  parse_file_set(file_set_name);
}

void HTKChunkDataReader::parse_file_set(const string& file_set_name) {
  // Read json to get file infomation.
  ptree doc;
  read_json(file_set_name, doc);

  // Read the file extensions that needs to parse.
  for (auto& item : doc.get_child("fileType")) {
    file_extensions_.push_back(std::move(item.second.get_value<std::string>()));
  }

  // Get the chunk file name and example count for each chunk.
  for (auto& item : doc.get_child("fileInfo")) {
    ChunkInfo info;
    info.file_name = item.second.get<std::string>("name");
    info.chunk_size = item.second.get<size_t>("count");
    chunk_info_list_.push_back(std::move(info));
  }
}

std::vector<Utterance> HTKChunkDataReader::read_chunk(size_t chunk_id) {
  std::vector<Utterance> result;

  if (chunk_id >= chunk_info_list_.size()) {
    std::string error_msg =
        "chunk with id = " + std::to_string(chunk_id) + "does not exist.";
    throw std::runtime_error(error_msg);
  }

  for (auto& extension : file_extensions_) {
    std::stringstream ss;
    ss << file_directory_ << "chunk" << chunk_id << "." << extension;
    std::string file_full_name = ss.str();
    size_t example_count_in_chunk = chunk_info_list_[chunk_id].chunk_size;
    result.resize(example_count_in_chunk);

    if (extension == "feature") {
      parse_data<FeatureType>(
          file_full_name, "Feature", example_count_in_chunk, result);
    } else if (extension == "lattice") {
      parse_data<LatticeType>(
          file_full_name, "Lattice", example_count_in_chunk, result);
    } else if (extension == "label") {
      parse_data<LabelType>(
          file_full_name, "Label", example_count_in_chunk, result);
    } else {
      std::string error_msg = "unknown field " + extension;
      throw std::runtime_error(error_msg);
    }
  }

  return result;
}

// Helper method to output feature data to output file. It can be used for data
// validation.
void write_feature(
    size_t seq_len,
    size_t feature_dim,
    std::string output,
    std::vector<Utterance>& chunk) {
  ofstream ss;
  ss.open(output, ios::out | std::ios::ate);
  ss << "feature output\n";
  for (int i = 0; i < chunk.size(); ++i) {
    ss << "[ ";
    for (int j = 0; j < seq_len; ++j) {
      ss << "  [ ";
      for (int w = 0; w < feature_dim; ++w) {
        ss << chunk[i].feature[j * feature_dim + w] << ", ";
      }
      ss << "]\n";
    }
    ss << "]\n";
    std::cout << "actual element for " << i << "'s sequence is "
              << chunk[i].feature.size() << endl;
  }
  ss.close();
}

// Helper method to output label data to output file. It can be used for data
// validation.
void write_label(
    size_t seq_len,
    std::string output,
    std::vector<Utterance>& chunk) {
  ofstream ss;
  ss.open(output, ios::out | std::ios::ate);
  ss << "label output\n";
  for (int i = 0; i < chunk.size(); ++i) {
    ss << "[ ";
    for (int j = 0; j < seq_len; ++j) {
      ss << chunk[i].label[j] << ", ";
    }
    ss << "]\n";
  }
  ss.close();
}

} // namespace htk
} // namespace data
} // namespace torch