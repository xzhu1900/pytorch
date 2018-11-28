#include <torch/data/ctf/ctf_parser.h>
#include <torch/data/ctf/reader_constants.h>

#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace torch {
namespace data {
namespace ctf {

/*
 * CTF general format
 * [Sequence_Id](Sample or Comment)+
 *   where
 *          sequence_Id=(empty|[0-9]+)
 *          Sample=|Input_Name (Value )*
 *          Comment=|# some content
 *
 * 100 |a 1 2 3 |b 100 200
 * 100 |a 4 5 6 |b 101 201
 * 100 |b 102983 14532 |a 7 8 9
 * 100 |a 7 8 9
 * 200 |b 300 400 |a 10 20 30
 * 333 |b 500 100
 * 333 |b 600 -900
 * 400 |a 1 2 3 |b 100 200
 * |a 4 5 6 |b 101 201
 * |a 4 5 6 |b 101 201
 * 500 |a 1 2 3 |b 100 200
 */

CTFParser::CTFParser(const CTFConfigHelper& config)
    : stream_defs_(config.get_stream_definitions()),
      element_type_(config.get_ctf_data_type()),
      buffer_pos_(0) {
  buffer_.reserve(CTFParser::BUFFER_SIZE);
  reader_ = std::make_shared<Reader>(config.get_file_path());
  dataset_ = std::make_shared<CTFDataset>(config.get_ctf_data_type());

  if (stream_defs_.find(CTF_STREAM_DEFINITION_FEATURES) == stream_defs_.end() ||
      stream_defs_.find(CTF_STREAM_DEFINITION_LABELS) == stream_defs_.end()) {
    std::string error_msg(
        "Missing 'features' or 'labels' on CTF stream definitions!");
#ifdef CTF_DEBUG
    std::cerr << error_msg << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }
}

CTFParser::~CTFParser() {
  buffer_.clear();
}

bool CTFParser::get_sequence_id(CTFSequenceID& sequence_id) {
  size_t runner = buffer_pos_;

  // Sequence ID must start with a digit
  if (!is_digit(buffer_[runner])) {
#ifdef CTF_DEBUG
    std::cout << "Not a Sequence ID at index " << runner << std::endl;
#endif
    return false;
  }

  // Get all consecutive digits
  while (is_digit(buffer_[runner])) {
    ++runner;
  }
  // Store the final index of the sequence id string
  size_t end_seq_id = runner;

  // Discard delimiters after the ID
  while (is_value_delimiter(buffer_[runner])) {
    ++runner;
  }

  // After Sequence ID, there must be a '|'
  if (!is_name_prefix(buffer_[runner])) {
    std::string error_msg(
        "Missing name delimiter for one of the sequences at index " +
        std::to_string(runner));
#ifdef CTF_DEBUG
    std::cerr << error_msg << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }

  // Convert string, update buffer state and return the integral ID
  sequence_id = static_cast<CTFSequenceID>(std::stoull(buffer_.data()));
#ifdef CTF_DEBUG
  std::cout << "Found Sequence ID '" << sequence_id << "' at index "
            << buffer_pos_ << std::endl;
#endif
  buffer_pos_ = runner;
  return true;
}

bool CTFParser::get_name(CTFName& name) {
  // temporary index for iterating over the string buffer
  size_t runner = buffer_pos_;
  size_t beg_name, end_name;

  // CTF Name must start with a |
  if (!is_name_prefix(buffer_[runner])) {
#ifdef CTF_DEBUG
    std::cout << "Not a CTF Name at index " << runner << std::endl;
#endif
    return false;
  }
  beg_name = ++runner;

  // Get all consecutive digits and alpha characters
  while (is_digit(buffer_[runner]) || is_alpha(buffer_[runner])) {
    ++runner;
  }
  end_name = runner;

  // Discard delimiters after the CTF Name
  while (is_value_delimiter(buffer_[runner])) {
    ++runner;
  }

  // After CTF Name, there must be a CTF value or another CTF Name
  if (!is_number(buffer_[runner]) && !is_name_prefix(buffer_[runner]) &&
      !is_eol(buffer_[runner])) {
#ifdef CTF_DEBUG
    std::cerr << "Unexpected symbol '" << buffer_[runner]
              << "' after CTF Name at index " << runner << std::endl;
#endif
    return false;
  }

  // Return the CTF Name
  name = std::string(buffer_.data() + beg_name, end_name - beg_name);
#ifdef CTF_DEBUG
  std::cout << "Found CTF Name '" << name << "' at index " << buffer_pos_
            << std::endl;
#endif
  buffer_pos_ = runner;
  return true;
}

bool CTFParser::get_value(CTFValue& value, CTFValueFormat format) {
  // temporary index for iterating over the string buffer
  size_t runner = buffer_pos_;

  size_t beg_index = SIZE_MAX, end_index = SIZE_MAX;
  size_t beg_value = runner, end_value = runner;

  // CTF Value must start with a digit or signal
  if (!is_number(buffer_[runner])) {
#ifdef CTF_DEBUG
    std::cerr << "Unexpected symbol '" << buffer_[runner] << "' at index "
              << runner << std::endl;
#endif
    return false;
  }
  beg_value = runner;

  // Get all consecutive digits and decimal point, if any
  // TODO: Should support 1.23e-45 format?
  bool is_float = false;
  size_t sign_count = 0;
  bool has_exponent = false;
  while (is_number(buffer_[runner]) ||
         is_sparse_value_delimiter(buffer_[runner])) {
    if (is_exponent(buffer_[runner])) {
      has_exponent = true;
    }
    if (is_sign(buffer_[runner])) {
      if ((sign_count > 1 && !has_exponent) || (sign_count > 2)) {
        std::string error_msg(
            "Invalid CTF Value. CTF value with more than one "
            "positive or negative sign at index " +
            std::to_string(runner));
#ifdef CTF_DEBUG
        std::cerr << error_msg << std::endl;
#endif
        throw std::runtime_error(error_msg);
      }
      ++sign_count;
    }
    if (is_decimal_point(buffer_[runner])) {
      if (is_float) {
        std::string error_msg(
            "Invalid CTF Value. CTF value with more than one "
            "decimal point at index " +
            std::to_string(runner));
#ifdef CTF_DEBUG
        std::cerr << error_msg << std::endl;
#endif
        return false;
      }
      is_float = true;
    }
    if (is_sparse_value_delimiter(buffer_[runner])) {
      // TODO: Look for decimal point on index? It will be truncated anyway
      beg_index = beg_value;
      end_index = runner;
      beg_value = end_index + 1;
    }
    ++runner;
  }
  end_value = runner;

  // Discard delimiters after the CTF Value
  while (is_value_delimiter(buffer_[runner])) {
    ++runner;
  }

  // After CTF Value, there must be another CTF Value or an optional CTF Comment
  if (!is_number(buffer_[runner]) && !is_comment_prefix(buffer_[runner]) &&
      !is_eol(buffer_[runner])) {
    std::string error_msg(
        "Unexpected symbol '" + std::string(1, buffer_[runner]) +
        "' after CTF Value at index " + std::to_string(runner));
#ifdef CTF_DEBUG
    std::cerr << error_msg << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }

  // Convert string, update buffer state and return the integral ID
  value.value =
      static_cast<CTFValueValue>(std::stod(buffer_.data() + beg_value));
  if (beg_index != SIZE_MAX) {
    if (format == CTFValueFormat::Dense) {
      std::string error_msg(
          "Unexpected CTF Value format. Dense format was expected but "
          "a sparse one was found at index " +
          std::to_string(runner));
#ifdef CTF_DEBUG
      std::cerr << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }
    value.index =
        static_cast<CTFValueIndex>(std::stoull(buffer_.data() + beg_index));
  } else {
    if (format != CTFValueFormat::Dense) {
      std::string error_msg(
          "Unexpected CTF Value format. Sparse format was expected but "
          "a dense one was found at index " +
          std::to_string(runner));
#ifdef CTF_DEBUG
      std::cerr << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }
  }
#ifdef CTF_DEBUG
  std::cout << "Found CTF Value '" << value.value << "', CTF Index '"
            << value.index << "' at index " << buffer_pos_ << std::endl;
#endif
  buffer_pos_ = runner;
  return true;
}

bool CTFParser::get_values(
    std::vector<CTFValue>& values,
    CTFValueFormat format) {
  while (!is_name_prefix(buffer_[buffer_pos_]) &&
         !is_comment_prefix(buffer_[buffer_pos_]) &&
         !is_eol(buffer_[buffer_pos_]) &&
         (buffer_pos_ != reader_->file_size())) {
    CTFValue value;
    if (!get_value(value, format)) {
#ifdef CTF_DEBUG
      std::cout << "CTF Value not found. An empty one will be used."
                << std::endl;
#endif
    }
    values.push_back(std::move(value));
  }

  // Remove EOL
  while (is_eol(buffer_[buffer_pos_])) {
    ++buffer_pos_;
  }

  return true;
}

bool CTFParser::get_sample(
    CTFSample& sample,
    const CTFSequenceID& sequence_id) {
  CTFName name;
  std::vector<CTFValue> values;

  if (!get_name(name)) {
    return false;
  }

  bool found = false;
  std::vector<CTFStreamInformation>::const_iterator it_stream;
  for (const auto& stream_vector : stream_defs_) {
    it_stream = std::find(
        stream_vector.second.begin(),
        stream_vector.second.end(),
        CTFStreamInformation(name));
    if (it_stream != stream_vector.second.end()) {
      found = true;
      break;
    }
  }

  if (!found) {
    std::string error_msg(
        "CTF Value format not found for input name '" + name + "'.");
#ifdef CTF_DEBUG
    std::cerr << error_msg << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }
  if (!get_values(values, it_stream->format)) {
    return false;
  }

  if (values.size() != 0 &&
      ((it_stream->format == CTFValueFormat::Dense &&
        it_stream->dimension != values.size()) ||
       (it_stream->format != CTFValueFormat::Dense &&
        it_stream->dimension != 0 && it_stream->dimension < values.size()))) {
    std::string error_msg(
        "Invalid CTF File. Unexpected dimension for input name '" + name +
        "' was found at index " + std::to_string(buffer_pos_));
#ifdef CTF_DEBUG
    std::cout << error_msg << buffer_pos_ << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }

  sample.sequence_id = std::move(sequence_id);
  sample.input_name = std::move(name);
  sample.values = std::move(values);
  return true;
}

bool CTFParser::get_comment(CTFComment& comment) {
  size_t quote_count = 0;

  // temporary index for iterating over the string buffer
  size_t runner = buffer_pos_;
  size_t beg_comment, end_comment;

  // CTF Comment must start with |#
  if (!is_comment_prefix(buffer_[runner])) {
#ifdef CTF_DEBUG
    std::cout << "Not a CTF Comment at index " << runner << std::endl;
#endif
    return false;
  }
  ++runner;
  if (!is_comment_suffix(buffer_[runner])) {
    std::string error_msg(
        "Not a CTF Comment at index " + std::to_string(runner));
#ifdef CTF_DEBUG
    std::cout << error_msg << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }
  beg_comment = ++runner;
  // Get all consecutive digits and alpha characters
  while (!is_eol(buffer_[runner])) {
    ++runner;

    if (is_escape_delimiter(buffer_[runner])) {
      ++quote_count;
    }

    if (is_name_prefix(buffer_[runner]) && (quote_count % 2 == 0)) {
      break;
    }
  }
  end_comment = runner;

  // Remove EOL
  while (is_eol(buffer_[runner])) {
    ++runner;
  }

#ifdef CTF_COMMENT
  // Return the CTF Comment
  comment =
      std::string(buffer_.data() + beg_comment, end_comment - beg_comment);
#ifdef CTF_DEBUG
  std::cout << "Found CTF Comment '" << comment << "' at index " << buffer_pos_
            << std::endl;
#endif
  buffer_pos_ = runner;
  return true;
#else
#ifdef CTF_DEBUG
  std::cout << "Skipping CTF Comment at index " << buffer_pos_ << std::endl;
#endif
  buffer_pos_ = runner;
  comment = std::string();
  return true;
#endif
}

void CTFParser::read_from_file_(long int offset) {
#ifdef CTF_DEBUG
  size_t read_count = 0;
#endif
  CTFSequenceID sequence_id;
  CTFSequenceID previous_sequence_id = 0;
  bool has_initial_sequence_id = false;
  if (offset >= 0) {
    reader_->seek(offset);
  }
  while (reader_->can_read()) {
    size_t len = reader_->read_line(buffer_, CTFParser::BUFFER_SIZE);
#ifdef CTF_DEBUG
    std::cout << "Read file count: " << ++read_count << " (" << len << " bytes)"
              << std::endl;
#endif
    buffer_pos_ = 0;

    // CTF files start with valid alpha-numeric characters
    if (is_non_printable(buffer_[buffer_pos_])) {
      throw std::runtime_error("Non printable character at CTF file!");
    }

    // There can be an explicit sequence ID at the beginning of the line or the
    // last known is used implicitly
    if (!get_sequence_id(sequence_id)) {
      if (has_initial_sequence_id) {
        sequence_id = previous_sequence_id;
#ifdef CTF_DEBUG
        std::cout << "Using previous Sequence ID (" << previous_sequence_id
                  << ")" << std::endl;
#endif
      } else {
        sequence_id = ++previous_sequence_id;
#ifdef CTF_DEBUG
        std::cout << "Incrementing previous Sequence ID ("
                  << previous_sequence_id << ")" << std::endl;
#endif
      }
    } else {
      has_initial_sequence_id = true;
    }

    bool is_new = (previous_sequence_id == sequence_id);
    previous_sequence_id = sequence_id;

    while (buffer_pos_ < len) {
      // After the sequence ID, there can be many samples/comments
      CTFComment comment;
      CTFSample sample;
      if (!get_sample(sample, sequence_id)) {
        if (!get_comment(comment)) {
          std::string error_msg(
              "Invalid CTF File. Neither a CTF Value nor a "
              "CTF Comment was found at index " +
              std::to_string(buffer_pos_));
#ifdef CTF_DEBUG
          std::cout << error_msg << buffer_pos_ << std::endl;
#endif
          dataset_->features.clear();
          dataset_->labels.clear();
          throw std::runtime_error(error_msg);
        } else {
#ifdef CTF_COMMENT
          // Line starts with a comment
          // Each sequence has a single comment, so the last comment override
          // the previous ones

          dataset_->features[sequence_id].sequence_id = sequence_id;
          dataset_->labels[sequence_id].sequence_id = sequence_id;
          if (!comment.empty()) {
            dataset_->features[sequence_id].comment = std::move(comment);
          }

#endif
        }
      }
      // Appends a new sample to the dataset
      std::vector<CTFStreamInformation>::const_iterator it_stream;
      if (!sample.input_name.empty()) {
        std::vector<CTFStreamInformation>::const_iterator it_stream = std::find(
            stream_defs_[CTF_STREAM_DEFINITION_FEATURES].begin(),
            stream_defs_[CTF_STREAM_DEFINITION_FEATURES].end(),
            CTFStreamInformation(sample.input_name));
        if (it_stream != stream_defs_[CTF_STREAM_DEFINITION_FEATURES].end()) {
                dataset_->features[sequence_id].sequence_id = sequence_id;
          dataset_->features[sequence_id].samples.push_back(std::move(sample));
        } else {
          dataset_->labels[sequence_id].sequence_id = sequence_id;
          dataset_->labels[sequence_id].samples.push_back(std::move(sample));
        }
      }
#ifdef CTF_COMMENT
      // Updates the comment for the sequence. Previous comments are overwritten
      if (!comment.empty()) {
        if (it_stream != stream_defs_[CTF_STREAM_DEFINITION_FEATURES].end()) {
          dataset_->features[sequence_id].comment = std::move(comment);
        } else {
          dataset_->labels[sequence_id].comment = std::move(comment);
        }
      }
#endif
    }
  }
}

void CTFParser::read_from_file(long int offset) {
  read_from_file_(offset);
}

void CTFParser::read_from_file() {
  read_from_file_(-1);
}

const CTFDataset& CTFParser::get_dataset(void) const {
  return *dataset_;
}

/*
 * Overriding operator<< for base classes
 */
std::ostream& operator<<(std::ostream& os, const CTFValue& ctf_value) {
#ifdef CTF_DEBUG
  os << "Value: " << ctf_value.value;
  if (ctf_value.index != SIZE_MAX) {
    os << ", Index: " << ctf_value.index;
  }
#else
  if (ctf_value.index != SIZE_MAX) {
    os << ctf_value.index << ":";
  }
  os << ctf_value.value << " ";
#endif
  return os;
}

std::ostream& operator<<(std::ostream& os, const CTFSample& ctf_sample) {
#ifdef CTF_DEBUG
  os << "Input name: " << ctf_sample.input_name << ", "
     << "Values: " << std::endl;
  if (ctf_sample.values.empty()) {
    os << '\t' << "[ null optional value ]" << std::endl;
  } else {
    for (auto it = ctf_sample.values.begin(); it != ctf_sample.values.end();
         ++it) {
      os << '\t' << "[" << *it << "]" << std::endl;
    }
  }
#else
  os << "|" << ctf_sample.input_name;
  if (ctf_sample.values.empty()) {
    os << '\t' << "[ null optional value ]" << std::endl;
  } else {
    for (auto it = ctf_sample.values.begin(); it != ctf_sample.values.end();
         ++it) {
      os << " " << *it;
    }
  }
#endif
  return os;
}

std::ostream& operator<<(std::ostream& os, const CTFSequence& ctf_sequence) {
#ifdef CTF_DEBUG
  os << "Sequence ID: " << ctf_sequence.sequence_id << std::endl;
#ifdef CTF_COMMENT
  if (!ctf_sequence.comment.empty()) {
    os << "Comment: " << ctf_sequence.comment << std::endl;
  }
#endif
  for (auto it = ctf_sequence.samples.begin(); it != ctf_sequence.samples.end();
       ++it) {
    os << *it;
  }
#else
  os << std::endl << ctf_sequence.sequence_id << " ";
  for (auto it = ctf_sequence.samples.begin(); it != ctf_sequence.samples.end();
       ++it) {
    os << *it;
  }
#ifdef CTF_COMMENT
  if (!ctf_sequence.comment.empty()) {
    os << " |#" << ctf_sequence.comment;
  }
#endif
#endif
  return os;
}

std::ostream& operator<<(std::ostream& os, const CTFDataset& ctf_dataset) {
  std::vector<std::unordered_map<CTFSequenceID, CTFSequence>::const_iterator>
      samples_begin;
  std::vector<std::unordered_map<CTFSequenceID, CTFSequence>::const_iterator>
      samples_end;
  samples_begin.push_back(ctf_dataset.features.begin());
  samples_end.push_back(ctf_dataset.features.end());
  samples_begin.push_back(ctf_dataset.labels.begin());
  samples_end.push_back(ctf_dataset.labels.end());

  for (auto i = 0; i < 2; ++i) {
    for (auto it = samples_begin[i]; it != samples_end[i];
         ++it) {
      os << it->second;
    }
  }

  return os;
}

void CTFParser::print_data() const {
  std::vector<std::unordered_map<CTFSequenceID, CTFSequence>::const_iterator>
      samples_begin;
  std::vector<std::unordered_map<CTFSequenceID, CTFSequence>::const_iterator>
      samples_end;
  samples_begin.push_back(dataset_->features.begin());
  samples_end.push_back(dataset_->features.end());
  samples_begin.push_back(dataset_->labels.begin());
  samples_end.push_back(dataset_->labels.end());

  for (auto i = 0; i < 2; ++i) {
    for (auto it_features = samples_begin[i]; it_features != samples_end[i];
         ++it_features) {
      std::cout << it_features->second.sequence_id;
#ifdef CTF_COMMENT
      if (!it_features->second.comment.empty()) {
        std::cout << " |#" << it_features->second.comment;
      }
#endif
      std::cout << std::endl;
      for (const auto& sample : it_features->second.samples) {
        std::cout << " |" << sample.input_name << " ";
        if (sample.values.empty()) {
          std::cout << "<null optional value>";
        } else {
          for (const auto& value : sample.values) {
            if (value.index != SIZE_MAX) {
              std::cout << value.index << ":";
            }
            std::cout << value.value << " ";
          }
        }
        std::cout << std::endl;
      }
    }
  }
}

bool CTFValue::operator==(const CTFValue& rhs) const {
  return (this->index == rhs.index && this->value == rhs.value);
}

bool CTFSample::operator==(const CTFSample& rhs) const {
  return (
      this->input_name == rhs.input_name &&
      std::equal(this->values.begin(), this->values.end(), rhs.values.begin()));
}

bool CTFSequence::operator==(const CTFSequence& rhs) const {
  return (
      this->sequence_id == rhs.sequence_id &&
      this->samples.size() == rhs.samples.size() &&
      std::equal(
          this->samples.begin(), this->samples.end(), rhs.samples.begin()));
}

bool CTFDataset::operator==(const CTFDataset& rhs) const {
  return (
      this->features.size() == rhs.features.size() &&
      this->labels.size() == rhs.labels.size() &&
      std::equal(
          this->features.begin(), this->features.end(), rhs.features.begin()) &&
      std::equal(this->labels.begin(), this->labels.end(), rhs.labels.begin())

  );

  return true;
}

} // namespace ctf
} // namespace data
} // namespace torch
