#pragma once

#include <torch/data/ctf/reader.h>
#include <torch/data/ctf/reader_constants.h>
#include <torch/data/ctf/utils.h>

#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#include <memory>
#include <ostream>
#include <vector>

namespace torch {
namespace data {
namespace ctf {

// #define CTF_DEBUG
/*
 * CTF general format
 * [Sequence_Id](Sample or Comment)+
 *   where
 *          sequence_Id=(empty|[0-9]+)
 *          Sample=|Input_Name (Value )*
 *          Comment=|# some content
 * Example:
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

///
/// Enumeration type denoting data type of symbolic data entities or actual
/// data.
///
enum class CTFDataType : unsigned int {
  Unknown = 0,
  Float = 1,
  Double = 2,
  UChar = 3, // So far only used internally in deserializers.
  Float16 = 4,
  Int8 = 5,
  Int16 = 6,
};

///
/// Enumeration type denoting the format of storage
///
enum class CTFValueFormat { Dense, Sparse };

struct CTFStreamInformation {
  std::string name; // Unique name of the stream
  std::string alias; // sample name prefix used in the input data
  size_t dimension; // expected number of elements in a sample
  CTFValueFormat format; // Storage format of the stream

  CTFStreamInformation(std::string name) : name(name){};
  CTFStreamInformation(
      std::string name,
      std::string alias,
      size_t dimension,
      CTFValueFormat format)
      : name(name), alias(alias), dimension(dimension), format(format){};
};
inline bool operator==(
    const CTFStreamInformation& lhs,
    const CTFStreamInformation& rhs) {
  return lhs.name == rhs.name;
}

class CTFConfigHelper {
 public:
  explicit CTFConfigHelper(
      std::string filepath,
      std::vector<CTFStreamInformation> features_info,
      std::vector<CTFStreamInformation> labels_info,
      CTFDataType data_type)
      : filepath_(std::move(filepath)),
        features_info_(std::move(features_info)),
        labels_info_(std::move(labels_info)),
        element_type_(data_type){};

  const std::vector<CTFStreamInformation>& get_features_info() const {
    return features_info_;
  }

  const std::vector<CTFStreamInformation>& get_labels_info() const {
    return labels_info_;
  }

  const std::string& get_file_path() const {
    return filepath_;
  }
  CTFDataType get_ctf_data_type() const {
    return element_type_;
  }

 private:
  std::string filepath_;
  std::vector<CTFStreamInformation> features_info_;
  std::vector<CTFStreamInformation> labels_info_;
  CTFDataType element_type_;
};

typedef long int CTFSequenceID;
typedef std::string CTFName;
typedef size_t CTFValueIndex;
const size_t CTFValueIndexUninitialized = SIZE_MAX;

template <typename DataType>
struct CTFValue {
  explicit CTFValue() : value(0), index(CTFValueIndexUninitialized){};
  explicit CTFValue(DataType value, size_t index = CTFValueIndexUninitialized)
      : value(value), index(index) {}

  DataType value;
  size_t index;
  bool operator==(const CTFValue<DataType>& rhs) const {
    return (this->index == rhs.index && this->value == rhs.value);
  }
};

template <typename DataType>
struct CTFSample {
  explicit CTFSample(std::string input_name)
      : input_name(std::move(input_name)) {}

  bool operator==(const CTFSample& rhs) const {
    return (
        this->input_name == rhs.input_name &&
        std::equal(
            this->values.begin(), this->values.end(), rhs.values.begin()));
  }
  std::string input_name;
  std::vector<CTFValue<DataType>> values;
};

template <typename DataType>
struct CTFExample {
  CTFExample(CTFSequenceID id) : sequence_id(id) {}
  CTFExample(
      CTFSequenceID id,
      size_t features_info_size,
      size_t labels_info_size)
      : sequence_id(id) {
    features.reserve(features_info_size);
    labels.reserve(labels_info_size);
  }

  bool operator==(const CTFExample<DataType>& rhs) const {
    return (
        this->sequence_id == rhs.sequence_id &&
        this->features.size() == rhs.features.size() &&
        this->labels.size() == rhs.labels.size() &&
        std::equal(
            this->features.begin(),
            this->features.end(),
            rhs.features.begin()) &&
        std::equal(
            this->labels.begin(), this->labels.end(), rhs.labels.begin()));

    return true;
  }

  size_t sequence_id;
  std::vector<CTFSample<DataType>> features;
  std::vector<CTFSample<DataType>> labels;
};

template <typename DataType>
struct CTFDataset {
  explicit CTFDataset(CTFDataType type) : type(type) {}
  explicit CTFDataset(CTFDataType type, size_t total_examples) : type(type) {
    examples.reserve(total_examples);
  }

  bool operator==(const CTFDataset<DataType>& rhs) const {
    return (
        this->examples.size() == rhs.examples.size() &&
        std::equal(
            this->examples.begin(),
            this->examples.end(),
            rhs.examples.begin()));

    return true;
  }

  CTFDataType type;
  std::vector<CTFExample<DataType>> examples;
};

template <typename DataType>
std::ostream& operator<<(
    std::ostream& os,
    const CTFValue<DataType>& ctf_value) {
#ifdef CTF_DEBUG
  os << "Value: " << ctf_value.value;
  if (ctf_value.index != CTFValueIndexUninitialized) {
    os << ", Index: " << ctf_value.index;
  }
#else
  if (ctf_value.index != CTFValueIndexUninitialized) {
    os << ctf_value.index << ":";
  }
  os << ctf_value.value << " ";
#endif
  return os;
}

template <typename DataType>
std::ostream& operator<<(
    std::ostream& os,
    const CTFSample<DataType>& ctf_sample) {
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

template <typename DataType>
std::ostream& operator<<(
    std::ostream& os,
    const CTFExample<DataType>& ctf_example) {
#ifdef CTF_DEBUG
  os << "Sequence ID: " << ctf_example.sequence_id << std::endl;
  os << "Features: " << std::endl;
  for (const auto& feature : ctf_example.features) {
    os << feature << std::endl;
  }
  os << "Labels: " << std::endl;
  for (const auto& label : ctf_example.labels) {
    os << label << std::endl;
  }
#else
  os << std::endl << ctf_example.sequence_id << " ";
  for (const auto& feature : ctf_example.features) {
    os << feature << std::endl;
  }
  for (const auto& label : ctf_example.labels) {
    os << label << std::endl;
  }
#endif
  return os;
}

template <typename DataType>
std::ostream& operator<<(
    std::ostream& os,
    const CTFDataset<DataType>& ctf_dataset) {
  for (const auto& example : ctf_dataset.examples) {
    os << example;
  }

  return os;
}

template <typename DataType>
class CTFParser {
 public:
  explicit CTFParser(const CTFConfigHelper& config)
      : features_info_(std::move(config.get_features_info())),
        labels_info_(std::move(config.get_labels_info())),
        element_type_(config.get_ctf_data_type()),
        scratch_(CTF_SCRATCH_LENGTH, '\0'),
        has_initial_sequence_id_(false),
        previous_sequence_id_(-1) {
    reader_ = std::make_shared<Reader>(config.get_file_path());
    dataset_ =
        std::make_shared<CTFDataset<DataType>>(config.get_ctf_data_type());

    if ((features_info_.size() <= 0) || (labels_info_.size() <= 0)) {
      std::string error_msg(
          "Missing 'features' or 'labels' CTF stream definitions!");
#ifdef CTF_DEBUG
      std::cerr << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }
  }

  void print_data(void) const {
    std::cout << "Examples: " << dataset_->examples.size() << std::endl;
    for (const auto& example : dataset_->examples) {
      std::cout << example.sequence_id << " ";
      for (const auto& sample : example.features) {
        std::cout << " |" << sample.input_name << "(F)";
        if (sample.values.empty()) {
          std::cout << " <null optional value>";
        } else {
          for (const auto& value : sample.values) {
            if (value.index != CTFValueIndexUninitialized) {
              std::cout << value.index << ":";
            }
            std::cout << " " << value.value;
          }
        }
      }

      for (const auto& sample : example.labels) {
        std::cout << " |" << sample.input_name << "(L)";
        if (sample.values.empty()) {
          std::cout << " <null optional value>";
        } else {
          for (const auto& value : sample.values) {
            if (value.index != CTFValueIndexUninitialized) {
              std::cout << value.index << ":";
            }
            std::cout << " " << value.value;
          }
        }
      }
      std::cout << std::endl;
    }
  }

  std::shared_ptr<CTFDataset<DataType>> get_dataset() {
    return dataset_;
  }

  void read_from_file() {
#ifdef CTF_DEBUG
    size_t read_count = 0;
#endif

    do {
#ifdef CTF_DEBUG
      std::cout << "Read count: " << ++read_count << " starting at "
                << reader_->get_position() << std::endl;
#endif
      // CTF files start with valid alpha-numeric characters
      if (is_non_printable(reader_->peek_char())) {
        std::string error_msg(
            "Non printable character anon print CTF file at position " +
            std::to_string(reader_->get_position()) + "(" +
            std::to_string(static_cast<int>(reader_->peek_char())) + ")");
#ifdef CTF_DEBUG
        std::cout << error_msg << std::endl;
#endif
        throw std::runtime_error(error_msg);
      }

      // There can be an explicit sequence ID at the beginning of the line or
      // the last known is used implicitly
      get_sequence_id();

      while (!is_eol(reader_->peek_char())) {
        // After the sequence ID, there can be many samples/comments
        if (!get_sample()) {
          if (!discard_comment()) {
            std::string error_msg(
                "Invalid CTF File. Neither a CTF Value nor a "
                "CTF Comment was found at position " +
                std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
            std::cout << error_msg << std::endl;
#endif
            throw std::runtime_error(error_msg);
          }
        }
      }
      // Discard EOL
      reader_->get_char();
    } while (reader_->can_read());
  }

 private:
  CTFParser() = delete;
  DISALLOW_COPY_AND_ASSIGN(CTFParser);

  bool get_sequence_id(void) {
    // Current sequence just parsed
    CTFSequenceID sequence_id;

#ifdef CTF_DEBUG
    // For logging purposes
    size_t initial_pos = reader_->get_position();
#endif
    // idx will be used to iterate through scratch_ for local string parsing
    size_t idx = 0;

    // Sequence ID must start with a digit
    char c = reader_->peek_char();
    if (!is_digit(c)) {
#ifdef CTF_DEBUG
      std::cout << "Not a Sequence ID at position " << initial_pos << std::endl;
#endif
      if (has_initial_sequence_id_) {
        sequence_id = previous_sequence_id_;
#ifdef CTF_DEBUG
        std::cout << "Using previous Sequence ID (" << previous_sequence_id_
                  << ")" << std::endl;
#endif
      } else {
        sequence_id = previous_sequence_id_ + 1;
        dataset_->examples.emplace_back(
            sequence_id, features_info_.size(), labels_info_.size());
#ifdef CTF_DEBUG
        std::cout << "Incremented previous Sequence ID (" << sequence_id << ")"
                  << std::endl;
#endif
      }
      previous_sequence_id_ = sequence_id;
      return false;
    }

    // Get all consecutive digits
    while (is_digit(reader_->peek_char())) {
      c = reader_->get_char();
      scratch_[idx++] = c;
    }
    scratch_[idx] = '\0';

    // Discard delimiters after the ID
    while (is_value_delimiter(reader_->peek_char())) {
      c = reader_->get_char();
    }

    // After Sequence ID, there must be a '|'
    if (!is_name_prefix(reader_->peek_char())) {
      std::string error_msg(
          "Missing name delimiter for one of the sequences at position " +
          std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
      std::cerr << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }

    // Convert string and return integral value
    sequence_id = static_cast<CTFSequenceID>(std::stoull(scratch_.data()));
#ifdef CTF_DEBUG
    std::cout << "Found Sequence ID '" << std::to_string(sequence_id)
              << "' at position " << initial_pos << std::endl;
#endif

    // Decides whether this is a new example or an existing one
    if (previous_sequence_id_ != sequence_id && sequence_id != LONG_MAX) {
      dataset_->examples.emplace_back(
          sequence_id, features_info_.size(), labels_info_.size());
#ifdef CTF_DEBUG
      std::cout << "Created new example with Sequence ID "
                << std::to_string(sequence_id) << std::endl;
#endif
    }

    previous_sequence_id_ = sequence_id;
    has_initial_sequence_id_ = true;
    return true;
  }

  // Parses input name from buffer and add into a new sample to the dataset_
  // true is returned when the input name belongs to an existing stream
  // is_feature is true if the new sample belongs to a feature stream.
  // is_feature is false if the new sample belongs to a label stream.
  bool get_name(
      bool& is_feature_stream,
      std::vector<CTFStreamInformation>::const_iterator& it_stream) {
#ifdef CTF_DEBUG
    // For logging purposes
    size_t initial_pos = reader_->get_position();
#endif
    // idx will be used to iterate through scratch_ for local string parsing
    size_t idx = 0;

    // CTF Name must start with a '|'
    if (!is_name_prefix(reader_->peek_char())) {
#ifdef CTF_DEBUG
      std::cout << "Not a CTF Name at position " << initial_pos << std::endl;
#endif
      return false;
    }

    // Discard | and get all consecutive digits and alpha characters
    char c = reader_->get_char();
    while (is_digit(reader_->peek_char()) || is_alpha(reader_->peek_char())) {
      c = reader_->get_char();
      scratch_[idx++] = c;
    }
    scratch_[idx] = '\0';

    // Discard delimiters after the CTF Name
    while (is_value_delimiter(reader_->peek_char())) {
      c = reader_->get_char();
    }

    // After CTF Name, there must be a CTF value or another CTF Name
    c = reader_->peek_char();
    if (!is_number(c) && !is_name_prefix(c) && !is_eol(c)) {
#ifdef CTF_DEBUG
      std::cerr << "Unexpected symbol '" << c << "' after CTF Name at position "
                << reader_->get_position() << std::endl;
#endif
      reader_->rewind_char();
      return false;
    }

    // Return the CTF Name
    // TODO: Can be done better?
    CTFName name = CTFName(scratch_.begin(), scratch_.begin() + idx);
#ifdef CTF_DEBUG
    std::cout << "Found CTF Name '" << name << "' at position " << initial_pos
              << std::endl;
#endif

    /// Match input name with the ones at 'features' and 'labels' definitions
    it_stream = std::find(
        labels_info_.begin(), labels_info_.end(), CTFStreamInformation(name));
    if (it_stream != labels_info_.end()) {
      // The new sample belongs to the labels stream
      dataset_->examples.back().labels.emplace_back(std::move(name));
      is_feature_stream = false;
    } else {
      it_stream = std::find(
          features_info_.begin(),
          features_info_.end(),
          CTFStreamInformation(name));
      if (it_stream != features_info_.end()) {
        // The new sample belongs to the features stream
        dataset_->examples.back().features.emplace_back(std::move(name));
        is_feature_stream = true;
      } else {
        std::string error_msg(
            "CTF Stream not found for input name '" + name + "'.");
#ifdef CTF_DEBUG
        std::cerr << error_msg << std::endl;
#endif
        throw std::runtime_error(error_msg);
      }
    }

    return true;
  }

  bool get_value(
      const bool& is_feature_stream,
      const std::vector<CTFStreamInformation>::const_iterator& it_stream) {
#ifdef CTF_DEBUG
    // For logging purposes
    size_t initial_pos = reader_->get_position();
#endif
    // idx will be used to iterate through scratch_ for local string parsing
    size_t idx = 0;

    // Temporary data/index holders
    CTFValueIndex ctf_index = CTFValueIndexUninitialized;
    DataType ctf_value;

    // CTF Value must start with a digit, dot, signal or exponent symbol
    char c = reader_->peek_char();
    if (!is_number(c)) {
#ifdef CTF_DEBUG
      std::cerr << "Unexpected symbol '" << c << "' at position " << initial_pos
                << std::endl;
#endif
      return false;
    }

    // Get all consecutive digits and decimal point, if any
    bool is_float = false;
    size_t sign_count = 0;
    bool has_exponent = false;
    while (is_number(reader_->peek_char()) ||
           is_sparse_value_delimiter(reader_->peek_char())) {
      c = reader_->get_char();
      if (is_exponent(c)) {
        has_exponent = true;
      }
      if (is_sign(c)) {
        if ((sign_count > 1 && !has_exponent) || (sign_count > 2)) {
          std::string error_msg(
              "Invalid CTF Value. CTF value with more than one "
              "positive or negative sign at position " +
              std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
          std::cerr << error_msg << std::endl;
#endif
          throw std::runtime_error(error_msg);
        }
        ++sign_count;
      }
      if (is_decimal_point(c)) {
        if (is_float) {
          std::string error_msg(
              "Invalid CTF Value. CTF value with more than one "
              "decimal point at position " +
              std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
          std::cerr << error_msg << std::endl;
#endif
          throw std::runtime_error(error_msg);
        }
        is_float = true;
      }
      if (is_sparse_value_delimiter(c)) {
        // Validate found ctf value index
        if (is_float) {
          std::string error_msg(
              "Unexpected symbol '.' at index of CTF Value at position " +
              std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
          std::cerr << error_msg << std::endl;
#endif
          throw std::runtime_error(error_msg);
        } else {
          // Discard colon, grab cft index value and reset ctf value string
          c = reader_->get_char();
          ctf_index = static_cast<CTFValueIndex>(std::stoull(
              std::string(scratch_.begin(), scratch_.begin() + idx)));
          idx = 0;
#ifdef CTF_DEBUG
          std::cout << "Found CTF Value Index '" << ctf_index
                    << "' at position " << reader_->get_position() << std::endl;
#endif
        }
      }
      scratch_[idx++] = c;
    }
    scratch_[idx] = '\0';

    // Discard delimiters after the CTF Value
    while (is_value_delimiter(reader_->peek_char())) {
      c = reader_->get_char();
    }

    // After CTF Value, there must be another CTF Value or CTF Comment
    c = reader_->peek_char();
    if (!is_number(c) && !is_comment_prefix(c) && !is_eol(c)) {
      std::string error_msg(
          "Unexpected symbol '" + std::to_string(c) +
          "' after CTF Value at position " +
          std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
      std::cerr << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }

    // Grab CTF value
    if (ctf_index != CTFValueIndexUninitialized) {
      if (it_stream->format == CTFValueFormat::Dense) {
        std::string error_msg(
            "Unexpected CTF Value format. Dense format was expected but "
            "a sparse one was found at position " +
            std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
        std::cerr << error_msg << std::endl;
#endif
        throw std::runtime_error(error_msg);
      }
    } else {
      if (it_stream->format != CTFValueFormat::Dense) {
        std::string error_msg(
            "Unexpected CTF Value format. Sparse format was expected but "
            "a dense one was found at position " +
            std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
        std::cerr << error_msg << std::endl;
#endif
        throw std::runtime_error(error_msg);
      }
    }
    ctf_value = static_cast<DataType>(
        std::stod(std::string(scratch_.begin(), scratch_.begin() + idx)));
#ifdef CTF_DEBUG
    std::cout << "Found CTF Value '" << ctf_value << "' at position "
              << reader_->get_position() << std::endl;
#endif

    if (is_feature_stream) {
      dataset_->examples.back().features.back().values.emplace_back(
          ctf_value, ctf_index);
    } else {
      dataset_->examples.back().labels.back().values.emplace_back(
          ctf_value, ctf_index);
    }
    return true;
  }

  bool discard_comment(void) {
#ifdef CTF_DEBUG
    // For logging purposes
    size_t initial_pos = reader_->get_position();
#endif

    // Used for matching quotes inside a comment
    // Helps detecting end of comment
    size_t quote_count = 0;

    // CTF Comment must start with |#
    char c = reader_->get_char();
    if (!is_comment_prefix(c)) {
#ifdef CTF_DEBUG
      std::cout << "Not a CTF Comment at position " << initial_pos << std::endl;
#endif
      reader_->rewind_char();
      return false;
    }

    c = reader_->get_char();
    if (!is_comment_suffix(c)) {
      std::string error_msg(
          "Not a CTF Comment at position " +
          std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
      std::cout << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }

    // Get all consecutive digits and alpha characters
    while (!is_eol(reader_->peek_char())) {
      c = reader_->peek_char();
      // Comment symbol can show up when properly escaped
      if (is_escape_delimiter(c)) {
        ++quote_count;
      }

      // If new ctf sample is found, end current comment
      if (is_name_prefix(c) && (quote_count % 2 == 0)) {
        break;
      }

      c = reader_->get_char();
    }

#ifdef CTF_DEBUG
    std::cout << "Skipping CTF Comment at position " << reader_->get_position()
              << std::endl;
#endif
    return true;
  }

  bool get_values(
      const bool& is_feature_stream,
      const std::vector<CTFStreamInformation>::const_iterator& it_stream) {
    // Reserve vector capacity for dense values
    // TODO: Reserve something for sparse input? 1%? 0.05% of dimension?
    if ((it_stream->dimension > 0) &&
        (it_stream->format == CTFValueFormat::Dense)) {
      if (is_feature_stream) {
        dataset_->examples.back().features.reserve(it_stream->dimension);
      } else {
        dataset_->examples.back().labels.reserve(it_stream->dimension);
      }
    }

    // Get them all and push to th right stream
    while (!is_name_prefix(reader_->peek_char()) &&
           !is_comment_prefix(reader_->peek_char()) &&
           !is_eol(reader_->peek_char())) {
      if (!get_value(is_feature_stream, it_stream)) {
#ifdef CTF_DEBUG
        std::cout << "CTF Value not found. An empty one will be used."
                  << std::endl;
#endif
      }
    }

    return true;
  }

  bool get_sample(void) {
    // get_name gets info about the new stream and get_values uses it
    bool is_feature_stream;
    std::vector<CTFStreamInformation>::const_iterator it_stream;
    if (!get_name(is_feature_stream, it_stream)) {
      return false;
    }

    if (!get_values(is_feature_stream, it_stream)) {
      return false;
    }

    // Checking actual number of values records of the stream
    if (is_feature_stream &&
        (dataset_->examples.back().features.back().values.size() != 0) &&
        ((it_stream->format == CTFValueFormat::Dense &&
          it_stream->dimension !=
              dataset_->examples.back().features.back().values.size()) ||
         (it_stream->format != CTFValueFormat::Dense &&
          it_stream->dimension != 0 &&
          it_stream->dimension <
              dataset_->examples.back().features.back().values.size()))) {
      std::string error_msg(
          "Invalid CTF File. Unexpected dimension for feature name '" +
          dataset_->examples.back().features.back().input_name +
          "' was found at position " + std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
      std::cout << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }
    if (!is_feature_stream &&
        (dataset_->examples.back().labels.back().values.size() != 0) &&
        ((it_stream->format == CTFValueFormat::Dense &&
          it_stream->dimension !=
              dataset_->examples.back().labels.back().values.size()) ||
         (it_stream->format != CTFValueFormat::Dense &&
          it_stream->dimension != 0 &&
          it_stream->dimension <
              dataset_->examples.back().labels.back().values.size()))) {
      std::string error_msg(
          "Invalid CTF File. Unexpected dimension for label name '" +
          dataset_->examples.back().labels.back().input_name +
          "' was found at position " + std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
      std::cout << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }

    return true;
  }

  // CTF Stream definitions for features and labels
  std::vector<CTFStreamInformation> features_info_;
  std::vector<CTFStreamInformation> labels_info_;
  // type for CTF values
  CTFDataType element_type_;
  // dataset holding all parsed entries
  std::shared_ptr<CTFDataset<DataType>> dataset_;
  // resposible for reading the CTF file
  std::shared_ptr<Reader> reader_;
  // Local buffer for string parsing
  const size_t CTF_SCRATCH_LENGTH = 128;
  std::vector<char> scratch_;
  // Used to decide whether first row of CTF file has a Sequence ID
  bool has_initial_sequence_id_;
  // Used to detect when a sequence is over
  CTFSequenceID previous_sequence_id_;

}; // namespace ctf

} // namespace ctf
} // namespace data
} // namespace torch
