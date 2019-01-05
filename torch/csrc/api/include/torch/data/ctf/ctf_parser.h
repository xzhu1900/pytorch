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
      : filepath_(filepath),
        features_info_(features_info),
        labels_info_(labels_info),
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
typedef std::string CTFComment;
typedef size_t CTFValueIndex;
typedef double CTFValueValue;

template <typename DataType>
struct CTFValue {
  explicit CTFValue() : value(0), index(SIZE_MAX){};
  explicit CTFValue(DataType value, size_t index = SIZE_MAX)
      : value(value), index(index) {}

  DataType value;
  size_t index;
  bool operator==(const CTFValue<DataType>& rhs) const {
    return (this->index == rhs.index && this->value == rhs.value);
  }
};

template <typename DataType>
struct CTFSample {
  explicit CTFSample() {}
  explicit CTFSample(CTFSequenceID sequence_id, std::string input_name)
      : sequence_id(sequence_id), input_name(std::move(input_name)) {}
  explicit CTFSample(
      CTFSequenceID sequence_id,
      std::string input_name,
      std::vector<CTFValue<DataType>> values)
      : sequence_id(sequence_id),
        input_name(std::move(input_name)),
        values(std::move(values)) {}

  bool operator==(const CTFSample& rhs) const {
    return (
        this->input_name == rhs.input_name &&
        this->sequence_id == rhs.sequence_id &&
        std::equal(
            this->values.begin(), this->values.end(), rhs.values.begin()));
  }
  CTFSequenceID sequence_id;
  std::string input_name;
  std::vector<CTFValue<DataType>> values;
};

template <typename DataType>
struct CTFExample {
  CTFExample() = delete;
  CTFExample(size_t features_info_size, size_t labels_info_size)
      : sequence_id(0) {
    features.clear();
    labels.clear();
    features.reserve(features_info_size);
    labels.reserve(labels_info_size);
  }
  CTFExample(
      CTFSequenceID id,
      size_t features_info_size,
      size_t labels_info_size)
      : sequence_id(id) {
    features.clear();
    labels.clear();
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
        element_type_(config.get_ctf_data_type()) {
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
            if (value.index != SIZE_MAX) {
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
            if (value.index != SIZE_MAX) {
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
    CTFExample<DataType> example(
        features_info_.size(), labels_info_.size());
    CTFSequenceID sequence_id;
    CTFSequenceID previous_sequence_id = -1;
    bool has_initial_sequence_id = false;

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
      if (!get_sequence_id(sequence_id)) {
        if (has_initial_sequence_id) {
          sequence_id = previous_sequence_id;
#ifdef CTF_DEBUG
          std::cout << "Using previous Sequence ID (" << previous_sequence_id
                    << ")" << std::endl;
#endif
        } else {
          sequence_id = previous_sequence_id + 1;
#ifdef CTF_DEBUG
          std::cout << "Using incremented Sequence ID (" << previous_sequence_id
                    << ")" << std::endl;
#endif
        }
      } else {
        has_initial_sequence_id = true;
      }

      bool is_new =
          (previous_sequence_id != sequence_id && sequence_id != LONG_MAX);
      if (is_new &&
          (example.features.size() > 0 || example.labels.size() > 0)) {
        dataset_->examples.emplace_back(std::move(example));
        example.features.clear();
        example.labels.clear();
      }
      previous_sequence_id = sequence_id;

      while (!is_eol(reader_->peek_char())) {
        // After the sequence ID, there can be many samples/comments
        CTFComment comment;
        CTFSample<DataType> sample;
        if (!get_sample(sample, sequence_id)) {
          if (!get_comment(comment)) {
            dataset_->examples.clear();
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
        // Appends a new sample to the dataset
        if (!sample.input_name.empty()) {
          std::vector<CTFStreamInformation>::const_iterator it_stream =
              std::find(
                  labels_info_.begin(),
                  labels_info_.end(),
                  CTFStreamInformation(sample.input_name));

          example.sequence_id = sequence_id;
          if (it_stream != labels_info_.end()) {
            example.labels.emplace_back(std::move(sample));
          } else {
            example.features.emplace_back(std::move(sample));
          }
        }
      }
      // Discard EOL
      reader_->get_char();
    } while (reader_->can_read());
    dataset_->examples.emplace_back(std::move(example));
  }

 private:
  CTFParser() = delete;
  DISALLOW_COPY_AND_ASSIGN(CTFParser);

  bool get_sequence_id(CTFSequenceID& sequence_id) {
    size_t initial_pos = reader_->get_position();
    size_t sequence_id_str_size = 0;
    std::vector<char> sequence_id_str(CTF_MAX_SEQUENCE_ID_LENGTH, '\0');

    // Sequence ID must start with a digit
    char c = reader_->peek_char();
    if (!is_digit(c)) {
#ifdef CTF_DEBUG
      std::cout << "Not a Sequence ID at position " << initial_pos << std::endl;
#endif
      return false;
    }

    // Get all consecutive digits
    while (is_digit(reader_->peek_char())) {
      c = reader_->get_char();
      sequence_id_str[sequence_id_str_size++] = c;
    }

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
    sequence_id =
        static_cast<CTFSequenceID>(std::stoull(sequence_id_str.data()));
#ifdef CTF_DEBUG
    std::cout << "Found Sequence ID '" << std::to_string(sequence_id)
              << "' at position " << initial_pos << std::endl;
#endif
    return true;
  }

  bool get_name(CTFName& name) {
    size_t initial_pos = reader_->get_position();
    size_t name_size = 0;
    std::vector<char> name_str(CTF_MAX_NAME_LENGTH, '\0');

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
      name_str[name_size++] = c;
    }

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
    name = std::string(name_str.begin(), name_str.begin() + name_size);
#ifdef CTF_DEBUG
    std::cout << "Found CTF Name '" << name << "' at position " << initial_pos
              << std::endl;
#endif
    return true;
  }

  bool get_value(CTFValue<DataType>& value, CTFValueFormat format) {
    size_t initial_pos = reader_->get_position();

    size_t value_str_size = 0;
    std::vector<char> value_str(CTF_MAX_VALUE_LENGTH, '\0');

    size_t index_str_size = 0;
    std::vector<char> index_str(CTF_MAX_INDEX_LENGTH, '\0');

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
        // Discard colon, grab index value and reset value
        c = reader_->get_char();
        index_str = value_str;
        index_str_size = value_str_size;
        value_str_size = 0;
        if (is_float) {
          std::string error_msg(
              "Unexpected symbol '.' at index of CTF Value at position " +
              std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
          std::cerr << error_msg << std::endl;
#endif
          throw std::runtime_error(error_msg);
        }
      }
      value_str[value_str_size++] = c;
    }

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

    // Convert string and return the integral values
    value.value = static_cast<CTFValueValue>(std::stod(
        std::string(value_str.begin(), value_str.begin() + value_str_size)));
    if (index_str_size > 0) {
      if (format == CTFValueFormat::Dense) {
        std::string error_msg(
            "Unexpected CTF Value format. Dense format was expected but "
            "a sparse one was found at position " +
            std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
        std::cerr << error_msg << std::endl;
#endif
        throw std::runtime_error(error_msg);
      }
      value.index = static_cast<CTFValueIndex>(std::stoull(
          std::string(index_str.begin(), index_str.begin() + index_str_size)));
    } else {
      if (format != CTFValueFormat::Dense) {
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
#ifdef CTF_DEBUG
    if (index_str_size > 0) {
      std::cout << "Found CTF Value '" << value.value << "', CTF Index '"
                << value.index << "' at position " << reader_->get_position()
                << std::endl;
    } else {
      std::cout << "Found CTF Value '" << value.value << "' at position "
                << reader_->get_position() << std::endl;
    }
#endif
    return true;
  }

  bool get_comment(CTFComment& comment, bool discard = true) {
    size_t initial_pos = reader_->get_position();
    size_t quote_count = 0;

    size_t comment_size = 0;
    std::vector<char> comment_str(CTF_MAX_COMMENT_LENGTH, '\0');

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
      comment_str[comment_size++] = c;
    }

    if (!discard) {
      comment =
          std::string(comment_str.begin(), comment_str.begin() + comment_size);
    }

#ifdef CTF_DEBUG
    std::cout << "Skipping CTF Comment ("
              << std::string(
                     comment_str.begin(), comment_str.begin() + comment_size)
              << ") at position " << reader_->get_position() << std::endl;
#endif
    return true;
  }

  bool get_values(
      std::vector<CTFValue<DataType>>& values,
      CTFValueFormat format,
      size_t dimension) {
    // TODO: Reserve something for sparse input?
    if ((dimension > 0) && (format == CTFValueFormat::Dense)) {
       values.reserve(dimension);
     }
    while (!is_name_prefix(reader_->peek_char()) &&
           !is_comment_prefix(reader_->peek_char()) &&
           !is_eol(reader_->peek_char())) {
      CTFValue<DataType> value;
      if (!get_value(value, format)) {
#ifdef CTF_DEBUG
        std::cout << "CTF Value not found. An empty one will be used."
                  << std::endl;
#endif
      }
      values.emplace_back(std::move(value));
    }

    return true;
  }

  bool get_sample(
      CTFSample<DataType>& sample,
      const CTFSequenceID& sequence_id) {
    CTFName name;

    /// Get input name
    if (!get_name(name)) {
      return false;
    }

    /// Match input name with the ones at 'features' and 'labels' definitions
    bool found = false;
    auto it_stream = std::find(
        labels_info_.begin(),
        labels_info_.end(),
        CTFStreamInformation(name));
    if (it_stream != labels_info_.end()) {
      found = true;
    } else {
      it_stream = std::find(
          features_info_.begin(),
          features_info_.end(),
          CTFStreamInformation(name));
      if (it_stream != features_info_.end()) {
        found = true;
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

    if (!get_values(sample.values, it_stream->format, it_stream->dimension)) {
      sample.values.clear();
      return false;
    }

    if (sample.values.size() != 0 &&
        ((it_stream->format == CTFValueFormat::Dense &&
          it_stream->dimension != sample.values.size()) ||
         (it_stream->format != CTFValueFormat::Dense &&
          it_stream->dimension != 0 &&
          it_stream->dimension < sample.values.size()))) {
      std::string error_msg(
          "Invalid CTF File. Unexpected dimension for input name '" + name +
          "' was found at position " + std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
      std::cout << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }

    sample.sequence_id = std::move(sequence_id);
    sample.input_name = std::move(name);
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

  const static size_t CTF_MAX_NAME_LENGTH = 128;
  const static size_t CTF_MAX_SEQUENCE_ID_LENGTH = 32;
  const static size_t CTF_MAX_COMMENT_LENGTH = 1024;
  const static size_t CTF_MAX_VALUE_LENGTH = 128;
  const static size_t CTF_MAX_INDEX_LENGTH = 128;

}; // namespace ctf

} // namespace ctf
} // namespace data
} // namespace torch

