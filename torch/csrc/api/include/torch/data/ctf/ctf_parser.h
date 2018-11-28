#pragma once

#include <torch/data/ctf/reader.h>
#include <torch/data/ctf/utils.h>

#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <ostream>
#include <unordered_map>
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
/// Enumeration type denoting the format of storage underlying an instance of a
/// NDArrayView.
///
enum class CTFValueFormat {
  Dense,
  Sparse
};

struct CTFStreamInformation {
  std::string name; // Unique name of the stream
  std::string alias; // sample name prefix used in the input data
  size_t dimension; // expected number of elements in a sample
                    // (can be omitted for sparse input)
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

// There must be 'features' and 'labels' streams to define input ad output
typedef std::unordered_map<std::string, std::vector<CTFStreamInformation>>
    CTFStreamDefinitions;

class CTFConfigHelper {
 public:
  explicit CTFConfigHelper(
      std::string filepath,
      CTFStreamDefinitions stream_defs,
      CTFDataType data_type)
      : filepath_(filepath),
        stream_defs_(stream_defs),
        element_type_(data_type){};

  const CTFStreamDefinitions& get_stream_definitions() const {
    return stream_defs_;
  }
  const std::string& get_file_path() const {
    return filepath_;
  }
  CTFDataType get_ctf_data_type() const {
    return element_type_;
  }

  // TODO: Disable copy and move?
  // DISABLE_COPY_AND_MOVE(CTFConfigHelper);

 private:
  std::string filepath_;
  CTFStreamDefinitions stream_defs_;
  CTFDataType element_type_;
};

static const std::string ctf_value_type_str[] =
    {"Unknown", "Float", "Double", "Float16", "Int8", "Int16"};

static const std::string CTF_STREAM_DEFINITION_FEATURES("features");
static const std::string CTF_STREAM_DEFINITION_LABELS("labels");

typedef size_t CTFSequenceID;
typedef std::string CTFName;
typedef std::string CTFComment;
typedef size_t CTFValueIndex;
typedef double CTFValueValue;

struct CTFValue {
  explicit CTFValue() : value(0), index(SIZE_MAX){};
  explicit CTFValue(double value, size_t index = SIZE_MAX)
      : value(value), index(index) {}

  double value;
  size_t index;
  bool operator==(const CTFValue& rhs) const;

  // TODO: Disable copy and assignment?
  // DISALLOW_COPY_AND_ASSIGN(CTFValue);
};

struct CTFSample {
  explicit CTFSample() {}
  explicit CTFSample(CTFSequenceID sequence_id, std::string input_name)
      : sequence_id(sequence_id), input_name(input_name) {}
  explicit CTFSample(
      CTFSequenceID sequence_id,
      std::string input_name,
      std::vector<CTFValue> values)
      : sequence_id(sequence_id), input_name(input_name), values(values) {}

  CTFSequenceID sequence_id;
  std::string input_name;
  std::vector<CTFValue> values;
  bool operator==(const CTFSample& rhs) const;
};

struct CTFSequence {
  explicit CTFSequence() {}
  explicit CTFSequence(CTFSequenceID sequence_id) : sequence_id(sequence_id) {}
  explicit CTFSequence(
      CTFSequenceID sequence_id,
      std::vector<CTFSample> samples)
      : sequence_id(sequence_id), samples(samples) {}
#ifdef CTF_COMMENT
  explicit CTFSequence(CTFSequenceID sequence_id, CTFComment comment)
      : sequence_id(sequence_id), comment(comment) {}
  explicit CTFSequence(
      CTFSequenceID sequence_id,
      std::vector<CTFSample> samples,
      CTFComment comment)
      : sequence_id(sequence_id), samples(samples), comment(comment) {}
  CTFComment comment;
#endif

  CTFSequenceID sequence_id;
  std::vector<struct CTFSample> samples;
  bool operator==(const CTFSequence& rhs) const;
};

struct CTFDataset {
  explicit CTFDataset(CTFDataType type) : type(type) {}

  CTFDataType type;
  std::unordered_map<CTFSequenceID, CTFSequence> features;
  std::unordered_map<CTFSequenceID, CTFSequence> labels;
  bool operator==(const CTFDataset& rhs) const;
};

std::ostream& operator<<(std::ostream& os, const CTFValue& ctf_value);
std::ostream& operator<<(std::ostream& os, const CTFSample& ctf_sample);
std::ostream& operator<<(std::ostream& os, const CTFSequence& ctf_sequence);
std::ostream& operator<<(std::ostream& os, const CTFDataset& ctf_dataset);

class CTFParser {
 public:
  explicit CTFParser(const CTFConfigHelper& config);
  virtual ~CTFParser();

  void read_from_file(void);
  void read_from_file(long int offset);
  void print_data() const;
  const CTFDataset& get_dataset() const;

 private:
  CTFParser() = delete;
  DISALLOW_COPY_AND_ASSIGN(CTFParser);

  bool get_sequence_id(CTFSequenceID& sequence_id);
  bool get_name(CTFName& name);
  bool get_value(CTFValue& value, CTFValueFormat format);
  bool get_comment(CTFComment& comment);
  bool get_values(std::vector<CTFValue>& values, CTFValueFormat format);
  bool get_sample(CTFSample& sample, const CTFSequenceID& sequence_id);
  void read_from_file_(long int offset);

  CTFStreamDefinitions stream_defs_;
  CTFDataType element_type_;

  // BUFFER_SIZE must be big enough to fit a really long line
  static const size_t BUFFER_SIZE = 2 * 1024 * 1024;
  // buffer for temporarily holding a CTF line during parsing
  std::vector<char> buffer_;
  // parsing position of buffer_
  size_t buffer_pos_;
  // dataset holding all parsed entries
  std::shared_ptr<CTFDataset> dataset_;
  // resposible for reading the CTF file
  std::shared_ptr<Reader> reader_;
};

} // namespace ctf
} // namespace data
} // namespace torch
