#pragma once

#include <torch/data/ctf/reader.h>
#include <torch/data/ctf/utils.h>

#include <stdint.h>
#include <algorithm>
#include <cassert>
#include <climits>
#include <iostream>
#ifdef CTF_DEBUG
#include <map>
#endif
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
/// Beginning of type definitions
///

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
  Int32 = 7,
};

///
/// Enumeration type denoting the format of storage
///
enum class CTFDataStorage { Dense, Sparse };
enum class CTFInputStreamType { Feature, Label };

///
/// Input Stream information
///
struct CTFInputStreamInformation {
  // Self-assigned Unique ID of the input stream (do not assign it!)
  // TODO: ugly, fix this!
  size_t __id__;
  // Unique name of the input stream
  std::string name;
  // Unique alias of the input
  // Useful when the name is long
  std::string alias;
  // expected number of elements in a sample
  // TODO: Only useful if number of samples is known
  size_t dimension;
  // Input streams belong to either Feature or Label
  CTFInputStreamType type;
  // Data storage of the stream
  CTFDataStorage storage;

  CTFInputStreamInformation(
      std::string name,
      std::string alias,
      size_t dimension,
      CTFInputStreamType type,
      CTFDataStorage storage)
      : name(std::move(name)),
        alias(std::move(alias)),
        dimension(dimension),
        type(type),
        storage(storage){};

  // Used for unit tests
  CTFInputStreamInformation(
      size_t id,
      std::string name,
      std::string alias,
      size_t dimension,
      CTFInputStreamType type,
      CTFDataStorage storage)
      : __id__(id),
        name(std::move(name)),
        alias(std::move(alias)),
        dimension(dimension),
        type(type),
        storage(storage){};
};
inline bool operator==(
    const CTFInputStreamInformation& lhs,
    const CTFInputStreamInformation& rhs) {
  return (
      lhs.__id__ == rhs.__id__ && lhs.name == rhs.name &&
      lhs.alias == rhs.alias && lhs.dimension == rhs.dimension &&
      lhs.type == rhs.type && lhs.storage == rhs.storage);
}

inline bool operator!=(
    const CTFInputStreamInformation& lhs,
    const CTFInputStreamInformation& rhs) {
  return !(lhs == rhs);
}

///
/// Helper to centralize all input information in a single object
///
class CTFConfiguration {
 public:
  explicit CTFConfiguration(
      const std::string& filepath,
      const std::vector<CTFInputStreamInformation>& input_streams_info,
      CTFDataType data_type)
      : filepath_(std::move(filepath)),
        input_streams_info_(std::move(input_streams_info)),
        data_type_(data_type){};

  const std::vector<CTFInputStreamInformation>& get_input_streams_info() const {
    return input_streams_info_;
  }

  const std::string& get_file_path() const {
    return filepath_;
  }
  CTFDataType get_ctf_data_type() const {
    return data_type_;
  }

 private:
  std::string filepath_;
  std::vector<CTFInputStreamInformation> input_streams_info_;
  CTFDataType data_type_;
};

///
/// Sequence ID type
/// -1 is used to flag an uninitialized Sequence ID
///
typedef long int CTFSequenceID;

#ifdef CTF_DEBUG
///
/// Maps Sequenced ID to index at vector<CTFSequenceData>
///
typedef std::map<size_t, size_t> CTFSequenceMap;
#endif

///
/// Input Stream ID type
/// All Input Streamsare stored on a vector<CTFINputStreamInformation>
/// and CTFInputStreamID is the index of a particular stream
///
typedef size_t CTFInputStreamID;

///
/// Maps Input Stream names to a unique index
///
typedef std::unordered_map<std::string, CTFInputStreamID>
    CTFInputStreamMapByName;

///
/// Used during sparse data parsing
///
const size_t CTFValueIndexUninitialized = SIZE_MAX;
typedef size_t CTFValueIndex;

///
/// Sequence data type
/// The global vector of sequences and the vector of samples will use it
///
struct CTFInpuStreamDataBase {
  explicit CTFInpuStreamDataBase(size_t input_stream_id)
      : input_stream_id(input_stream_id) {}
  size_t input_stream_id;
};
typedef std::shared_ptr<CTFInpuStreamDataBase> CTFInpuStreamDataBasePtr;
typedef std::vector<CTFInpuStreamDataBasePtr> CTFSequenceData;

///
/// Dense data
///
template <typename DataType>
struct CTFDenseInputStreamData : CTFInpuStreamDataBase {
  explicit CTFDenseInputStreamData(size_t input_stream_id, size_t capacity = 0)
      : CTFInpuStreamDataBase(input_stream_id) {
    if (capacity > 0) {
      // TODO: On a per input stream storage, sample dimension is not useful
      // data.reserve(capacity);
    }
  }

  std::vector<DataType> data;
};

///
/// Sparse data
///
template <typename DataType>
struct CTFSparseInputStreamData : CTFInpuStreamDataBase {
  explicit CTFSparseInputStreamData(
      size_t input_stream_id,
      size_t dimension = 0)
      : CTFInpuStreamDataBase(input_stream_id) {
    if (dimension > 0) {
      /// TODO: Reserve something for sparse input? 1%? 0.05% of dimension?
      // data.reserve(dimension);
    }
  }

  std::vector<size_t> indices;
  std::vector<DataType> data;
};

///
/// CTFDataset centralizes all parsed data
///
template <typename DataType>
struct CTFDataset {
  explicit CTFDataset(
      CTFDataType data_type,
      const std::vector<CTFInputStreamInformation>& input_streams_info)
      : data_type(data_type), input_streams_info(input_streams_info) {}

  bool operator==(const CTFDataset<DataType>& rhs) const {
    // Datasets must have the same type and number of sequences
    if (this->data_type != rhs.data_type ||
        this->sequences.size() != rhs.sequences.size()) {
      return false;
    }

    for (size_t sequence_index = 0; sequence_index < this->sequences.size();
         ++sequence_index) {
      // Each sequence buffer must have the same number of input streams
      if (this->sequences[sequence_index].size() !=
          rhs.sequences[sequence_index].size()) {
        return false;
      }
      // Each input stream must have the same number of values
      for (size_t sequence_data_index = 0;
           sequence_data_index < this->sequences[sequence_index].size();
           ++sequence_data_index) {
        auto this_stream_ptr =
            this->sequences[sequence_index][sequence_data_index].get();
        auto this_stream_id = this_stream_ptr->input_stream_id;
        auto rhs_stream_ptr =
            rhs.sequences[sequence_index][sequence_data_index].get();
        auto rhs_stream_id = rhs_stream_ptr->input_stream_id;
        // Input streams IDs must match
        if (this_stream_id != rhs_stream_id) {
          return false;
        }

        // Input stream metadata must match
        const auto& this_input_stream_info =
            this->input_streams_info[this_stream_id];
        const auto& rhs_input_stream_info =
            rhs.input_streams_info[rhs_stream_id];
        if (this_input_stream_info != rhs_input_stream_info) {
          return false;
        }

        // Values inside each input stream must match
        if (rhs_input_stream_info.storage == CTFDataStorage::Dense) {
          auto this_dense_stream_ptr =
              static_cast<CTFDenseInputStreamData<DataType>*>(this_stream_ptr);
          auto rhs_dense_stream_ptr =
              static_cast<CTFDenseInputStreamData<DataType>*>(rhs_stream_ptr);
          if (this_dense_stream_ptr->data != rhs_dense_stream_ptr->data) {
            return false;
          }
        } else {
          auto this_sparse_stream_ptr =
              static_cast<CTFSparseInputStreamData<DataType>*>(this_stream_ptr);
          auto rhs_sparse_stream_ptr =
              static_cast<CTFSparseInputStreamData<DataType>*>(rhs_stream_ptr);
          if ((this_sparse_stream_ptr->indices !=
               rhs_sparse_stream_ptr->indices) ||
              (this_sparse_stream_ptr->data != rhs_sparse_stream_ptr->data)) {
            return false;
          }
        }
      }
    }

    return true;
  }

  bool operator!=(const CTFDataset<DataType>& rhs) const {
    return !(this == rhs);
  }

  // TODO: Do we need this? Maybe for logging, only
  CTFDataType data_type;
  // Contains all sequences
  // TODO: Performance consideration: CNTK knows the number of sequences in the
  // chunk, allowing accurate memory reservation. Pytorch approach doesn't
  std::vector<CTFSequenceData> sequences;

  // CTF Input Stream definitions for features and labels
  std::vector<CTFInputStreamInformation> input_streams_info;

  // Input stream map (maps input stream name to a unique ID)
  CTFInputStreamMapByName input_streams_map;
#ifdef CTF_DEBUG
  std::vector<size_t> sequences_id;
#endif
};

///
/// Beginning of implementation
///

template <typename DataType>
class CTFParser {
 public:
  explicit CTFParser(const CTFConfiguration& config)
      : data_type_(config.get_ctf_data_type()),
        dataset_(std::make_shared<CTFDataset<DataType>>(
            config.get_ctf_data_type(),
            config.get_input_streams_info())),
        scratch_(CTF_SCRATCH_LENGTH, '\0'),
        reader_(std::make_shared<Reader>(config.get_file_path())),
        has_initial_sequence_id_(false),
        previous_sequence_id_(-1) {
    // TODO: Improve validation by iterating all streams checking
    // CTFInputStreamType?
    if (dataset_->input_streams_info.size() < 2) {
      std::string error_msg(
          "Missing 'features' or 'labels' CTF stream definitions!");
#ifdef CTF_DEBUG
      std::cerr << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }

    // TODO: Improve it for unit testing too
    // Creating unique IDs for all input streams
    for (size_t i = 0; i < dataset_->input_streams_info.size(); ++i) {
      CTFInputStreamInformation& stream = dataset_->input_streams_info[i];
      const std::string& name = stream.name;
      dataset_->input_streams_map[name] = i;
      dataset_->input_streams_info[i].__id__ = i;
    }
  }

#ifdef CTF_DEBUG
  void print_data(void) const {
    size_t index = 0;
    for (const auto& sequence_data : dataset_->sequences) {
      std::cerr << dataset_->sequences_id[index] << " ";
      for (const auto input_stream : sequence_data) {
        auto input_stream_id = input_stream.get()->input_stream_id;
        const auto& input_stream_info =
            dataset_->input_streams_info[input_stream_id];

        std::string input_stream_type;
        if (input_stream_info.type == CTFInputStreamType::Feature) {
          input_stream_type = "F";
        } else {
          input_stream_type = "L";
        }
        std::cerr << " |" << input_stream_info.name << "(" << input_stream_type
                  << ")";

        if (input_stream_info.storage == CTFDataStorage::Dense) {
          CTFDenseInputStreamData<DataType>* dense_data =
              reinterpret_cast<CTFDenseInputStreamData<DataType>*>(
                  input_stream.get());

          if (dense_data->data.empty()) {
            std::cerr << " <empty>";
          } else {
            for (const auto& value : dense_data->data) {
              std::cerr << " " << value;
            }
          }
        } else {
          CTFSparseInputStreamData<DataType>* sparse_data =
              reinterpret_cast<CTFSparseInputStreamData<DataType>*>(
                  input_stream.get());

          if (sparse_data->data.empty()) {
            std::cerr << " <empty>";
          } else {
            size_t col_index = 0;
            for (const auto& value : sparse_data->data) {
              std::cerr << " " << sparse_data->indices[col_index++] << ":"
                        << value;
            }
          }
        }
      }
      std::cerr << std::endl;
      ++index;
    }
  }
#endif

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
      CTFSequenceID sequence_id;
      bool is_new_sequence = get_sequence_id(sequence_id);

      while (!is_eol(reader_->peek_char())) {
        // After the sequence ID, there can be many input streams/comments
        if (!get_input_stream(sequence_id, is_new_sequence)) {
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

  bool get_sequence_id(CTFSequenceID& sequence_id) {
#ifdef CTF_DEBUG
    // For logging purposes
    size_t initial_pos = reader_->get_position();
#endif

    // Flag to identify when a new Sequence ID is found
    bool is_new = false;

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
        is_new = true;
        sequence_id = previous_sequence_id_ + 1;

#ifdef CTF_DEBUG
        std::cout << "Incremented previous Sequence ID (" << sequence_id << ")"
                  << std::endl;
#endif
      }
      previous_sequence_id_ = sequence_id;
      return is_new;
    }

    // Get all consecutive digits
    while (is_digit(reader_->peek_char())) {
      c = reader_->get_char();
      scratch_[idx++] = c;
    }
    scratch_[idx] = '\0';

    // Discard delimiters after the ID
    while (is_value_delimiter(reader_->peek_char())) {
      reader_->get_char();
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
              << "' at position " << std::to_string(initial_pos) << std::endl;
#endif

    // Decides whether this is a new example or an existing one
    if (previous_sequence_id_ != sequence_id && sequence_id != LONG_MAX) {
      is_new = true;
    }

    previous_sequence_id_ = sequence_id;
    has_initial_sequence_id_ = true;
    return is_new;
  }

  bool get_input_stream(
      const CTFSequenceID& sequence_id,
      bool& is_new_sequence) {
    // Create a new sequence with input_streams_info pre-allocated
    if (is_new_sequence) {
      is_new_sequence = false;

#ifdef CTF_DEBUG
      dataset_->sequences_id.emplace_back(sequence_id);
#endif

      // New sequence to be appended to dataset_->sequences
      CTFSequenceData sequence;

      for (auto const& stream : dataset_->input_streams_info) {
        CTFInputStreamID input_stream_id =
            dataset_->input_streams_map[stream.name];
        if (stream.storage == CTFDataStorage::Dense) {
          // TODO: Performance consideration: CNTK knows the number of samples
          // in the sequence, allowing accurate memory reservation (index built
          // during init)
          sequence.emplace_back(
              std::make_shared<CTFDenseInputStreamData<DataType>>(
                  input_stream_id, stream.dimension));
        } else {
          sequence.emplace_back(
              std::make_shared<CTFSparseInputStreamData<DataType>>(
                  input_stream_id, stream.dimension));
        }
      }

      dataset_->sequences.emplace_back(sequence);
    }

    // Reads the Input Stream name and lookup its input stream reference
    CTFInputStreamID input_stream_id;
    if (!get_input_stream_name(input_stream_id)) {
      return false;
    }
    const CTFInputStreamInformation& input_stream =
        dataset_->input_streams_info[input_stream_id];

    // Appends all values to the input stream
    if (!get_input_stream(input_stream)) {
      return false;
    }

    // TODO: Check actual number of values records of the stream
    return true;
  }

  // Parses input name from buffer and returns both CTFInputStreamInformation
  // reference and true if the input name belongs to an existing Input Stream
  bool get_input_stream_name(CTFInputStreamID& input_stream_id) {
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
    std::string name = std::string(scratch_.begin(), scratch_.begin() + idx);
#ifdef CTF_DEBUG
    std::cout << "Found CTF Name '" << name << "' at position " << initial_pos
              << std::endl;
#endif

    /// Match input name with the ones at 'features' and 'labels'
    bool found = false;
    auto it = dataset_->input_streams_map.find(name);
    if (it != dataset_->input_streams_map.end()) {
      input_stream_id = it->second;
      found = true;
    }

    if (!found) {
      std::string error_msg(
          "CTF Stream not found for input name '" + name + "'.");
#ifdef CTF_DEBUG
      std::cerr << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }

    return true;
  }

  bool get_input_stream_value(const CTFInputStreamInformation& input_stream) {
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
        if (input_stream.storage == CTFDataStorage::Dense) {
          std::string error_msg(
              "Unexpected sparse index delimiter ':' at position " +
              std::to_string(reader_->get_position()));
#ifdef CTF_DEBUG
          std::cerr << error_msg << std::endl;
#endif
          throw std::runtime_error(error_msg);
        }
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
      if (input_stream.storage == CTFDataStorage::Dense) {
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
      if (input_stream.storage != CTFDataStorage::Dense) {
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

    if (input_stream.storage == CTFDataStorage::Dense) {
      CTFDenseInputStreamData<DataType>* dense_data =
          static_cast<CTFDenseInputStreamData<DataType>*>(
              (dataset_->sequences.back())[input_stream.__id__].get());
      dense_data->data.emplace_back(ctf_value);
    } else {
      CTFSparseInputStreamData<DataType>* sparse_data =
          static_cast<CTFSparseInputStreamData<DataType>*>(
              (dataset_->sequences.back())[input_stream.__id__].get());
      // std::cerr << "data.emplace_back(" << ctf_value << ") for input stream
      // id " << input_stream.__id__ << std::endl;
      sparse_data->data.emplace_back(ctf_value);
      sparse_data->indices.emplace_back(ctf_index);
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

  bool get_input_stream(const CTFInputStreamInformation& input_stream) {
    // Adds a new row start for the input stream
    if (input_stream.storage == CTFDataStorage::Sparse) {
      CTFSparseInputStreamData<DataType>* sparse_data =
          static_cast<CTFSparseInputStreamData<DataType>*>(
              (dataset_->sequences.back())[input_stream.__id__].get());
    }

    // Get them all and push to th right stream
    while (!is_name_prefix(reader_->peek_char()) &&
           !is_comment_prefix(reader_->peek_char()) &&
           !is_eol(reader_->peek_char())) {
      if (!get_input_stream_value(input_stream)) {
#ifdef CTF_DEBUG
        std::cout << "CTF Value not found. An empty one will be used."
                  << std::endl;
#endif
      }
    }

    return true;
  }

  // type for CTF values
  CTFDataType data_type_;
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
};

} // namespace ctf
} // namespace data
} // namespace torch
