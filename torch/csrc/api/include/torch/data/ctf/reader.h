#pragma once

#include <torch/data/ctf/utils.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace data {
namespace ctf {

// TODO: Should we use Memory mapped files to speed buffering?

/// A sequential text reader to feed CTF parser
///
/// C File API was used due to performance constraints
/// Current implementation caches chunks of data from file in memory
/// and parses CTF from it. When it gets empty, buffer is refilled and the cycle
/// is repeated until EOF is reached
///
class Reader {
 public:
  virtual ~Reader();
  explicit Reader(const std::string& filename);

  inline bool can_read(void) const {
    return (!is_buffer_empty() || can_buffer());
  }
  inline const char& peek_char(void) {
    if (is_buffer_empty()) {
      refill();
    }
    if (rewinded_char_) {
      return previous_char_;
    } else {
      return buffer_[buffer_pos_];
    }
  }
  inline const char& get_char(void) {
    if (buffer_pos_ > 0) {
      previous_char_ = buffer_[buffer_pos_ - 1];
    }
    if (is_buffer_empty()) {
      refill();
    }
    if (rewinded_char_) {
      rewinded_char_ = false;
      return previous_char_;
    } else {
      return buffer_[buffer_pos_++];
    }
  }
  inline const size_t& get_position(void) const {
    return buffer_pos_;
  }
  inline void rewind_char(void) {
    rewinded_char_ = true;
  }

 private:
  /// File handling
  bool refill(void);
  inline bool can_buffer(void) const {
    return (!is_eof_);
  }
  inline bool is_buffer_empty(void) const {
    return ((buffer_size_ == 0) || (buffer_size_ == buffer_pos_));
  }
  std::string filename_;
  std::shared_ptr<FILE> file_;
  bool is_eof_;

  /// Buffer handling buffer_size must be big enough
  /// to fit a really long line on the CTF file
  const size_t CTF_MAX_BUFFER_SIZE = 2 * 1024 * 1024;
  std::vector<char> buffer_;
  size_t buffer_pos_;
  size_t buffer_size_;
  bool rewinded_char_;
  char previous_char_;

  Reader() = delete;
  DISALLOW_COPY_AND_ASSIGN(Reader);
};

static const char SPACE_CHAR = ' ';
static const char TAB_CHAR = '\t';
static const char NAME_PREFIX = '|';
static const char INDEX_DELIMITER = ':';
static const char ESCAPE_SYMBOL = '#';

inline bool is_name_prefix(const char& c) {
  return (c == NAME_PREFIX);
}

inline bool is_comment_prefix(const char& c) {
  return (is_name_prefix(c));
}

inline bool is_comment_suffix(const char& c) {
  return (c == '#');
}

inline bool is_decimal_point(const char& c) {
  return (c == '.');
}

inline bool is_sparse_value_delimiter(const char& c) {
  return (c == ':');
}

inline bool is_digit(const char& c) {
  return (c >= '0' && c <= '9');
}

inline bool is_alpha(const char& c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

inline bool is_sign(const char& c) {
  return c == '+' || c == '-';
}

inline bool is_exponent(const char& c) {
  return c == 'e' || c == 'E';
}

inline bool is_number(const char& c) {
  return (is_digit(c) || is_decimal_point(c) || is_sign(c) || is_exponent(c));
}

inline bool is_printable(const char& c) {
  return c >= SPACE_CHAR;
}

inline bool is_non_printable(const char& c) {
  return !is_printable(c);
}

inline bool is_value_delimiter(const char& c) {
  return c == SPACE_CHAR || c == TAB_CHAR;
}

inline bool is_eol(const char& c) {
  return (c == '\r' || c == '\n');
}

inline bool is_escape_delimiter(const char& c) {
  return (c == '\'' || c == '"');
}

inline bool is_column_delimiter(const char& c) {
  return is_value_delimiter(c) || (is_non_printable(c) && !is_eol(c));
}

} // namespace ctf
} // namespace data
} // namespace torch