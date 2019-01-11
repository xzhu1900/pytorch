#pragma once

#include <torch/data/ctf/utils.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace torch {
namespace data {
namespace ctf {

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

} // namespace ctf
} // namespace data
} // namespace torch