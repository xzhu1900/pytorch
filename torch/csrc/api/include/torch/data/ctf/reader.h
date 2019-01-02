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

  bool can_read(void) const;
  const char& peek_char();
  const char& get_char();
  const size_t& get_position(void) const;
  void rewind_char();

 private:
  /// File handling
  bool refill(void);
  bool can_buffer(void) const;
  bool is_buffer_empty(void) const;
  std::string filename_;
  std::size_t file_size_;
  std::shared_ptr<FILE> file_;
  size_t file_pos_;
  bool is_eof_;

  /// Buffer handling buffer_size must be big enough
  /// to fit a really long line on the CTF file
  static const size_t MAX_BUFFER_SIZE = 2 * 1024 * 1024;
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