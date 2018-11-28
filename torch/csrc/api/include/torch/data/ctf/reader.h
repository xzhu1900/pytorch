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
/// Currently, one character at a time is read from the file
/// C File API was used due to performance constraints
/// TODO: Switch implementation to cache chunks of data from file in memory
///       instead of reading each character directly from file
class Reader {
 public:
  virtual ~Reader();
  explicit Reader(const std::string& filename);

  size_t read_line(std::vector<char>& buffer, size_t size);
  size_t file_size() const;
  bool seek(long int offset);
  bool can_read();
  std::string get_filename() const;

 private:
  /// File handling
  std::string filename_;
  std::size_t file_size_;
  std::shared_ptr<FILE> file_;
  size_t file_pos_;
  bool is_eof_;

  /// Buffer handling buffer_size must be big enough
  /// to fit a really long line on the CTF file
  static const size_t MAX_BUFFER_SIZE = 2 * 1024 * 1024;
  std::vector<char> buffer_;

  Reader() = delete;
  DISALLOW_COPY_AND_ASSIGN(Reader);
};

} // namespace ctf
} // namespace data
} // namespace torch