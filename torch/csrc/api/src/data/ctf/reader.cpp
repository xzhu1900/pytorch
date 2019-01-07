#include <torch/data/ctf/reader.h>
#include <torch/data/ctf/reader_constants.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>

namespace torch {
namespace data {
namespace ctf {

/*
 * Reader class for CTF
 *
 * RAII pattern was used for file descriptor
 */

Reader::~Reader() {}

Reader::Reader(const std::string& filename)
    : filename_(filename),
      file_pos_(0),
      is_eof_(false),
      buffer_pos_(0),
      buffer_size_(0),
      rewinded_char_(false),
      previous_char_(0) {
  std::FILE* const tmp = fopen(filename_.c_str(), "rbS");
  if (!tmp) {
    std::string error_msg(
        "Reader could not open the specified file (" + filename + ")");
#ifdef CTF_DEBUG
    std::cerr << error_msg << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }
  // Get file length
  file_ = std::shared_ptr<std::FILE>(tmp, std::fclose);
  std::fseek(file_.get(), 0, SEEK_END); // TODO: Non-portable as binary streams
                                        // are not required to support SEEK_END
  file_size_ = std::ftell(file_.get());
  std::rewind(file_.get());

   buffer_.resize(Reader::CTF_MAX_BUFFER_SIZE);
  refill();
}

bool Reader::refill(void) {
  if (!is_buffer_empty()) {
#ifdef CTF_DEBUG
    std::cout << "Buffer is not empty yet. Not refilling it" << std::endl;
#endif
    return false;
  }
  if (!can_buffer()) {
#ifdef CTF_DEBUG
    std::cout << "Nothing to read from file " << filename_ << ". ("
              << strerror(errno) << ")";
#endif
    is_eof_ = true;
    return false;
  }

  buffer_pos_ = 0;
  file_pos_ = ftell(file_.get());
  size_t bytes_read =
      std::fread(&buffer_[0], 1, Reader::CTF_MAX_BUFFER_SIZE, file_.get());

  if (feof(file_.get()) != 0) {
    is_eof_ = true;
  }

  if ((ferror(file_.get()) != 0) ||
      (bytes_read != Reader::CTF_MAX_BUFFER_SIZE && !std::feof(file_.get()))) {
    std::string error_msg(
        "Error reading file " + filename_ + ". " + strerror(errno));
#ifdef CTF_DEBUG
    std::cerr << error_msg << buffer_pos_ << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }
  buffer_size_ = bytes_read;
#ifdef CTF_DEBUG
  std::cout << "Buffer refilled. Read " << std::to_string(bytes_read)
            << " from file " << filename_ << std::endl;
#endif
  return true;
}

} // namespace ctf
} // namespace data
} // namespace torch