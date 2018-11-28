#include <torch/data/ctf/reader.h>
#include <torch/data/ctf/reader_constants.h>

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
      is_eof_(false) {
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

  buffer_.reserve(Reader::MAX_BUFFER_SIZE);
}

size_t Reader::read_line(std::vector<char>& buffer, size_t size) {
  if (size <= 0) {
    std::string error_msg("Invalid buffer");
#ifdef CTF_DEBUG
    std::cerr << error_msg << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }
  if (fgets(buffer.data(), size, file_.get()) == nullptr) {
    if (feof(file_.get()) != 0) {
      is_eof_ = true;
    } else if (ferror(file_.get()) != 0) {
      std::string error_msg("Something went wrong while reading CTF file");
#ifdef CTF_DEBUG
      std::cerr << error_msg << std::endl;
#endif
      throw std::runtime_error(error_msg);
    }
  };
  return strnlen(buffer.data(), size);
}

size_t Reader::file_size() const {
  return file_size_;
}

bool Reader::seek(long int offset) {
  if (offset < 0 || offset > file_size_) {
    std::string error_msg(
        "Seeking file to invalid position (" + std::to_string(offset) + ")");
#ifdef CTF_DEBUG
    std::cerr << error_msg << std::endl;
#endif
    throw std::runtime_error(error_msg);
  }

#ifdef CTF_DEBUG
  std::cout << "Seeking file to offset " << offset << std::endl;
#endif

  // TODO: Non-portable as binary streams
  if (std::fseek(file_.get(), offset, SEEK_SET) == 0) {
    return true;
  } else {
    return false;
  }
}

bool Reader::can_read() {
  return (
      !is_eof_ && !std::feof(file_.get()) &&
      std::ftell(file_.get()) != file_size_);
}

std::string Reader::get_filename() const {
  return filename_;
}

} // namespace ctf
} // namespace data
} // namespace torch