#pragma once

namespace torch {
namespace data {
namespace ctf {

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