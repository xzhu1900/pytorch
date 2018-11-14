#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <functional>

//#include "c10/util/Optional.h"
//#include "c10/util/C++17.h"
#include <optional>

#include <cstddef>

using namespace std;
//using c10::optional;

#define _DEBUG 1

struct Utterance
{
  std::vector<float> feature;
  std::vector<int32_t> label;
  std::vector<uint8_t> lattice;
};

class HTKParser
{
public:
  using BYTE = std::uint8_t;
  explicit HTKParser(const string &file_directory, const string &file_set_name);

  virtual ~HTKParser(){
    if(m_chunk_file_stream.is_open()){
      m_chunk_file_stream.close();
    }

  };

  std::vector<Utterance> parse_chunk(size_t chunk_id);

  void PrintData() const {};

private:
  HTKParser() = delete;
  void parse_file_set(const string &file_set_name);
  void parse_feature(const std::string &file_name, size_t chunk_size, std::vector<Utterance> &data);
  void parse_label(const std::string &file_name, const size_t chunk_size, std::vector<Utterance> &data);
  void parse_lattice(const std::string &file_name, std::vector<Utterance> &data);

  // XXX make it to be static
  template <typename T>
  T ReadData()
  {
    T result;
    m_chunk_file_stream.read(reinterpret_cast<char *>(&result), sizeof(result));
    if (is_little_endian_)
    {
      return swap_endian<T>(result);
    }
    else
    {
      return result;
    }
  }

  template <typename T>
  //c10::optional<T> ReadDebugData()
  std::optional<T> ReadDebugData()
  {
#ifdef _DEBUG
    T result;
    m_chunk_file_stream.read(reinterpret_cast<char *>(&result), sizeof(result));
    if (is_little_endian_)
    {
      return swap_endian<T>(result);
    }
    else
    {
      return result;
    }
#else
    m_chunk_file_stream.ignore(sizeof(T));
    return {};
#endif
  }

  template <typename T>
  T swap_endian(T u)
  {
    union {
      T u;
      unsigned char u8[sizeof(T)];
    } source, dest;

    source.u = u;

    for (size_t k = 0; k < sizeof(T); k++)
      dest.u8[k] = source.u8[sizeof(T) - k - 1];

    return dest.u;
  }

  template <typename T>
  void ReadData(T *value, size_t length)
  {
    size_t value_size = sizeof(T);
    while (length > 0)
    {
      T v;
      m_chunk_file_stream.read(reinterpret_cast<char *>(&v), value_size);

      if (is_little_endian_)
      {
        v = swap_endian<T>(v);
      }
      *value = v;
      //m_chunk_file_stream.read(reinterpret_cast<char *>(value), value_size);
      //assert(length >= value_size);
      length -= value_size;
      value++;
    }
  }

  void ReadString(string_view tagName);

  struct SupportedField
  {
    const char *field_name;
    //const std::string field_name;
    //int a;
    std::function<void()> parser_function;
  };

  struct ChunkInfo
  {
    std::string file_name;
    size_t chunk_size;
  };

  string file_directory_;
  ifstream m_chunk_file_stream;
  std::vector<ChunkInfo> chunk_info_list_;
  std::vector<std::string> file_extensions_;

  static bool is_little_endian_;

  //SupportedField t = { "feature", &HTKParser::parse_feature};

  // make it static?
  /*const std::vector<SupportedField> supported_fields_ = {
    { "feature", &HTKParser::parse_feature},
    { "label", &HTKParser::parse_label},
    { "lattice", &HTKParser::parse_lattice}
  };*/
  /*const std::vector<SupportedField> supported_fields_ = {
    { "feature", 1},
    { "label", 2}
  };*/
};
