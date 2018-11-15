#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <optional>

#include <cstddef>
#include <typeinfo>

using namespace std;

#define _DEBUG 1

struct Utterance
{
  std::vector<float> feature;
  std::vector<int32_t> label;
  std::vector<uint8_t> lattice;
};

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

class HTKParser
{
public:
  using FeatureType = float;
  using LabelType = int32_t;
  using LatticeType = uint8_t;

  explicit HTKParser(const string &file_directory, const string &file_set_name);

  virtual ~HTKParser()
  {
    if (m_chunk_file_stream.is_open())
    {
      m_chunk_file_stream.close();
    }
  };

  std::vector<Utterance> parse_chunk(size_t chunk_id);

private:
  HTKParser() = delete;
  void parse_file_set(const string &file_set_name);

  template <typename T>
  void parse_data(const std::string &file_name, const std::string &target_name, const size_t chunk_size, std::vector<Utterance> &data)
  {
    assert(!m_chunk_file_stream.is_open());
    m_chunk_file_stream.open(file_name.data(), ios::in | ios::binary);

    // Read header string
    ReadString(m_chunk_file_stream, target_name);

    // Read version number
    std::optional<int32_t> version = ReadDebugData<int32_t>(m_chunk_file_stream);

    int element_count = 0;

    for (int element_count = 0; element_count < chunk_size; ++element_count)
    {

      std::optional<int32_t> id = ReadDebugData<int32_t>(m_chunk_file_stream);

#ifdef _DEBUG
      if (id && id.value() != element_count)
      {
        std::string error_msg = "File " + file_name + "is corrupted. The number " + std::to_string(element_count) + " element is missing";
        throw std::runtime_error(error_msg);
      }
#endif
      int32_t data_byte_size = ReadData<int32_t>(m_chunk_file_stream);

      if (data_byte_size % sizeof(T) != 0)
      {
        std::string error_msg = "data size in byte is not a multiplication of " + std::string(typeid(T).name());
        throw std::runtime_error(error_msg);
      }

      int32_t data_size = data_byte_size / sizeof(T);
      void *data_ptr;

      if (target_name == "Feature")
      {
        data[element_count].feature.resize(data_size);
        data_ptr = reinterpret_cast<void *>(data[element_count].feature.data());
      }
      else if (target_name == "Lattice")
      {
        data[element_count].lattice.resize(data_size);
        data_ptr = reinterpret_cast<void *>(data[element_count].lattice.data());
      }
      else if (target_name == "Label")
      {
        data[element_count].label.resize(data_size);
        data_ptr = reinterpret_cast<void *>(data[element_count].label.data());
      }
      else
      {
        std::string error_msg = "unknown field " + target_name;
        throw std::runtime_error(error_msg);
      }
      ReadDataArray(m_chunk_file_stream, reinterpret_cast<T *>(data_ptr), data_byte_size);
    }
    m_chunk_file_stream.close();
  }

  template <typename T>
  static T ReadData(ifstream &reader)
  {
    T result;
    reader.read(reinterpret_cast<char *>(&result), sizeof(result));

    return is_little_endian_ ? swap_endian<T>(result) : result;
  }

  template <typename T>
  static std::optional<T> ReadDebugData(ifstream &reader)
  {
#ifdef _DEBUG
    T result;
    reader.read(reinterpret_cast<char *>(&result), sizeof(result));
    return is_little_endian_ ? swap_endian<T>(result) : result;
#else
    reader.ignore(sizeof(T));
    return {};
#endif
  }

  template <typename T>
  static void ReadDataArray(ifstream &reader, T *value, size_t byte_length)
  {
    size_t value_size = sizeof(T);
    while (byte_length > 0)
    {
      T v;
      reader.read(reinterpret_cast<char *>(&v), value_size);

      *value = is_little_endian_ ? swap_endian<T>(v) : v;

      byte_length -= value_size;
      value++;
    }
  }

  static void ReadString(ifstream &reader, string_view tagName);

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
};
