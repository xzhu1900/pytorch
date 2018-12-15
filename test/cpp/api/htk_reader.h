#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <cassert>
#include <optional>

#include <cstddef>
#include <typeinfo>
#include <cstring>

using namespace std;

#define DEBUG 1

template <typename FunctorType>
struct DeferCleanupType
{
public:
    explicit DeferCleanupType(FunctorType f) : f_(f) {}
    ~DeferCleanupType() { if (m_cleanup) { f_(); } }
    void Reset() { m_cleanup = false; }

private:
    FunctorType f_;
    bool m_cleanup = true;
};

template <typename FunctorType>
DeferCleanupType<FunctorType> DeferCleanup(FunctorType f) { return DeferCleanupType<FunctorType>(f); }

union {
    uint16_t s;
    unsigned char c[2];
} constexpr static endian_checker{1};

constexpr bool is_little_endian()
{
    return endian_checker.c[0] == 1;
}

static bool is_little_endian_  = is_little_endian();

struct Utterance
{
  std::vector<float> feature;
  std::vector<uint16_t> label; // The models speech team have are about 10k labels. so ushort is more than enougth.
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

template <typename T>
static torch::optional<T> ReadDebugData(char* buffer, size_t& start_pos)
{
#ifdef DEBUG
  T result;
  memcpy(&result, buffer+start_pos, sizeof(T));

  start_pos += sizeof(T);
  return is_little_endian_ ? swap_endian<T>(result) : result;
#else
  start_pos += sizeof(T);
  return {};
#endif

}

static void ReadString(char* buffer, size_t& start_pos, const char* tagName, size_t tag_len)
{
#ifdef DEBUG
    //auto tag_len= tagName.size();

    int compare_result = std::memcmp(buffer, tagName, tag_len);
    if(compare_result != 0)
    {
        std::string actualData(buffer, tag_len);
        std::string tag(tagName, tag_len);
        std::string error_msg = "Tag mismatch. Expected: " + tag + ", actual: " + actualData;
        throw std::runtime_error(error_msg);
    }
    start_pos += tag_len;
#else
    start_pos += tag_len;
#endif
}

template <typename T>
static T ReadData(char* buffer, size_t& start_pos)
{
  T result;
  memcpy(&result, buffer+start_pos, sizeof(T));

  start_pos += sizeof(T);

  return is_little_endian_ ? swap_endian<T>(result) : result;
}

template <typename T>
static void ReadDataArray(char* buffer, size_t& start_pos, T *value, size_t byte_length)
{
  size_t value_size = sizeof(T);
  size_t array_len = byte_length / value_size;

  T data[array_len];
  memcpy(&data[0], buffer+start_pos, byte_length);

  for(int i=0;i<array_len;++i)
  {
    *value = is_little_endian_ ? swap_endian<T>(data[i]) : data[i];
    value++;
  }
  start_pos += byte_length;
}

template <typename T>
static void ReadEncodedDataArray(char* buffer, size_t& start_pos, T *value, size_t full_length, size_t byte_length)
{
  size_t value_size = sizeof(T);
  size_t decoded_array_len = full_length / sizeof(uint32_t);

  size_t binary_array_len = byte_length / value_size;
  T encoded_data[binary_array_len];

  if(binary_array_len % 2 != 0)
  {
      std::string error_msg = "Encoded label data corrupted. Expected binary is not a multiplication of 2";
      throw std::runtime_error(error_msg);
  }

  size_t label_count = binary_array_len / 2;

  memcpy(encoded_data, buffer+start_pos, byte_length);

  size_t decoded_index = 0;
  for(int i=0;i<label_count;++i){
      T label = is_little_endian_? swap_endian<T>(encoded_data[2 * i]) : encoded_data[2 * i];
      T duplicate_count = is_little_endian_? swap_endian<T>(encoded_data[2 * i + 1]) : encoded_data[2 * i + 1];
      assert(decoded_index + duplicate_count <=  decoded_array_len);
      std::fill_n(value + decoded_index, duplicate_count, label);
      decoded_index += duplicate_count;
  }
  assert(decoded_index == decoded_array_len);
  start_pos += byte_length;
}

class HTKParser
{
public:
  using FeatureType = float;
  using LabelType = uint16_t;
  using LatticeType = uint8_t;

  explicit HTKParser(const string &file_directory, const string &file_set_name);

  size_t get_chunk_count() { return chunk_info_list_.size(); }

  virtual ~HTKParser()
  {
  };

  std::vector<Utterance> parse_chunk(size_t chunk_id);

private:
  HTKParser() = delete;
  void parse_file_set(const string &file_set_name);

  void parse_encoded_data(const std::string &file_name, const size_t chunk_size, std::vector<Utterance> &data);

  template <typename T>
  void parse_data(const std::string &file_name, const std::string &target_name, const size_t chunk_size, std::vector<Utterance> &data)
  {
    ifstream m_chunk_file_stream;
    m_chunk_file_stream.open(file_name.data(), ios::in | ios::binary | std::ios::ate);

    auto deferredCleanup = DeferCleanup([&]()
    {
        m_chunk_file_stream.close();
    });

    std::streamsize size = m_chunk_file_stream.tellg();
    m_chunk_file_stream.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (m_chunk_file_stream.read(buffer.data(), size))
    {
      // Read header string
      size_t start_pos = 0;
      size_t end_pos = size;
      ReadString(buffer.data(), start_pos, target_name.data(), target_name.length());

      // Read version number
      torch::optional<int32_t> version = ReadDebugData<int32_t>(buffer.data(), start_pos);

      int element_count = 0;

      for (int element_count = 0; element_count < chunk_size; ++element_count)
      {

        torch::optional<int32_t> id = ReadDebugData<int32_t>(buffer.data(), start_pos);

  #ifdef DEBUG
        if (id && id.value() != element_count)
        {
          std::string error_msg = "File " + file_name + "is corrupted. The number " + std::to_string(element_count) + " element is missing";
          throw std::runtime_error(error_msg);
        }
  #endif

        // TODO: we don't need to parse all data_nyte_size data, if it is greater than truncated_dim * feature dim.
        uint32_t data_byte_size = ReadData<uint32_t>(buffer.data(), start_pos);

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
          ReadDataArray(buffer.data(), start_pos, reinterpret_cast<T *>(data_ptr), data_byte_size);
        }
        else if (target_name == "Lattice")
        {
          data[element_count].lattice.resize(data_size);
          data_ptr = reinterpret_cast<void *>(data[element_count].lattice.data());
          ReadDataArray(buffer.data(), start_pos, reinterpret_cast<T *>(data_ptr), data_byte_size);
        }
        else if (target_name == "Label")
        {
          data[element_count].label.resize(data_size);
          data_ptr = reinterpret_cast<void *>(data[element_count].label.data());

          uint16_t encoded_byte_size = ReadData<uint16_t>(buffer.data(), start_pos);

          if(encoded_byte_size % sizeof(uint16_t) != 0)
          {
              std::string error_msg = "Encoded label data corrupted. Expected binary is not a multiplication of " + std::to_string(sizeof(uint16_t));
              throw std::runtime_error(error_msg);
          }

          ReadEncodedDataArray(buffer.data(), start_pos, reinterpret_cast<T *>(data_ptr), data_byte_size, encoded_byte_size);
        }
        else
        {
          std::string error_msg = "unknown field " + target_name;
          throw std::runtime_error(error_msg);
        }

      }
      assert(start_pos==end_pos);
      m_chunk_file_stream.close();
    }
  }

  struct ChunkInfo
  {
    std::string file_name;
    size_t chunk_size;
  };

  string file_directory_;
  std::vector<ChunkInfo> chunk_info_list_;
  std::vector<std::string> file_extensions_;
};