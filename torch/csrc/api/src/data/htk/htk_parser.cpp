// XXXX
//#include "torch/data/htk/htk_parser.h"
#include "htk_parser.h"
#include <cassert>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;

union {
  uint16_t s;
  unsigned char c[2];
} constexpr static  d {1};

constexpr bool is_little_endian() {
  return d.c[0] == 1;
}

bool HTKParser::is_little_endian_ = is_little_endian();

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

/*std::vector<HTKParser::SupportedField> HTKParser::supported_fields_ = {
    { "feature", HTKParser::parse_feature},
    { "label", HTKParser::parse_label},
    { "lattice", HTKParser::parse_lattice}
};*/

HTKParser::HTKParser(const string& file_directory, const string& file_set_name):
    file_directory_(file_directory)
{
    parse_file_set(file_set_name);
}

void HTKParser::parse_file_set(const string& file_set_name)
{
    // Read json.
    ptree doc;
    read_json(file_set_name, doc);
    //std::istringstream json_file_stream(file_set_name);
    //read_json (is, pt2);
    //auto foo = doc.get_child("fileType");

    //std::vector<std::string> r;
    for(auto& item : doc.get_child("fileType"))
    {
        file_extensions_.push_back(item.second.get_value<std::string>());
    }

    for(auto& item : doc.get_child("fileInfo"))
    {
        ChunkInfo info;
        info.file_name = item.second.get<std::string>("name");
        info.chunk_size = item.second.get<size_t>("count");
        chunk_info_list_.push_back(info);
    }

    /*file_extensions_.push_back("feature");
    file_extensions_.push_back("label");

    chunk_info_list_ = {{"chunk0", 10}, {"chunk1",10}, {"chunk2", 10}};*/
}

std::vector<Utterance> HTKParser::parse_chunk(size_t chunk_id)
{
    std::vector<Utterance> result;
    if(chunk_id >= chunk_info_list_.size())
    {
        std::string error_msg = "chunk with id = " + std::to_string(chunk_id) + "does not exist.";
        throw std::runtime_error(error_msg);
    }

    assert(!m_chunk_file_stream.is_open());

    for(auto& extension : file_extensions_)
    {
      for (auto& chunk_info : chunk_info_list_) {
        std::string file_full_name = file_directory_ + chunk_info.file_name + "." +extension;
        result.resize(chunk_info.chunk_size);


        if (extension == "feature") {
          parse_feature(file_full_name, chunk_info.chunk_size, result);
        } else if (extension == "lattice") {
          parse_lattice(file_full_name, result);
        } else if (extension == "label") {
          parse_label(file_full_name, chunk_info.chunk_size, result);
        } else {
          std::string error_msg = "unknown field " + extension;
          throw std::runtime_error(error_msg);
        }
      }
    }
    return result;
}

void ValidateTagMatch(const char* data, const char* header, size_t dataLen, size_t headerLen)
{
    if(dataLen != headerLen)
    {
        std::string tag(header, headerLen);
        std::string actualData(data, dataLen);
        std::string error_msg = "Tag mismatch. Expected: " + tag + ", actual: " + actualData;
        throw std::runtime_error(error_msg);
    }
    for(int i = 0; i<dataLen; ++i)
    {
        if(data[i] != header[i])
        {
            std::string tag(header, headerLen);
            std::string actualData(data, dataLen);
            std::string error_msg = "Tag mismatch. Expected: " + tag + ", actual: " + actualData;
            throw std::runtime_error(error_msg);
        }
    }
}

void HTKParser::ReadString(string_view tagName)
{
    #ifdef _DEBUG
    std::vector<char> header(tagName.size());
    m_chunk_file_stream.read(header.data(), header.size());
ValidateTagMatch(header.data(), tagName.data(), header.size(), tagName.size());
#else
    m_chunk_file_stream.ignore(tagName.size() - 1);

    #endif
}

void HTKParser::parse_feature(const std::string& file_name, const size_t chunk_size, std::vector<Utterance>& data){
    assert(!m_chunk_file_stream.is_open());
    m_chunk_file_stream.open(file_name.data(), ios::in | ios::binary);

    ReadString("Feature");

    // Read version number
    auto version = ReadDebugData<int>();

    int element_count = 0;

    for(int element_count=0;element_count<chunk_size;++element_count)
    {

        auto id = ReadDebugData<int>();

    #ifdef _DEBUG
        if(id && id.value() != element_count)
        {
            std::string error_msg = "File " + file_name + "is corrupted. The number" + std::to_string(element_count) + "element is missing";
            throw std::runtime_error(error_msg);
        }
    #endif
        int32_t feature_size = ReadData<int32_t>();

        if(feature_size % sizeof(float) != 0)
        {
            std::string error_msg = "feature size in byte is not a multiplication of float";
            throw std::runtime_error(error_msg);
        }

        int32_t feature_dim = feature_size / sizeof(float);
        data[element_count].feature.resize(feature_dim);

        float* data_ptr = data[element_count].feature.data();
        ReadData(data_ptr, feature_size);
    }
}
void HTKParser::parse_label(const std::string& file_name, const size_t chunk_size, std::vector<Utterance>& data){
    assert(!m_chunk_file_stream.is_open());
    m_chunk_file_stream.open(file_name.data(), ios::in | ios::binary);

    ReadString("Label");

    // Read version number
    auto version = ReadDebugData<int>();

    int element_count = 0;

    for(int element_count=0;element_count<chunk_size;++element_count)
    {

        auto id = ReadDebugData<int>();

    #ifdef _DEBUG
        if(id && id.value() != element_count)
        {
            std::string error_msg = "File " + file_name + "is corrupted. The number" + std::to_string(element_count) + "element is missing";
            throw std::runtime_error(error_msg);
        }
    #endif
        int32_t feature_size = ReadData<int32_t>();

        if(feature_size % sizeof(int32_t) != 0)
        {
            std::string error_msg = "feature size in byte is not a multiplication of float";
            throw std::runtime_error(error_msg);
        }
        int32_t feature_dim = feature_size / sizeof(int32_t);
        data[element_count].label.resize(feature_dim);
        int32_t* data_ptr = data[element_count].label.data();
        ReadData(data_ptr, feature_size);
    }

    m_chunk_file_stream.close();

}
void HTKParser::parse_lattice(const std::string& file_name, std::vector<Utterance>& data){

}