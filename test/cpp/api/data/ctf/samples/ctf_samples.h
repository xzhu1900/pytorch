#pragma once

#include <torch/data/ctf/ctf_parser.h>
#include <string>

namespace torch {
namespace data {
namespace ctf {

static const std::string CTF_SAMPLE_DIR("./test/cpp/api/data/ctf/samples");

#ifdef CTF_DEBUG
template <typename DataType>
void print_data(CTFDataset<DataType> dataset) {

  size_t index = 0;
  for (const auto& sequence_data : dataset.sequences) {
    std::cerr << dataset.sequences_id[index] << " ";
    for (const auto input_stream : sequence_data) {
      auto input_stream_id = input_stream.get()->input_stream_id;
      const auto& input_stream_info = dataset.input_streams[input_stream_id];

      std::string input_stream_type;
      if (input_stream_info.type == CTFInputStreamType::Feature) {
        input_stream_type = "F";
      } else {
        input_stream_type = "L";
      }
      std::cerr << " |" << input_stream_info.name << "(" << input_stream_type
                << ")";

      if (input_stream_info.storage == CTFDataStorage::Dense) {
        CTFDenseInputStreamData<DataType>* dense_data =
            reinterpret_cast<CTFDenseInputStreamData<DataType>*>(
                input_stream.get());

        if (dense_data->data.empty()) {
          std::cerr << " <empty>";
        } else {
          for (const auto& value : dense_data->data) {
            std::cerr << " " << value;
          }
        }
      } else {
        // TODO: print row start somewhere
        CTFSparseInputStreamData<DataType>* sparse_data =
            reinterpret_cast<CTFSparseInputStreamData<DataType>*>(
                input_stream.get());

        if (sparse_data->data.empty()) {
          std::cerr << " <empty>";
        } else {
          size_t col_index = 0;
          for (const auto& value : sparse_data->data) {
            std::cerr << " " << sparse_data->indices[col_index++] << ":"
                      << value;
          }
        }
      }
    }
    std::cerr << std::endl;
    ++index;
  }
}
#endif

} // namespace ctf
} // namespace data
} // namespace torch
