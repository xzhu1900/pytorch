from htk_dataset import HTKDataset
import numpy
import math

class HTKConverter():

    def __init__(self, dataset):
        self.dataset = dataset
        self.chunk_list = list()

    def writeBinary(self, is_big_endian, chunk_size, output_dir, version, mlf_format):
        print("*** write binary with chunk size:{}".format(chunk_size))

        assert mlf_format != "debug-binary" , "Error MLF file format. Expecting text or binary"

        isBinary = (mlf_format == "binary")

        if(output_dir[-1] != '/'):
            output_dir += '/'

        feature_size = len(self.dataset.utt_list)
        utterance_per_chunk = chunk_size
        chunkFileCount = int(math.ceil(feature_size  / utterance_per_chunk))
        index = 0
        while index < chunkFileCount:
            featureName = "{}chunk{}.feature".format(output_dir, index)
            labelName = "{}chunk{}.label".format(output_dir, index)

            print("writing to file {} and {}".format(featureName, labelName))
            endian = "big"
            if(is_big_endian!= True) :
                endian = "little"

            with open(featureName, 'wb') as newFeature:
                with open(labelName, 'wb') as newLabel:
                    newFeature.write("Feature".encode())
                    newFeature.write((version).to_bytes(4, byteorder=endian))
                    newLabel.write("Label".encode())
                    newLabel.write((version).to_bytes(4, byteorder=endian))

                    start = index * utterance_per_chunk
                    end = min(start + utterance_per_chunk, feature_size)
                    chunk_idx = 0
                    self.chunk_list.append(end-start)
                    for i in range(start, end):
                        x, size, y = self.dataset.__getitembinary__(i)
                        newFeature.write(chunk_idx.to_bytes(4, byteorder=endian))
                        # somehow newLabel.write doesn't work for variable size -__-, switch to numpy tofile method for this one.
                        numpy.array([size],dtype='>u4').tofile(newFeature)
                        newFeature.write(x)
                        newFeature.flush()

                        newLabel.write(chunk_idx.to_bytes(4, byteorder=endian))
                        y_len = len(y)
                        if (isBinary):
                            frame_len = self.dataset.__getitemframe__(i)
                            uttrance_len = (frame_len * 2)
                            newLabel.write(uttrance_len.to_bytes(4, byteorder=endian))
                            y_len_in_bytes = y_len * 2 * 2
                            newLabel.write(y_len_in_bytes.to_bytes(2, byteorder=endian))
                            for j in range(y_len):
                                newLabel.write(y[j][0].to_bytes(2, byteorder=endian))
                                dup_count = y[j][1]
                                newLabel.write(dup_count.to_bytes(2, byteorder=endian))
                        else:
                            full_length = 0
                            uttrance_len = self.dataset.__getitemframe__(i)
                            newLabel.write(uttrance_len.to_bytes(4, byteorder=endian))

                            for j in range(y_len):
                                dup_count = y[j][1]
                                full_length += dup_count
                                for w in range(dup_count):
                                    newLabel.write(y[j][0].to_bytes(2, byteorder=endian))
                            assert full_length == uttrance_len
                        newLabel.flush()
                        chunk_idx = chunk_idx + 1
            index = index + 1

        jsonFileName = "{}fileSet.json".format(output_dir)
        with open(jsonFileName, 'w') as newJson:
            print("writing to file {}".format(jsonFileName))
            newJson.write("{\n\"fileType\": [\n    \"feature\",\n    \"label\"\n],\n\n\"fileInfo\": [\n")
            i = 0
            while i < (len(self.chunk_list) - 1):
                newJson.write("    {{\"name\":\"chunk{}\",\"count\":{}}},\n".format(i, self.chunk_list[i]))
                i = i+1
            last_index = (len(self.chunk_list) - 1)
            newJson.write("    {{\"name\":\"chunk{}\",\"count\":{}}}\n".format(last_index, self.chunk_list[last_index]))
            newJson.write("]\n}\n")

    def writeFeatureData(self, output_dir, seq_len, feature_dim):
        print("*** write feature data")

        if(output_dir[-1] != '/'):
            output_dir += '/'

        featureName = "{}feature.txt".format(output_dir)
        feature_size = len(self.dataset.utt_list)
        index = 0

        with open(featureName, 'w') as newFeature:
            while index < feature_size:
                x, _, _ = self.dataset.__getitem__(index)
                row = len(x)
                column = len(x[0])
                newFeature.write("feature {}: size: {}, min: {}, max: {}\n".format(index, x.size, x.min(), x.max()))
                newFeature.write("[")
                for i in range(row):
                    newFeature.write("[")
                    for j in range(column):
                        newFeature.write("{:.2f}, ".format(x[i][j], 2))
                        if ((j + 1 ) % feature_dim ==0):
                            newFeature.write("]\n[")
                    newFeature.write("]\n")
                newFeature.write("]\n")
                index = index + 1

        full_array = []
        index = 0
        while index < feature_size:
            x, _, _ = self.dataset.__getitemchunkated__(index, 16)
            full_array.append(x)
            index = index + 1
        x_np = numpy.asarray(full_array)
        numpy.save("feature_converter.npy", x_np)

    def writeLabelData(self, output_dir, mlf_format):
        print("*** write label data")

        if(output_dir[-1] != '/'):
            output_dir += '/'

        labelName = "{}label.txt".format(output_dir)
        label_size = len(self.dataset.utt_list)
        index = 0

        with open(labelName, 'w') as newLabel:
            while index < label_size:
                _, y, _ = self.dataset.__getitem__(index)
                full_length = 0
                y_len = len(y)
                for i in range(y_len):
                    full_length += y[i][1]

                uttrance_len = self.dataset.__getitemframe__(index)
                assert (uttrance_len) == full_length

                newLabel.write("label {}: label count: {}, full length: {}\n".format(index, len(y), full_length))
                newLabel.write("[")

                for i in range(y_len):
                    newLabel.write("{{{}, {}}}, ".format(y[i][0], y[i][1]))
                newLabel.write("]\n")
                index = index + 1



