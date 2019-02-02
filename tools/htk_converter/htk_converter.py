from htk_dataset import HTKDataset
import numpy
import math

# TODO: casing: camel or _ delimeter

class HTKConverter():

    def __init__(self, dataset):
        self.dataset = dataset
        self.chunk_list = list()

    def writeChunkFiles(self, is_big_endian, chunk_size, output_dir, version):
        print("*** write binary with chunk size:{}".format(chunk_size))

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
                        # Convert to numpy array and use numpy function to write.
                        # Somehow the old python newLabel.write(size.to_bytes(4, byteorder=endian))
                        # function writes gabage on the first two bytes.
                        if is_big_endian:
                            numpy.array([size],dtype='>u4').tofile(newFeature)
                        else:
                            numpy.array([size],dtype='<u4').tofile(newFeature)
                        newFeature.write(x)
                        newFeature.flush()

                        newLabel.write(chunk_idx.to_bytes(4, byteorder=endian))
                        y_len = len(y)

                        # write label in binary format. If an utterance's label is 10, 10, 1, 2, 2, 2, 2, 2, 3
                        # then the written file will write 18, 16, 10, 2, 1, 1, 2, 5, 3, 1
                        # the first 18 means the label array takes 18 bytes. Since each label is stored in
                        # 2 bytes, we can get there are 9 labels. The second 16 means we are going to write
                        # the following 16 bytes for the label data. The 16 bytes, with 2 bytes for a uint number,
                        # present 8 numbers. With 2 numbers in a pair, we write 4 pairs. Then we read the rest in pairs (10, 2),
                        # (1, 1), (2, 5), (3, 1). It means 10 appears 2 times, 1 one time, 2 five times and
                        # 3 one time, thus we can restore the full label array.
                        frame_len = self.dataset.__getitemframe__(i)

                        # label_byte_size is twice frame_len as each label is stored with 2 bytes.
                        label_byte_size = (frame_len * 2)
                        newLabel.write(label_byte_size.to_bytes(4, byteorder=endian))

                        # label length in bytes. Each label has the label value and how many times it repeats.
                        y_len_in_bytes = y_len * 2 * 2
                        newLabel.write(y_len_in_bytes.to_bytes(2, byteorder=endian))
                        for j in range(y_len):
                            newLabel.write(y[j][0].to_bytes(2, byteorder=endian))
                            dup_count = y[j][1]
                            newLabel.write(dup_count.to_bytes(2, byteorder=endian))

                        newLabel.flush()
                        chunk_idx = chunk_idx + 1
            index = index + 1

        # generate the json file
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
        """
        debug usage function. Write the parsed feature data in plain text.
        """
        print("*** write feature data in plain text")

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
            x, _, _ = self.dataset.__getchunkateditem__(index, 16)
            full_array.append(x)
            index = index + 1
        x_np = numpy.asarray(full_array)
        numpy.save("feature_converter.npy", x_np)

    def writeLabelData(self, output_dir):
        """
        debug usage function. Write the parsed label data in plain text.
        """
        print("*** write label data in plain text")

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