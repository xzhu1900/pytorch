from struct import unpack, pack
import numpy

import math
import struct

LPC = 1
LPCREFC = 2
LPCEPSTRA = 3
LPCDELCEP = 4
IREFC = 5
MFCC = 6
FBANK = 7
MELSPEC = 8
USER = 9
DISCRETE = 10
PLP = 11

_E = 0o000100  # has energy
_N = 0o000200  # absolute energy suppressed
_D = 0o000400  # has delta coefficients
_A = 0o001000  # has acceleration (delta-delta) coefficients
_C = 0o002000  # is compressed
_Z = 0o004000  # has zero mean static coefficients
_K = 0o010000  # has CRC checksum
_O = 0o020000  # has 0th cepstral coefficient
_V = 0o040000  # has VQ data
_T = 0o100000  # has third differential coefficients


class HTKFeat_read(object):
    """
    Read HTK format feature files
    From: https://github.com/ZhangAustin/Deep-Speech/blob/master/speech_io/feature_io_py35.py#L69
    """

    def __init__(self, filename=None):
        self.swap = (unpack('=i', pack('>i', 42))[0] != 42)
        if (filename != None):
            self.open1(filename)

    def __iter__(self):
        self.fh.seek(12, 0)
        return self

    def open1(self, filename):
        self.filename = filename
        self.fh = open(filename, "rb")
        self.readheader()

    def readheader(self):
        self.fh.seek(0, 0)
        spam = self.fh.read(12)

        self.nSamples, self.sampPeriod, self.sampSize, self.parmKind = unpack(">IIHH", spam)
        # Get coefficients for compressed data
        if self.parmKind & _C:
            self.dtype = 'h'
            self.veclen = int(self.sampSize / 2)
            if self.parmKind & 0x3f == IREFC:
                self.A = 32767
                self.B = 0
            else:
                self.A = numpy.fromfile(self.fh, 'f', self.veclen)
                self.B = numpy.fromfile(self.fh, 'f', self.veclen)
                if self.swap:
                    self.A = self.A.byteswap()
                    self.B = self.B.byteswap()
        else:
            self.dtype = 'f'
            self.veclen = int(self.sampSize / 4)
        self.hdrlen = self.fh.tell()

    def seek(self, idx):
        self.fh.seek(self.hdrlen + idx * self.sampSize, 0)

    def next(self):
        vec = numpy.fromfile(self.fh, self.dtype, self.veclen)
        if len(vec) == 0:
            raise StopIteration
        if self.swap:
            vec = vec.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            vec = (vec.astype('f') + self.B) / self.A
        return vec

    def readvec(self):
        return self.next()

    def getall(self):
        self.seek(0)
        data = numpy.fromfile(self.fh, self.dtype)

        if self.parmKind & _K:  # Remove and ignore checksum
            if (len(data) / self.veclen == self.nSamples):
                print(("MFC Header XXXX_K is not true! There is no additional checksum"
                       "at the end of the file! Ingore the XXXX_K in the header, "
                       "this will not affect anything."))
            else:
                data = data[:-1]

        data = data.reshape(len(data) // self.veclen, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

    def getchunk(self, startframe, endframe):
        self.seek(startframe)
        data = numpy.fromfile(self.fh, self.dtype, (endframe - startframe + 1) * self.veclen)

        data = data.reshape(endframe - startframe + 1, self.veclen)
        if self.swap:
            data = data.byteswap()
        # Uncompress data to floats if required
        if self.parmKind & _C:
            data = (data.astype('f') + self.B) / self.A
        return data

    def getchunkBinary(self, startframe, endframe):
        self.seek(startframe)

        chunk_binary = self.fh.read((endframe - startframe + 1) * self.veclen * 4)
        return chunk_binary


class HTKDataset():
    """
    Job is to read & procesc the mlflist, scp and statelist files.
    Make the index of the MLF and read a feature file on the fly.
    """

    def __init__(self, mlflist_file, scp_file, statelist_file, add_start_end=False):

        self._SOS = "<SOS>"
        self._EOS = "<EOS>"
        self.add_start_end = add_start_end

        self.phone2idx = None
        self.idx2phone = None
        self.token_list = None

        # will populate self.phone2idx & self.idx2phone within
        self.build_phone2idx(statelist_file)
        self.vocab_size = len(self.phone2idx)

        utt2labels = self.build_utt2labels(mlflist_file)

        self.utt_list = list()
        self.build_utt_list(scp_file, utt2labels)

        x, _, _ = self.__getitem__(0)
        self.input_dim = x.shape[1]

    def build_utt_list(self, scp_file, utt2labels):
        """
        internaly update self.utt_list
        """
        print("Starting to build utt_list")

        total_duration = 0
        total_duration_not_found = 0
        for line in open(scp_file, 'r'):
            line = line.rstrip()
            utt_id, chunk_path = line.split('=')
            #XXX bug here
            utt_id = utt_id.split(".")[0]
            chunk_path, frames = chunk_path.split('[')
            start_frame, end_frame = frames.split(',')
            end_frame = end_frame[:-1] # remove ']'
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            duration = end_frame - start_frame + 1
            duration = duration * 30 / 1000 # each frame 30ms, converting to secs
            # only add this sample to this list if a labels exists for it
            try:
                labels = utt2labels[utt_id]
                # TODO: do we need to filter this out?
                #if duration > 10 or len(labels) > 200:
                    #continue # Want to avoid absurdly long utterances that might cause OOM

                self.utt_list.append({
                    "utt_id": utt_id,
                    "chunk_path": chunk_path,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "duration": duration,
                    "labels": labels
                })
                total_duration += duration
            except KeyError as e:
                total_duration_not_found += duration
                pass

        print("utt_list built.")


    def build_utt2labels(self, mlflist_file):
        """
        Will read the MLF list file and accordingly return utt2labels
        """
        print("Starting to build utt2labels")

        utt2labels = dict()

        with open(mlflist_file, 'r') as fid:
            for mlf_chunk_file in fid:
                mlf_chunk_file = mlf_chunk_file.rstrip()
                # will internally update utt2labels
                self.read_mlf_chunk_file(mlf_chunk_file, utt2labels)

        print("utt2labels built")

        return utt2labels


    def build_phone2idx(self, statelist_file):
        """
        Read the statelist file or the label mapping file
        and populates self.phone2idx, self.idx2phone, self.token_list
        """
        print("Starting to build label list")

        self.token_list = list()
        self.phone2idx = dict()

        with open(statelist_file, 'r') as fid:
            for idx, line in enumerate(fid.readlines()):
                w = line.rstrip()
                self.phone2idx[w] = idx
                self.token_list.append(w)

        if self.add_start_end:
            self.phone2idx[self._EOS] = len(self.phone2idx)
            self.phone2idx[self._SOS] = len(self.phone2idx)

        self.idx2phone = {v: k for k, v in self.phone2idx.items()}

        print("label list built")


    def __len__(self):
        return len(self.utt_list)

    def __getitemframe__(self, idx):
        # read the chunk file
        start_frame = self.utt_list[idx]["start_frame"]
        end_frame = self.utt_list[idx]["end_frame"]
        return end_frame - start_frame + 1

    def __getitem__(self, idx):
        # read the chunk file
        chunk_path = self.utt_list[idx]["chunk_path"]
        start_frame = self.utt_list[idx]["start_frame"]
        end_frame = self.utt_list[idx]["end_frame"]
        x = self.read_chunk_path(chunk_path, start_frame, end_frame)
        y = self.utt_list[idx]["labels"]
        return x, y, self.utt_list[idx]["utt_id"]

    def __getitemchunkated__(self, idx, seq_len):
        # read the chunk file
        chunk_path = self.utt_list[idx]["chunk_path"]
        start_frame = self.utt_list[idx]["start_frame"]
        end_frame = self.utt_list[idx]["end_frame"]
        x = self.read_chunk_path(chunk_path, start_frame, start_frame + seq_len - 1)
        y = self.utt_list[idx]["labels"]
        return x, y, self.utt_list[idx]["utt_id"]

    def __getitembinary__(self, idx):
        # read the chunk file in binary
        chunk_path = self.utt_list[idx]["chunk_path"]
        start_frame = self.utt_list[idx]["start_frame"]
        end_frame = self.utt_list[idx]["end_frame"]
        x = self.read_chunk_path(chunk_path, start_frame, end_frame, True)
        yData = self.utt_list[idx]["labels"]
        return x, len(x), yData

    def read_chunk_path(self, chunk_path, start_frame, end_frame, readBinary=False):
        """
        Adapted from https://github.com/ZhangAustin/Deep-Speech/blob/master/speech_io/feature_io_py35.py#L232
        """
        mfc = HTKFeat_read(chunk_path)
        if (start_frame == end_frame == 0):
            obs = mfc.getall()
        elif (end_frame >= start_frame):
            if (readBinary) :
                return mfc.getchunkBinary(start_frame, end_frame)
            else:
                obs = mfc.getchunk(start_frame, end_frame)
        else:
            print("startframe must be less than or equal to endframe", flush=True)

        return obs

    def read_mlf_chunk_file(self, mlf_chunk_file, labels):

        # TODO: this only works with text format mlf file. We need to add suport for binary format as well.
        utt_id = None

        for line in open(mlf_chunk_file):
            line = line.rstrip()

            if len(line) < 1 or line[0] == '#':
                continue # empty line or a comment

            elif line[0] == '"': # a new utterance has started
                utt_id = line.split(".")[0]
                utt_id = utt_id[1:]
                labels[utt_id] = list()

            elif line[0] == '.': # the utterance has ended
                continue

            else:
                assert line[0].isdigit() , "Error parsing MLF chunk. Unexpected structure."
                tokens = line.split()
                start = tokens[0]
                start = int(int(start) / 100000)
                end = tokens[1]
                end = int(int(end) / 100000)
                dup = end - start
                u = tokens[2]
                token = self.phone2idx[u], dup
                labels[utt_id].append(token)