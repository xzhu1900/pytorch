import argparse
import os
import warnings

from htk_dataset import HTKDataset
from htk_converter import HTKConverter

parser = argparse.ArgumentParser(description='HTK format converter')

parser.add_argument('-c', '--chunk-size', default=1000, type=int)
parser.add_argument('-b', '--big-endian', default=True, type=bool)
parser.add_argument('-f', '--mlf-format', default="debug-binary", type=str) #text, binary, debug-binary
parser.add_argument('-m', '--mlf', default="/home/xuzhu/data/htkTest/smallData/glob_0000.mlflist", type=str)
parser.add_argument('-s', '--scp', default="/home/xuzhu/data/htkTest/smallData/glob_0000.scp", type=str)
parser.add_argument('-l', '--statelist', default="/home/xuzhu/data/htkTest/smallData/state.list", type=str)
parser.add_argument('-o', '--output-dir', default="/home/xuzhu/data/htkTest/smallData/", type=str)
parser.add_argument('-v', '--version', default=1, type=int)


def main():
    global args
    args = parser.parse_args()
    dataset = HTKDataset(args.mlf, args.scp, args.statelist)

    converter = HTKConverter(dataset)
    converter.writeBinary(args.big_endian, args.chunk_size, args.output_dir, args.version, args.mlf_format)
    #converter.writeFeatureData(args.output_dir, 16, 80)
    #converter.writeLabelData(args.output_dir, args.mlf_format)

if __name__ == '__main__':
    main()
