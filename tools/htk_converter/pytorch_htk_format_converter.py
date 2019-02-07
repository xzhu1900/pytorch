import argparse
import os
import warnings

from htk_dataset import HTKDataset
from htk_converter import HTKConverter

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Converter to convert exiting CNTK based HTK format to a chunk-based foramt,
# which presents with better performance paired with the pytorch htk chunk reader.
parser = argparse.ArgumentParser(description='HTK format converter')

parser.add_argument('-c', '--chunk-size', default=1000, type=int)
parser.add_argument('-b', '--output-big-endian', type=str2bool, nargs='?', const=True)
parser.add_argument('-i', '--input-big-endian', type=str2bool, nargs='?', const=False)
parser.add_argument('-f', '--mlf-format', default="binary", type=str) #text, binary
parser.add_argument('-m', '--mlf', default="/home/xuzhu/data/htkTest/smallData/glob_0000.mlflist", type=str)
parser.add_argument('-s', '--scp', default="/home/xuzhu/data/htkTest/smallData/glob_0000.scp", type=str)
parser.add_argument('-l', '--statelist', default="/home/xuzhu/data/htkTest/smallData/state.list", type=str)
parser.add_argument('-o', '--output-dir', default="/home/xuzhu/data/htkTest/smallData/", type=str)
parser.add_argument('-v', '--version', default=1, type=int)

def main():
    global args
    args = parser.parse_args()
    dataset = HTKDataset(args.mlf, args.scp, args.statelist, args.mlf_format, args.input_big_endian)

    converter = HTKConverter(dataset)
    converter.writeChunkFiles(args.output_big_endian, args.chunk_size, args.output_dir, args.version)
    #converter.writeFeatureData(args.output_dir, 16, 80)
    #converter.writeLabelData(args.output_dir)

if __name__ == '__main__':
    main()