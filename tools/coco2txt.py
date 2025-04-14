import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Convert coco val to quant txt')
    parser.add_argument(
        "-c",
        "--coco",
        type=str,
        default="coco",
        help="Path to your coco dataset val.json.",
    )
    parser.add_argument(
        "-q",
        "--quant",
        type=str,
        default="coco_val.txt",
        help="Path to your output quant txt.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    coco = args.coco
    quant = args.quant
    with open(coco, 'r') as f:
        data = json.load(f)
    with open(quant, 'w') as f:
        for i in data['images']:
            f.write(i['file_name'] + '\n')
    print('Convert coco val to quant txt done.')

if __name__ == '__main__':
    main()
