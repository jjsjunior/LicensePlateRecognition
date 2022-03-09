import argparse
import sys
from glob import glob
import xml.etree.ElementTree as ET
import os
import numpy as np
import shutil
import json
import cv2
from PIL import Image
import char_segmentation_evaluation
from darkflow.net.build import TFNet


def main(args):
    dir_images_validation_input = args.dir_images_validation_input
    dir_gt_validation_input = args.dir_gt_validation_input
    dir_images_validation_output = args.dir_images_validation_output
    if not os.path.exists(dir_images_validation_output):
        os.makedirs(dir_images_validation_output)
    # options = {"pbLoad": "yolo-character.pb", "metaLoad": "yolo-character.meta", "gpu": 0.9}
    # options = {"pbLoad": "yolo-character_ceia_4.pb", "metaLoad": "yolo-character_ceia_4.meta", "gpu": 0.9}
    # options = {"pbLoad": "yolo-character_ceia.pb", "metaLoad": "yolo-character_ceia.meta", "gpu": 0.9}
    # options = {"pbLoad": "yolo-character_ceia_3.pb", "metaLoad": "yolo-character_ceia_3.meta", "gpu": 0.9}
    # options = {"pbLoad": "yolo-character_ceia_5.pb", "metaLoad": "yolo-character_ceia_5.meta", "gpu": 0.9}
    # options = {"pbLoad": "yolo-character_ceia_8.pb", "metaLoad": "yolo-character_ceia_7.meta", "gpu": 0.9}


    options = {"pbLoad": "yolo-character_ceia_6.pb", "metaLoad": "yolo-character_ceia_6.meta", "gpu": 0.9}
    yolo_char_seg_model = TFNet(options)
    char_segmentation_evaluation.validate_char_segmentation_model(dir_images_validation_input, dir_gt_validation_input,
                                                                  yolo_char_seg_model, dir_images_validation_output,
                                                                  is_print_output=True)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_images_validation_input', type=str, default='1')
    parser.add_argument('--dir_gt_validation_input', type=str, default='0')
    parser.add_argument('--dir_images_validation_output', type=str, default='1')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))