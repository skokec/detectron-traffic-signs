#!/usr/bin/env python2

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
from utils.timer import Timer
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
import utils.logging
import utils.vis as vis_utils
import numpy as np

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: NONE - does not output images)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-detection-file',
        dest='output_detection_file',
        help='file for outputing detections (default: NONE - does not output detections)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text


def main(args):
    logger = logging.getLogger(__name__)
    merge_cfg_from_file(args.cfg)
    cfg.TEST.WEIGHTS = args.weights
    cfg.NUM_GPUS = 1
    assert_and_infer_cfg(cache_urls=False)
    model = infer_engine.initialize_model_from_cfg()
    dummy_dfg200_dataset = dummy_datasets.get_dfg200_dataset()

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    im_list = sorted(im_list)

    fn = None
    if args.output_detection_file is not None and len(args.output_detection_file) > 0:
        fn = open(args.output_detection_file, 'w')

    for i, im_name in enumerate(im_list):
        logger.info('Processing {}'.format(im_name))
        im = cv2.imread(im_name)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )

        if fn is not None :
            if isinstance(cls_boxes, list):
                boxes, segms, keypoints, classes = vis_utils.convert_from_cls_format(
                    cls_boxes, cls_segms, cls_keyps)

                fn.write(os.path.basename(im_name))
                fn.write('\n')

                if boxes is not None:
                    for i, box in enumerate(boxes):
                        bbox = box[:4]
                        score = box[-1]

                        fn.write(get_class_string(classes[i], score, dummy_dfg200_dataset)+' '+str(score)+' ')
                        fn.write(str(bbox[0])+' '+str(bbox[1])+' '+str(bbox[2]-bbox[0])+' '+str(bbox[3]-bbox[1]))
                        fn.write('\n')
                fn.flush()

        if args.output_dir is not None and len(args.output_dir) > 0:
            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                os.path.basename(im_name),
                args.output_dir,
                cls_boxes,
                cls_segms,
                cls_keyps,
                dataset=dummy_dfg200_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.4,
                kp_thresh=2,
                ext='png'
            )
    if fn is not None:
        fn.close()

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    utils.logging.setup_logging(__name__)
    args = parse_args()
    main(args)
