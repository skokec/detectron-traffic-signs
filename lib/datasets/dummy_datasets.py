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
"""Provide stub objects that can act as stand-in "dummy" datasets for simple use
cases, like getting all classes in a dataset. This exists so that demos can be
run without requiring users to download/install datasets first.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils.collections import AttrDict


def get_coco_dataset():
    """A dummy COCO dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds


def get_dfg200_dataset():
    """A dummy DFG200 dataset that includes only the 'classes' field."""
    ds = AttrDict()
    classes = [
        '__background__', 'I-1','I-1.1','I-10','I-11','I-13','I-13.1','I-14',
        'I-15','I-16','I-17','I-18','I-19','I-2','I-2.1','I-20','I-25','I-27',
        'I-28','I-28.1','I-29','I-29.1','I-3','I-30','I-32','I-34','I-36',
        'I-37','I-38','I-39-1','I-39-2','I-39-3','I-4','I-5','I-5.1','I-5.2',
        'I-8','I-9','II-1','II-10.1','II-14','II-17','II-18','II-19-4','II-2',
        'II-21','II-22','II-23','II-26','II-26.1','II-28','II-3','II-30-10',
        'II-30-30','II-30-40','II-30-50','II-30-60','II-30-70','II-32','II-33',
        'II-34','II-35','II-39','II-4','II-40','II-41','II-42','II-42.1','II-43',
        'II-45','II-45.1','II-45.2','II-46','II-46.1','II-46.2','II-47','II-47.1',
        'II-48','II-6','II-7','II-7.1','II-8','III-1','III-10','III-105',
        'III-105.1','III-105.3','III-107-1','III-107-2','III-107.1-1',
        'III-107.1-2','III-107.2-1','III-107.2-2','III-112','III-113','III-12',
        'III-120','III-120-1','III-120.1','III-123','III-124','III-14','III-14.1',
        'III-15','III-16','III-18-40','III-18-50','III-18-60','III-18-70','III-2',
        'III-202-5','III-203-2','III-206-1','III-21','III-23','III-25','III-25.1',
        'III-27','III-29-30','III-29-40','III-3','III-30-30','III-33','III-34',
        'III-35','III-37','III-39','III-40','III-42','III-43','III-45','III-46',
        'III-47','III-5','III-50','III-54','III-59','III-6','III-64','III-68',
        'III-74','III-77','III-78','III-8-1','III-84','III-84-1','III-85-2',
        'III-85-3','III-85.1','III-86-1','III-86-2','III-87','III-90','III-90.1',
        'III-90.2','III-91','IV-1','IV-1.1','IV-10','IV-11','IV-12','IV-12.1',
        'IV-13-1','IV-13-2','IV-13-3','IV-13-4','IV-13-5','IV-13-6','IV-13.1-2',
        'IV-13.1-3','IV-13.1-4','IV-16','IV-17','IV-18','IV-2','IV-20-1',
        'IV-3-1','IV-3-2','IV-3-4','IV-3-5','IV-5','IV-6','VI-2.1','VI-3-1',
        'VI-3-2','VI-3.1-1','VI-3.1-2','VI-8','VII-4','VII-4-1','VII-4-2',
        'VII-4.1-1','VII-4.3','VII-4.3-1','VII-4.3-2','VII-4.4-1','VII-4.4-2',
        'X-1.1','X-1.2','X-4','X-6-3'
    ]
    ds.classes = {i: name for i, name in enumerate(classes)}
    return ds