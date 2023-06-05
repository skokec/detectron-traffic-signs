#!/usr/bin/python

import sys, os

import argparse
import glob

import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

import numpy as np

from caffe2.python import workspace

from core.config import assert_and_infer_cfg
from core.config import cfg
from core.config import merge_cfg_from_file
import core.test_engine as infer_engine
import datasets.dummy_datasets as dummy_datasets
import utils.c2 as c2_utils
from utils.vis import convert_from_cls_format, get_class_string, vis_bbox, vis_class

c2_utils.import_detectron_ops()
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

class TSRDemo:

    def __init__(self, cfg_filename, weights_filename, catalog_folder=None,
                 vis_thresh=0.8, vis_font_scale=0.5, vis_box_thick=2,
                 vis_cls_im_show=True, vis_cls_im_min=30, vis_cls_im_max=150):
        workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])

        merge_cfg_from_file(cfg_filename)
        cfg.TEST.WEIGHTS = weights_filename
        cfg.NUM_GPUS = 1
        assert_and_infer_cfg(cache_urls=False)

        self.model = infer_engine.initialize_model_from_cfg()
        self.dfg200_dataset = dummy_datasets.get_dfg200_dataset()

        # load class images from catalog_folder
        if catalog_folder is not None and os.path.exists(catalog_folder):
            self.cls_image = {}
            for id,cls in self.dfg200_dataset.classes.iteritems():
                self.cls_image[id] = cv2.imread(os.path.join(catalog_folder,'%s.jpg' % cls))

        self.vis_thresh = vis_thresh
        self.vis_font_scale = vis_font_scale
        self.vis_box_thick = vis_box_thick
        self.vis_cls_im_show = vis_cls_im_show
        self.vis_cls_im_min = vis_cls_im_min
        self.vis_cls_im_max = vis_cls_im_max

    def predict(self, im):

        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(self.model, im, None)

        if cls_boxes is not None:
          im = self.visualizeResults(im, cls_boxes)

        return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    def visualizeResults(self, im, boxes):
        """Constructs a numpy array with the detections visualized."""

        if isinstance(boxes, list):
            boxes, segms, keypoints, classes = convert_from_cls_format(boxes, cls_segms=None, cls_keyps=None)

        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.vis_thresh:
            return im

        boxes_wh = np.stack((boxes[:,0],
                             boxes[:,1],
                             boxes[:,2] - boxes[:,0],
                             boxes[:,3] - boxes[:,1]),axis=1)

        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)

        forbidden_plot_regions = self._get_forbidden_plot_regions(boxes_wh, boxes[:, -1], (im.shape[1],im.shape[0]))

        for i in sorted_inds:
            bbox = boxes[i, :4]         # x1,y1,x2,y2
            bbox_wh = boxes_wh[i, :4]   # x,y,w,h
            score = boxes[i, -1]
            if score < self.vis_thresh:
                continue

            # show box (off by default)
            if self.vis_box_thick > 0:
                im = vis_bbox(im, (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), thick=self.vis_box_thick)

            # show class (off by default)
            if self.vis_font_scale > 0:
                class_str = get_class_string(classes[i], score, self.dfg200_dataset)
                im = vis_class(im, (bbox[0], bbox[1] - 2), class_str, font_scale=self.vis_font_scale)

            if self.vis_cls_im_show and classes[i] in self.cls_image:
                # get display image
                cat_img = self.cls_image[classes[i]]

                # get potential plot regions
                plot_candiates = self._get_potential_plot_regions(bbox_wh, (im.shape[1],im.shape[0]),
                                                                  cat_img.shape[1] / float(cat_img.shape[0]),
                                                                  self.vis_cls_im_min,
                                                                  self.vis_cls_im_max)
                plot_candiates = np.array(plot_candiates)
                plot_candiates_intersect = []
                # check how much each plot candidate overlaps with the forbidden_plot_regions
                for c,a in zip(plot_candiates, plot_candiates[:,2]*plot_candiates[:,3]):

                    intersect_area = 0
                    for f in forbidden_plot_regions:
                        r1 = ((c[0]+c[2]/2, c[1]+c[3]/2), (c[2], c[3]), 0)
                        r2 = ((f[0]+f[2]/2, f[1]+f[3]/2), (f[2], f[3]), 0)

                        int_pts = cv2.rotatedRectangleIntersection(r1,r2)
                        if int_pts[0] != 0:
                            intersect_area += cv2.contourArea(int_pts[1])

                    # allow 25% overlap to remove jumpines of windowos due to neighbor overlaps
                    if float(intersect_area)/a < 0.25:
                        intersect_area = 0

                    plot_candiates_intersect.append(intersect_area)

                    # we can stop if found one that does not intersect at all
                    if intersect_area == 0:
                        break

                # select one with the smallest interesection
                best_candidate_index= np.argmin(plot_candiates_intersect)

                plot_region = plot_candiates[best_candidate_index]
                plot_region = map(int,plot_region)

                # resize catalog image to the same size as detection
                resized_cat_img = cv2.resize(cat_img, dsize=(plot_region[2], plot_region[3]))

                cat_img_roi = [0, 0, resized_cat_img.shape[1], resized_cat_img.shape[0]]

                # clip = left (x), right (x), top (y), bottom (y)
                clip = [max(0, 0 - plot_region[0]),
                        max(0, (plot_region[0] + plot_region[2])  - im.shape[1]),
                        max(0, 0 - plot_region[1]),
                        max(0, (plot_region[1] + plot_region[3])  - im.shape[0])]

                # ensure image does not fall out of bounds
                if np.sum(clip) > 0:
                    plot_region = [plot_region[0] + clip[0],  plot_region[1] + clip[2], plot_region[2] - (clip[0] + clip[1]), plot_region[3] - (clip[2] + clip[3])]
                    cat_img_roi = [cat_img_roi[0] + clip[0],  cat_img_roi[1] + clip[2], cat_img_roi[2] - (clip[0] + clip[1]), cat_img_roi[3] - (clip[2] + clip[3])]

                if cat_img_roi[2]*cat_img_roi[3] > 0:

                    # append selected region to forbidden plots (before converting to x1,y1,x2,y2)
                    forbidden_plot_regions = np.concatenate((forbidden_plot_regions,np.array(plot_region).reshape((1,-1))),axis=0)

                    # convert to x1,y1,x2,y2
                    plot_region = [plot_region[0], plot_region[1], plot_region[0] + plot_region[2], plot_region[1] + plot_region[3]]
                    cat_img_roi = [cat_img_roi[0], cat_img_roi[1], cat_img_roi[0] + cat_img_roi[2], cat_img_roi[1] + cat_img_roi[3]]
                    im[plot_region[1]:plot_region[3],plot_region[0]:plot_region[2],:] = resized_cat_img[cat_img_roi[1]:cat_img_roi[3],cat_img_roi[0]:cat_img_roi[2],:]



        return im

    @staticmethod
    def _get_forbidden_plot_regions(boxes_wh, scores, frame_size):
        # add out-of-frame regions as forbidden (but can be used if nothing else is available
        forbidden_plot_regions = []

        forbidden_plot_regions.append((-frame_size[0],-frame_size[1],frame_size[0],frame_size[1]))
        forbidden_plot_regions.append((0,-frame_size[1],frame_size[0],frame_size[1]))
        forbidden_plot_regions.append((frame_size[0],-frame_size[1],frame_size[0],frame_size[1]))
        forbidden_plot_regions.append((-frame_size[0],0,frame_size[0],frame_size[1]))
        #forbidden_plot_regions.append((0,0,frame._size[0],frame._size[1]))
        forbidden_plot_regions.append((frame_size[0],0,frame_size[0],frame_size[1]))
        forbidden_plot_regions.append((-frame_size[0],frame_size[1],frame_size[0],frame_size[1]))
        forbidden_plot_regions.append((0,frame_size[1],frame_size[0],frame_size[1]))
        forbidden_plot_regions.append((frame_size[0],frame_size[1],frame_size[0],frame_size[1]))

        return np.concatenate((boxes_wh, np.array(forbidden_plot_regions)), axis=0)


    @staticmethod
    def _get_potential_plot_regions(bbox_wh, frame_size, required_aspect_ratio, min_size, max_size):
        # bbox = [x,y,w,h]
        candidate_regions = []
        candidate_size = [bbox_wh[2],bbox_wh[3]]

        # change the aspect ratio to maintain the small side
        if candidate_size[0] >= candidate_size[1]:
            candidate_size[0] = candidate_size[1] * required_aspect_ratio
        else:
            candidate_size[1] = candidate_size[0] / required_aspect_ratio

        # require region to have at least 30 px for the larger side
        resize_factor = float(min_size) / float(max(candidate_size[0], candidate_size[1]))

        if resize_factor > 1:
            candidate_size = [int(candidate_size[0] * resize_factor),int(candidate_size[1] * resize_factor)]

        # require region to have max 150 px (in width or height)
        resize_factor = float(max_size) / float(max(candidate_size[0], candidate_size[1]))

        if resize_factor < 1:
            candidate_size = [int(candidate_size[0] * resize_factor), int(candidate_size[1] * resize_factor)]

        if bbox_wh[0] + bbox_wh[2] / 2 < frame_size[0] / 2:
            # LEFT side
            candidate_regions.append((bbox_wh[0]+bbox_wh[2], bbox_wh[1], candidate_size[0], candidate_size[1]))
            candidate_regions.append((bbox_wh[0]-candidate_size[0], bbox_wh[1], candidate_size[0], candidate_size[1]))
            candidate_regions.append((bbox_wh[0], bbox_wh[1]+bbox_wh[3], candidate_size[0], candidate_size[1]))
            candidate_regions.append((bbox_wh[0], bbox_wh[1]-candidate_size[1], candidate_size[0], candidate_size[1]))
        else:
            # RIGHT side
            candidate_regions.append((bbox_wh[0]-candidate_size[0], bbox_wh[1], candidate_size[0], candidate_size[1]))
            candidate_regions.append((bbox_wh[0]+bbox_wh[2], bbox_wh[1], candidate_size[0], candidate_size[1]))
            candidate_regions.append((bbox_wh[0], bbox_wh[1]+bbox_wh[3], candidate_size[0], candidate_size[1]))
            candidate_regions.append((bbox_wh[0], bbox_wh[1]-candidate_size[1], candidate_size[0], candidate_size[1]))

        return candidate_regions


class FolderProcessing:
    def __init__(self, detection_method, folder, img_ext, out_folder=None):
        self.detection_method = detection_method
        self.img_list = glob.iglob(os.path.join(folder,'*.' + img_ext))
        self.img_list = sorted(self.img_list)

        self.out_folder = out_folder

    def run(self):
        import pylab as plt
        for img_filename in self.img_list:

            frame = cv2.imread(img_filename)
            frame = self.detection_method.predict(frame)

            if self.out_folder != None:
                cv2.imwrite(os.path.join(self.out_folder, os.path.basename(img_filename)),frame)
            else:
                plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                plt.show(block=True)

def main(args):

    if args.image_folder is None:
        from echolib_wrapper import EcholibWrapper
        processer = lambda d: EcholibWrapper(d)
    else:
        processer = lambda d: FolderProcessing(d, args.image_folder, args.image_ext, args.out_folder)

    cfg_filename = args.cfg
    weights_filename = args.weights
    catalog_folder = args.catalog

    demo = processer(TSRDemo(cfg_filename, weights_filename, catalog_folder=catalog_folder))

    try:
        demo.run()
    except KeyboardInterrupt:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description='Traffic Sign Detection')

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
        '--cat',
        dest='catalog',
        help='folder to dataset catalog (class template images) ',
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
        '--image-folder',
        dest='image_folder',
        help='folder to images for processing (default: None)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--out-folder',
        dest='out_folder',
        help='folder to store output (default: None)',
        default=None,
        type=str
    )

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    main(args)

