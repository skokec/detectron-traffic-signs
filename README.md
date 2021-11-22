# Detectron for Traffic Signs

This branch contains fixes for the Detectron code that allows aplication on domains with many small objects, specifically it was designed for traffic sign detection from the [*"Deep Learning for Large-Scale Traffic-Sign Detection and Recognition"* ITS 2019](https://prints.vicos.si/publications/369) journal paper.

The following changes are included:
- integrated Online Hard Negative Minning ([OHEM](https://arxiv.org/abs/1604.03540))
- modifed selection of training ROIs to cover small and large regions evenly
- added weights to loss of background samples during training (weight of 0.01 for RPN and 0.1 for classification)

YAML definition files of detectron models for the [DFG-dataset](https://www.vicos.si/Downloads/DFGTSD) are available below:
 - [detectron-model-yaml-DFG-dataset.zip](https://box.vicos.si/skokec/villard/detectron-model-yaml-DFG-dataset.zip)
 - [detectron-model-yaml-DFG-dataset-augmented.zip](https://box.vicos.si/skokec/villard/detectron-model-yaml-DFG-dataset-augmented.zip)

Each zip contains models based on ResNet101_FPN and ResNet50_FPN that have enabled OHEM (`OHEM: True`), even selection of small and large ROIs (`RPN_EVENLY_SELECT_POS_ROIS: True`) and weighting of pos/neg classes (`RPN_SIZE_WEIGHTED_LOSS: True` and `CLS_SIZE_WEIGHTED_LOSS: True`).

You can download weights trained on DFG-Dataset for the upper model in:
 - [dfg_mask_rcnn_weights.tar.gz](https://box.vicos.si/skokec/villard/dfg_mask_rcnn_weights.tar.gz)

Note: Due to slight update of the DFG dataset the resulty may vary from the ITS 2019 paper

## Installation

Our changes to the code require custom `WeightedSigmoidCrossEntropyLoss` operation (for `RPN_SIZE_WEIGHTED_LOSS` and `CLS_SIZE_WEIGHTED_LOSS` options) which is implemented in the `DETECTRON_PATH/caffe2-modules/` folder. All files in `DETECTRON_PATH/caffe2-modules/*` need to be coppied into your caffe2 source (`CAFFE2_SRC_PATH/modules/detectron`).

```bash
export DETECTRON_PATH=/path/to/detectron
export CAFFE2_SRC_PATH=/path/to/caffe2_source

cp $DETECTRON_PATH/caffe2-modules/* $CAFFE2_SRC_PATH/modules/detectron
```

After copying `caffe2-modules/*` caffe2 proceed with the instalation instructions for caffe2 and Detectron in [`INSTALL.md`](INSTALL.md)

## Citation

Please cite our [ITS 2019 paper](https://prints.vicos.si/publications/369) when using modifications for Detectron from this repository or for the [DFG dataset](https://www.vicos.si/Downloads/DFGTSD):

```
 @article{Tabernik2019ITS,
    author = {Tabernik, Domen and Sko{\v{c}}aj, Danijel},
    journal = {IEEE Transactions on Intelligent Transportation Systems},
    title = {{Deep Learning for Large-Scale Traffic-Sign Detection and Recognition}},
    year = {2019},
    doi={10.1109/TITS.2019.2913588}, 
    ISSN={1524-9050}
 }
```

# Detectron

Detectron is Facebook AI Research's software system that implements state-of-the-art object detection algorithms, including [Mask R-CNN](https://arxiv.org/abs/1703.06870). It is written in Python and powered by the [Caffe2](https://github.com/caffe2/caffe2) deep learning framework.

At FAIR, Detectron has enabled numerous research projects, including: [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144), [Mask R-CNN](https://arxiv.org/abs/1703.06870), [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333), [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002), [Non-local Neural Networks](https://arxiv.org/abs/1711.07971), [Learning to Segment Every Thing](https://arxiv.org/abs/1711.10370), and [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440).

<div align="center">
  <img src="demo/output/33823288584_1d21cf0a26_k_example_output.jpg" width="700px" />
  <p>Example Mask R-CNN output.</p>
</div>

## Introduction

The goal of Detectron is to provide a high-quality, high-performance
codebase for object detection *research*. It is designed to be flexible in order
to support rapid implementation and evaluation of novel research. Detectron
includes implementations of the following object detection algorithms:

- [Mask R-CNN](https://arxiv.org/abs/1703.06870) -- *Marr Prize at ICCV 2017*
- [RetinaNet](https://arxiv.org/abs/1708.02002) -- *Best Student Paper Award at ICCV 2017*
- [Faster R-CNN](https://arxiv.org/abs/1506.01497)
- [RPN](https://arxiv.org/abs/1506.01497)
- [Fast R-CNN](https://arxiv.org/abs/1504.08083)
- [R-FCN](https://arxiv.org/abs/1605.06409)

using the following backbone network architectures:

- [ResNeXt{50,101,152}](https://arxiv.org/abs/1611.05431)
- [ResNet{50,101,152}](https://arxiv.org/abs/1512.03385)
- [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) (with ResNet/ResNeXt)
- [VGG16](https://arxiv.org/abs/1409.1556)

Additional backbone architectures may be easily implemented. For more details about these models, please see [References](#references) below.

## License

Detectron is released under the [Apache 2.0 license](https://github.com/facebookresearch/detectron/blob/master/LICENSE). See the [NOTICE](https://github.com/facebookresearch/detectron/blob/master/NOTICE) file for additional details.

## Citing Detectron

If you use Detectron in your research or wish to refer to the baseline results published in the [Model Zoo](MODEL_ZOO.md), please use the following BibTeX entry.

```
@misc{Detectron2018,
  author =       {Ross Girshick and Ilija Radosavovic and Georgia Gkioxari and
                  Piotr Doll\'{a}r and Kaiming He},
  title =        {Detectron},
  howpublished = {\url{https://github.com/facebookresearch/detectron}},
  year =         {2018}
}
```

## Model Zoo and Baselines

We provide a large set of baseline results and trained models available for download in the [Detectron Model Zoo](MODEL_ZOO.md).

## Installation

Please find installation instructions for Caffe2 and Detectron in [`INSTALL.md`](INSTALL.md).

## Quick Start: Using Detectron

After installation, please see [`GETTING_STARTED.md`](GETTING_STARTED.md) for brief tutorials covering inference and training with Detectron.

## Getting Help

To start, please check the [troubleshooting](INSTALL.md#troubleshooting) section of our installation instructions as well as our [FAQ](FAQ.md). If you couldn't find help there, try searching our GitHub issues. We intend the issues page to be a forum in which the community collectively troubleshoots problems.

If bugs are found, **we appreciate pull requests** (including adding Q&A's to `FAQ.md` and improving our installation instructions and troubleshooting documents). Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information about contributing to Detectron.

## References

- [Data Distillation: Towards Omni-Supervised Learning](https://arxiv.org/abs/1712.04440).
  Ilija Radosavovic, Piotr Dollár, Ross Girshick, Georgia Gkioxari, and Kaiming He.
  Tech report, arXiv, Dec. 2017.
- [Learning to Segment Every Thing](https://arxiv.org/abs/1711.10370).
  Ronghang Hu, Piotr Dollár, Kaiming He, Trevor Darrell, and Ross Girshick.
  Tech report, arXiv, Nov. 2017.
- [Non-Local Neural Networks](https://arxiv.org/abs/1711.07971).
  Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
  Tech report, arXiv, Nov. 2017.
- [Mask R-CNN](https://arxiv.org/abs/1703.06870).
  Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
  Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár.
  IEEE International Conference on Computer Vision (ICCV), 2017.
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677).
  Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He.
  Tech report, arXiv, June 2017.
- [Detecting and Recognizing Human-Object Interactions](https://arxiv.org/abs/1704.07333).
  Georgia Gkioxari, Ross Girshick, Piotr Dollár, and Kaiming He.
  Tech report, arXiv, Apr. 2017.
- [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144).
  Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431).
  Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.
- [R-FCN: Object Detection via Region-based Fully Convolutional Networks](http://arxiv.org/abs/1605.06409).
  Jifeng Dai, Yi Li, Kaiming He, and Jian Sun.
  Conference on Neural Information Processing Systems (NIPS), 2016.
- [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](http://arxiv.org/abs/1506.01497)
  Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun.
  Conference on Neural Information Processing Systems (NIPS), 2015.
- [Fast R-CNN](http://arxiv.org/abs/1504.08083).
  Ross Girshick.
  IEEE International Conference on Computer Vision (ICCV), 2015.
