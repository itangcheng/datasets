# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
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

# Lint as: python3
"""Imagenet datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

import tensorflow.compat.v2 as tf
from tensorflow_datasets.image.imagenet import Imagenet2012
import tensorflow_datasets.public_api as tfds


_DESCRIPTION = '''\
Imagenet1pBalanced is a subset of original ILSVRC 2012 dataset, where only ~1%,
or 12811, images are sampled in labeled balanced fashion. This is supposed to
be used as a benchmark for semi-supervised learning, and has been originally
used in SimCLR paper (https://arxiv.org/abs/2002.05709).
'''

_CITATION = '''\
@article{chen2020simple,
  title={A Simple Framework for Contrastive Learning of Visual Representations},
  author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
  journal={arXiv preprint arXiv:2002.05709},
  year={2020}
}
@article{ILSVRC15,
Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
Title = {{ImageNet Large Scale Visual Recognition Challenge}},
Year = {2015},
journal   = {International Journal of Computer Vision (IJCV)},
doi = {10.1007/s11263-015-0816-y},
volume={115},
number={3},
pages={211-252}
}
'''

SUBSET_FILE = 'https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/1percent.txt'


class Imagenet1pBalanced(Imagenet2012):
  """1% (class balanced) subset of Imagenet 2012 dataset."""

  VERSION = tfds.core.Version(
      '1.0.0', 'Class balanced 1% ImageNet training dataset.')

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  manual_dir should contain three files: ILSVRC2012_img_train.tar,
  ILSVRC2012_img_val.tar, and subset specification file.
  You need to register on http://www.image-net.org/download-images in order
  to get the link to download the first two files (train and val).
  The subset specification file can be downloaded here:
  https://raw.githubusercontent.com/google-research/simclr/master/imagenet_subsets/1percent.txt
  """

  def _split_generators(self, dl_manager):
    train_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2012_img_train.tar')
    val_path = os.path.join(dl_manager.manual_dir, 'ILSVRC2012_img_val.tar')
    train_subset_path = os.path.join(dl_manager.manual_dir, '1percent.txt')

    # We don't import the original test split, as it doesn't include labels.
    # These were never publicly released.
    if not tf.io.gfile.exists(train_path) or not tf.io.gfile.exists(val_path):
      raise AssertionError(
          'ImageNet requires manual download of the data. Please download '
          'the train and val set and place them into: {}, {}'.format(
              train_path, val_path))

    # Load the filenames of sampled subset.
    if not tf.io.gfile.exists(train_path) or not tf.io.gfile.exists(val_path):
      raise AssertionError(
          'Subset specification file not found. Please download '
          'it from {} and place it into {}'.format(
              SUBSET_FILE, train_subset_path))
    with open(train_subset_path) as fp:
      subset = set(fp.read().split('\n'))

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'archive': dl_manager.iter_archive(train_path),
                'subset': subset,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'archive': dl_manager.iter_archive(val_path),
                'validation_labels': self._get_validation_labels(val_path),
            },
        ),
    ]

  def _generate_examples(self, archive, subset=None, validation_labels=None):
    """Yields examples."""
    if validation_labels:  # Validation split
      for key, example in self._generate_examples_validation(archive,
                                                             validation_labels):
        yield key, example
    # Training split. Main archive contains archives names after a synset noun.
    # Each sub-archive contains pictures associated to that synset.
    for fname, fobj in archive:
      label = fname[:-4]  # fname is something like 'n01632458.tar'
      # TODO(b/117643231): in py3, the following lines trigger tarfile module
      # to call `fobj.seekable()`, which Gfile doesn't have. We should find an
      # alternative, as this loads ~150MB in RAM.
      fobj_mem = io.BytesIO(fobj.read())
      for image_fname, image in tfds.download.iter_archive(
          fobj_mem, tfds.download.ExtractMethod.TAR_STREAM):
        image = self._fix_image(image_fname, image)
        if subset is None or image_fname in subset:  # filtering using subset.
          record = {
              'file_name': image_fname,
              'image': image,
              'label': label,
          }
          yield image_fname, record
