from __future__ import print_function, division

import os
import numpy as np
import torch
import shutil
import zipfile
import sys
import time
import warnings
import cv2
import random
import math

from PIL import Image
from glob import glob
from tqdm import tqdm
from collections import defaultdict
from typing import Optional, Tuple, List, Iterable
from urllib.request import urlretrieve
from torchvision import transforms
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

class ALOV300(torch.utils.data.Dataset):
    SUBSET_OPTIONS = ["train", "val", "test-dev", "test-challenge"]
    TASKS = ["semi-supervised", "unsupervised"]
    RESOLUTION_OPTIONS = ["480p", "Full-Resolution"]
    DATASET_WEB = "https://davischallenge.org/davis2017/code.html"
    VOID_LABEL = 255

    def __init__(
        self,
        root,
        transform,
        input_size,
        codalab=False,
        download=False,
        num_samples: int = -1,
    ):
        """
        Class to read the DAVIS dataset
        
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations, etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        :param resolution: Specify the resolution to use the dataset, choose between '480' and 'Full-Resolution'
        :param download: Specify whether to download the dataset if it is not present
        :param num_samples: Number of samples to pass to the batch
        """
        super(ALOV300, self).__init__()

        # Makes a unique path for a given instance of davis
        self.root = root
        self.download = download
        self.num_samples = num_samples
        self.frame_zip_path = os.path.join(self.root, "frame.zip")
        self.text_zip_path = os.path.join(self.root, "text.zip")
        self.img_path = os.path.join(self.root, "JPEGImages")
        self.box_path = os.path.join(self.root, "box/")
        self.frame_path = os.path.join(self.root, "frame/")

        # Check if Davis is installed and download it if necessary
        self._check_directories()

        self.exclude = [
            "01-Light_video00016",
            "01-Light_video00022",
            "01-Light_video00023",
            "02-SurfaceCover_video00012",
            "03-Specularity_video00003",
            "03-Specularity_video00012",
            "10-LowContrast_video00013",
        ]
        self.input_size = input_size
        self.transform = transform
        self.x, self.y = self._parse_data(self.frame_path, self.box_path)
        self.len = len(self.y)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sample, _ = self.get_sample(idx)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _parse_data(self, root, target_dir):
        """
        Parses ALOV dataset and builds tuples of (template, search region)
        tuples from consecutive annotated frames.
        """
        x = []
        y = []
        envs = os.listdir(target_dir)
        num_anno = 0
        print("Parsing ALOV dataset...")
        for env in envs:
            env_videos = os.listdir(root + env)
            for vid in env_videos:
                if vid in self.exclude:
                    continue
                vid_src = self.frame_path + env + "/" + vid
                vid_ann = self.box_path + env + "/" + vid + ".ann"
                frames = os.listdir(vid_src)
                frames.sort()
                frames = [vid_src + "/" + frame for frame in frames]
                f = open(vid_ann, "r")
                annotations = f.readlines()
                f.close()
                frame_idxs = [int(ann.split(" ")[0]) - 1 for ann in annotations]
                frames = np.array(frames)
                num_anno += len(annotations)
                for i in range(len(frame_idxs) - 1):
                    idx = frame_idxs[i]
                    next_idx = frame_idxs[i + 1]
                    x.append([frames[idx], frames[next_idx]])
                    y.append([annotations[i], annotations[i + 1]])
        x = np.array(x)
        y = np.array(y)
        self.len = len(y)
        print("ALOV dataset parsing done.")
        print("Total number of annotations in ALOV dataset = %d" % num_anno)
        return x, y

    def get_sample(self, idx):
        """
        Returns sample without transformation for visualization.

        Sample consists of resized previous and current frame with target
        which is passed to the network. Bounding box values are normalized
        between 0 and 1 with respect to the target frame and then scaled by
        factor of 10.
        """
        opts_curr = {}
        curr_sample = {}
        curr_img = self.get_orig_sample(idx, 1)["image"]
        currbb = self.get_orig_sample(idx, 1)["bb"]
        prevbb = self.get_orig_sample(idx, 0)["bb"]
        bbox_curr_shift = BoundingBox(prevbb[0], prevbb[1], prevbb[2], prevbb[3])
        (
            rand_search_region,
            rand_search_location,
            edge_spacing_x,
            edge_spacing_y,
        ) = cropPadImage(bbox_curr_shift, curr_img)
        bbox_curr_gt = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
        bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
        bbox_gt_recentered = bbox_curr_gt.recenter(
            rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered
        )
        curr_sample["image"] = rand_search_region
        curr_sample["bb"] = bbox_gt_recentered.get_bb_list()

        # additional options for visualization
        opts_curr["edge_spacing_x"] = edge_spacing_x
        opts_curr["edge_spacing_y"] = edge_spacing_y
        opts_curr["search_location"] = rand_search_location
        opts_curr["search_region"] = rand_search_region

        # build prev sample
        prev_sample = self.get_orig_sample(idx, 0)
        prev_sample, opts_prev = crop_sample(prev_sample)

        # scale
        scale = Rescale((self.input_size, self.input_size))
        scaled_curr_obj = scale(curr_sample, opts_curr)
        scaled_prev_obj = scale(prev_sample, opts_prev)
        training_sample = {
            "previmg": scaled_prev_obj["image"],
            "currimg": scaled_curr_obj["image"],
            "currbb": scaled_curr_obj["bb"],
        }
        return training_sample, opts_curr

    def get_orig_sample(self, idx, i=1):
        """
        Returns original image with bounding box at a specific index.
        Range of valid index: [0, self.len-1].
        """
        curr = cv2.imread(self.x[idx][i])
        curr = bgr2rgb(curr)
        currbb = self.get_bb(self.y[idx][i])
        sample = {"image": curr, "bb": currbb}
        return sample

    def get_bb(self, ann):
        """
        Parses ALOV annotation and returns bounding box in the format:
        [left, upper, width, height]
        """
        ann = ann.strip().split(" ")
        left = min(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        top = min(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))
        right = max(float(ann[1]), float(ann[3]), float(ann[5]), float(ann[7]))
        bottom = max(float(ann[2]), float(ann[4]), float(ann[6]), float(ann[8]))
        return [left, top, right, bottom]

    def show(self, idx, is_current=1):
        """
        Helper function to display image at a particular index with grounttruth
        bounding box.

        Arguments:
            idx: index
            is_current: 0 for previous frame and 1 for current frame
        """
        sample = self.get_orig_sample(idx, is_current)
        image = sample["image"]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        bb = sample["bb"]
        bb = [int(val) for val in bb]
        image = cv2.rectangle(image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
        cv2.imshow("alov dataset sample: " + str(idx), image)
        cv2.waitKey(0)

    def show_sample(self, idx):
        """
        Helper function to display sample, which is passed to GOTURN.
        Shows previous frame and current frame with bounding box.
        """
        x, _ = self.get_sample(idx)
        prev_image = x["previmg"]
        curr_image = x["currimg"]
        bb = x["currbb"]
        bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
        bbox.unscale(curr_image)
        bb = bbox.get_bb_list()
        bb = [int(val) for val in bb]
        prev_image = cv2.cvtColor(prev_image, cv2.COLOR_RGB2BGR)
        curr_image = cv2.cvtColor(curr_image, cv2.COLOR_RGB2BGR)
        curr_image = cv2.rectangle(
            curr_image, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2
        )
        concat_image = np.hstack((prev_image, curr_image))
        cv2.imshow("alov dataset sample: " + str(idx), concat_image)
        cv2.waitKey(0)

    def _check_directories(self):
        """
        Verifies that the correct dataset is downloaded; downloads if it isn't and download=True.

        :raises: FileNotFoundError if the subset sequence, annotation or root folder is missing.
        """
        if not os.path.exists(self.root):
            if self.download:
                self._download()
            else:
                raise FileNotFoundError(
                    f"ALOV300 not found in the specified directory, download it from {self.DATASET_WEB} or add download=True to your call"
                )
        if not os.path.exists(self.frame_path):
            raise FileNotFoundError(
                f"Frames not found, check the directory: {self.root}"
            )
        if not os.path.exists(self.box_path):
            raise FileNotFoundError(
                f"Boxes not found, check the directory: {self.root}"
            )

    def _download(self):
        """
        Downloads the correct dataset based on the given parameters

        Relies on self.tag to determine both the name of the folder created for the dataset and for the finding the correct download url. 
        """

        os.makedirs(self.root)

        # Grabs the correct zip url based on parameters
        frame_zip_url = f"http://isis-data.science.uva.nl/alov/alov300++_frames.zip"
        text_zip_url = f"http://isis-data.science.uva.nl/alov/alov300++GT_txtFiles.zip"

        # Downloads the relevant dataset
        print("\nDownloading ALOV300++ frame set from " + frame_zip_url + "\n")
        urlretrieve(frame_zip_url, self.frame_zip_path, reporthook=self.progress)

        print("\nDownloading ALOV300++ text set from " + text_zip_url + "\n")
        urlretrieve(text_zip_url, self.text_zip_path, reporthook=self.progress)

        print("\nDone! \n\nUnzipping and restructuring")

        # Extracts the dataset
        z = zipfile.ZipFile(self.frame_zip_path, "r")
        z.extractall(path=self.root)
        z.close()
        os.remove(self.frame_zip_path)

        z = zipfile.ZipFile(self.text_zip_path, "r")
        z.extractall(path=self.root)
        z.close()
        os.remove(self.text_zip_path)

        # Renames the folders containing the dataset
        box_folder = os.path.join(self.root, "alov300++_rectangleAnnotation_full/")
        frame_folder = os.path.join(self.root, "imagedata++")

        os.rename(box_folder, self.box_path)
        os.rename(frame_folder, self.frame_path)

    # Simple progress indicator for the download of the dataset
    def progress(self, count, block_size, total_size):
        global start_time
        if count == 0:
            start_time = time.time()
            return
        duration = time.time() - start_time
        progress_size = int(count * block_size)
        speed = int(progress_size / (1024 * duration))
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write(
            "\r...%d%%, %d MB, %d KB/s, %d seconds passed"
            % (percent, progress_size / (1024 * 1024), speed, duration)
        )
        sys.stdout.flush()


# Copyright (c) 2018 Abhinav Moudgil


""" Below is the implementation of necessary preprocessing for the dataset """

class Rescale(object):
    """Rescale image and bounding box.
    Args:
        output_size (tuple or int): Desired output size. If int, square crop
        is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample, opts):
        image, bb = sample["image"], sample["bb"]
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        # make sure that gray image has 3 channels
        img = cv2.resize(image, (new_h, new_w), interpolation=cv2.INTER_CUBIC)
        bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
        bbox.scale(opts["search_region"])
        return {"image": img, "bb": bbox.get_bb_list()}


class NormalizeToTensor(object):
    """Returns torch tensor normalized images."""

    def __call__(self, sample):
        prev_img, curr_img = sample["previmg"], sample["currimg"]
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        prev_img = self.transform(prev_img)
        curr_img = self.transform(curr_img)
        if "currbb" in sample:
            currbb = np.array(sample["currbb"])
            return {
                "previmg": prev_img,
                "currimg": curr_img,
                "currbb": torch.from_numpy(currbb).float(),
            }
        else:
            return {"previmg": prev_img, "currimg": curr_img}


def bgr2rgb(image):
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def shift_crop_training_sample(sample, bb_params):
    """
    Given an image with bounding box, this method randomly shifts the box and
    generates a training example. It returns current image crop with shifted
    box (with respect to current image).
    """
    output_sample = {}
    opts = {}
    currimg = sample["image"]
    currbb = sample["bb"]
    bbox_curr_gt = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
    bbox_curr_shift = BoundingBox(0, 0, 0, 0)
    bbox_curr_shift = bbox_curr_gt.shift(
        currimg,
        bb_params["lambda_scale_frac"],
        bb_params["lambda_shift_frac"],
        bb_params["min_scale"],
        bb_params["max_scale"],
        True,
        bbox_curr_shift,
    )
    (
        rand_search_region,
        rand_search_location,
        edge_spacing_x,
        edge_spacing_y,
    ) = cropPadImage(bbox_curr_shift, currimg)
    bbox_curr_gt = BoundingBox(currbb[0], currbb[1], currbb[2], currbb[3])
    bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
    bbox_gt_recentered = bbox_curr_gt.recenter(
        rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered
    )
    output_sample["image"] = rand_search_region
    output_sample["bb"] = bbox_gt_recentered.get_bb_list()

    # additional options for visualization

    opts["edge_spacing_x"] = edge_spacing_x
    opts["edge_spacing_y"] = edge_spacing_y
    opts["search_location"] = rand_search_location
    opts["search_region"] = rand_search_region
    return output_sample, opts


def crop_sample(sample):
    """
    Given a sample image with bounding box, this method returns the image crop
    at the bounding box location with twice the width and height for context.
    """
    output_sample = {}
    opts = {}
    image, bb = sample["image"], sample["bb"]
    orig_bbox = BoundingBox(bb[0], bb[1], bb[2], bb[3])
    (output_image, pad_image_location, edge_spacing_x, edge_spacing_y) = cropPadImage(
        orig_bbox, image
    )
    new_bbox = BoundingBox(0, 0, 0, 0)
    new_bbox = new_bbox.recenter(
        pad_image_location, edge_spacing_x, edge_spacing_y, new_bbox
    )
    output_sample["image"] = output_image
    output_sample["bb"] = new_bbox.get_bb_list()

    # additional options for visualization
    opts["edge_spacing_x"] = edge_spacing_x
    opts["edge_spacing_y"] = edge_spacing_y
    opts["search_location"] = pad_image_location
    opts["search_region"] = output_image
    return output_sample, opts


def cropPadImage(bbox_tight, image):
    pad_image_location = computeCropPadImageLocation(bbox_tight, image)
    roi_left = min(pad_image_location.x1, (image.shape[1] - 1))
    roi_bottom = min(pad_image_location.y1, (image.shape[0] - 1))
    roi_width = min(
        image.shape[1],
        max(1.0, math.ceil(pad_image_location.x2 - pad_image_location.x1)),
    )
    roi_height = min(
        image.shape[0],
        max(1.0, math.ceil(pad_image_location.y2 - pad_image_location.y1)),
    )

    err = 0.000000001  # To take care of floating point arithmetic errors
    cropped_image = image[
        int(roi_bottom + err) : int(roi_bottom + roi_height),
        int(roi_left + err) : int(roi_left + roi_width),
    ]
    output_width = max(math.ceil(bbox_tight.compute_output_width()), roi_width)
    output_height = max(math.ceil(bbox_tight.compute_output_height()), roi_height)
    if image.ndim > 2:
        output_image = np.zeros(
            (int(output_height), int(output_width), image.shape[2]), dtype=image.dtype
        )
    else:
        output_image = np.zeros(
            (int(output_height), int(output_width)), dtype=image.dtype
        )

    edge_spacing_x = min(bbox_tight.edge_spacing_x(), (image.shape[1] - 1))
    edge_spacing_y = min(bbox_tight.edge_spacing_y(), (image.shape[0] - 1))

    # rounding should be done to match the width and height
    output_image[
        int(edge_spacing_y) : int(edge_spacing_y) + cropped_image.shape[0],
        int(edge_spacing_x) : int(edge_spacing_x) + cropped_image.shape[1],
    ] = cropped_image
    return output_image, pad_image_location, edge_spacing_x, edge_spacing_y


def computeCropPadImageLocation(bbox_tight, image):
    # Center of the bounding box
    bbox_center_x = bbox_tight.get_center_x()
    bbox_center_y = bbox_tight.get_center_y()

    image_height = image.shape[0]
    image_width = image.shape[1]

    # Padded output width and height
    output_width = bbox_tight.compute_output_width()
    output_height = bbox_tight.compute_output_height()

    roi_left = max(0.0, bbox_center_x - (output_width / 2.0))
    roi_bottom = max(0.0, bbox_center_y - (output_height / 2.0))

    # Padded roi width
    left_half = min(output_width / 2.0, bbox_center_x)
    right_half = min(output_width / 2.0, image_width - bbox_center_x)
    roi_width = max(1.0, left_half + right_half)

    # Padded roi height
    top_half = min(output_height / 2.0, bbox_center_y)
    bottom_half = min(output_height / 2.0, image_height - bbox_center_y)
    roi_height = max(1.0, top_half + bottom_half)

    # Padded image location in the original image
    objPadImageLocation = BoundingBox(
        roi_left, roi_bottom, roi_left + roi_width, roi_bottom + roi_height
    )
    return objPadImageLocation


def sample_rand_uniform():
    RAND_MAX = 2147483647
    return (random.randint(0, RAND_MAX) + 1) * 1.0 / (RAND_MAX + 2)


def sample_exp_two_sides(lambda_):
    RAND_MAX = 2147483647
    pos_or_neg = random.randint(0, RAND_MAX)
    if (pos_or_neg % 2) == 0:
        pos_or_neg = 1
    else:
        pos_or_neg = -1

    rand_uniform = sample_rand_uniform()
    return math.log(rand_uniform) / (lambda_ * pos_or_neg)


""" implementation of bounding box class for cropping images """


class BoundingBox:
    def __init__(self, x1, y1, x2, y2):

        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.kContextFactor = 2
        self.kScaleFactor = 10

    def print_bb(self):
        print("------Bounding-box-------")
        print("(x1, y1): ({}, {})".format(self.x1, self.y1))
        print("(x2, y2): ({}, {})".format(self.x2, self.y2))
        print("(w, h)  : ({}, {})".format(self.x2 - self.x1 + 1, self.y2 - self.y1 + 1))
        print("--------------------------")

    def get_bb_list(self):
        return [self.x1, self.y1, self.x2, self.y2]

    def get_center_x(self):
        return (self.x1 + self.x2) / 2.0

    def get_center_y(self):
        return (self.y1 + self.y2) / 2.0

    def compute_output_height(self):
        bbox_height = self.y2 - self.y1
        output_height = self.kContextFactor * bbox_height

        return max(1.0, output_height)

    def compute_output_width(self):
        bbox_width = self.x2 - self.x1
        output_width = self.kContextFactor * bbox_width

        return max(1.0, output_width)

    def edge_spacing_x(self):
        output_width = self.compute_output_width()
        bbox_center_x = self.get_center_x()

        return max(0.0, (output_width / 2) - bbox_center_x)

    def edge_spacing_y(self):
        output_height = self.compute_output_height()
        bbox_center_y = self.get_center_y()

        return max(0.0, (output_height / 2) - bbox_center_y)

    def unscale(self, image):
        height = image.shape[0]
        width = image.shape[1]

        self.x1 = self.x1 / self.kScaleFactor
        self.x2 = self.x2 / self.kScaleFactor
        self.y1 = self.y1 / self.kScaleFactor
        self.y2 = self.y2 / self.kScaleFactor

        self.x1 = self.x1 * width
        self.x2 = self.x2 * width
        self.y1 = self.y1 * height
        self.y2 = self.y2 * height

    def uncenter(self, raw_image, search_location, edge_spacing_x, edge_spacing_y):
        self.x1 = max(0.0, self.x1 + search_location.x1 - edge_spacing_x)
        self.y1 = max(0.0, self.y1 + search_location.y1 - edge_spacing_y)
        self.x2 = min(raw_image.shape[1], self.x2 + search_location.x1 - edge_spacing_x)
        self.y2 = min(raw_image.shape[0], self.y2 + search_location.y1 - edge_spacing_y)

    def recenter(self, search_loc, edge_spacing_x, edge_spacing_y, bbox_gt_recentered):
        bbox_gt_recentered.x1 = self.x1 - search_loc.x1 + edge_spacing_x
        bbox_gt_recentered.y1 = self.y1 - search_loc.y1 + edge_spacing_y
        bbox_gt_recentered.x2 = self.x2 - search_loc.x1 + edge_spacing_x
        bbox_gt_recentered.y2 = self.y2 - search_loc.y1 + edge_spacing_y

        return bbox_gt_recentered

    def scale(self, image):
        height = image.shape[0]
        width = image.shape[1]

        self.x1 = self.x1 / width
        self.y1 = self.y1 / height
        self.x2 = self.x2 / width
        self.y2 = self.y2 / height

        self.x1 = self.x1 * self.kScaleFactor
        self.y1 = self.y1 * self.kScaleFactor
        self.x2 = self.x2 * self.kScaleFactor
        self.y2 = self.y2 * self.kScaleFactor

    def get_width(self):
        return self.x2 - self.x1

    def get_height(self):
        return self.y2 - self.y1

    def shift(
        self,
        image,
        lambda_scale_frac,
        lambda_shift_frac,
        min_scale,
        max_scale,
        shift_motion_model,
        bbox_rand,
    ):
        width = self.get_width()
        height = self.get_height()

        center_x = self.get_center_x()
        center_y = self.get_center_y()

        kMaxNumTries = 10

        new_width = -1
        num_tries_width = 0
        while ((new_width < 0) or (new_width > image.shape[1] - 1)) and (
            num_tries_width < kMaxNumTries
        ):
            if shift_motion_model:
                width_scale_factor = max(
                    min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac))
                )
            else:
                rand_num = sample_rand_uniform()
                width_scale_factor = rand_num * (max_scale - min_scale) + min_scale

            new_width = width * (1 + width_scale_factor)
            new_width = max(1.0, min((image.shape[1] - 1), new_width))
            num_tries_width = num_tries_width + 1

        new_height = -1
        num_tries_height = 0
        while ((new_height < 0) or (new_height > image.shape[0] - 1)) and (
            num_tries_height < kMaxNumTries
        ):
            if shift_motion_model:
                height_scale_factor = max(
                    min_scale, min(max_scale, sample_exp_two_sides(lambda_scale_frac))
                )
            else:
                rand_num = sample_rand_uniform()
                height_scale_factor = rand_num * (max_scale - min_scale) + min_scale

            new_height = height * (1 + height_scale_factor)
            new_height = max(1.0, min((image.shape[0] - 1), new_height))
            num_tries_height = num_tries_height + 1

        first_time_x = True
        new_center_x = -1
        num_tries_x = 0

        while (
            first_time_x
            or (new_center_x < center_x - width * self.kContextFactor / 2)
            or (new_center_x > center_x + width * self.kContextFactor / 2)
            or ((new_center_x - new_width / 2) < 0)
            or ((new_center_x + new_width / 2) > image.shape[1])
        ) and (num_tries_x < kMaxNumTries):

            if shift_motion_model:
                new_x_temp = center_x + width * sample_exp_two_sides(lambda_shift_frac)
            else:
                rand_num = sample_rand_uniform()
                new_x_temp = center_x + rand_num * (2 * new_width) - new_width

            new_center_x = min(
                image.shape[1] - new_width / 2, max(new_width / 2, new_x_temp)
            )
            first_time_x = False
            num_tries_x = num_tries_x + 1

        first_time_y = True
        new_center_y = -1
        num_tries_y = 0

        while (
            first_time_y
            or (new_center_y < center_y - height * self.kContextFactor / 2)
            or (new_center_y > center_y + height * self.kContextFactor / 2)
            or ((new_center_y - new_height / 2) < 0)
            or ((new_center_y + new_height / 2) > image.shape[0])
        ) and (num_tries_y < kMaxNumTries):

            if shift_motion_model:
                new_y_temp = center_y + height * sample_exp_two_sides(lambda_shift_frac)
            else:
                rand_num = sample_rand_uniform()
                new_y_temp = center_y + rand_num * (2 * new_height) - new_height

            new_center_y = min(
                image.shape[0] - new_height / 2, max(new_height / 2, new_y_temp)
            )
            first_time_y = False
            num_tries_y = num_tries_y + 1

        bbox_rand.x1 = new_center_x - new_width / 2
        bbox_rand.x2 = new_center_x + new_width / 2
        bbox_rand.y1 = new_center_y - new_height / 2
        bbox_rand.y2 = new_center_y + new_height / 2

        return bbox_rand
