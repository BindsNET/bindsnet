import math
import random

import cv2
import numpy as np
import torch
from torchvision import transforms


def gray_scale(image: np.ndarray) -> np.ndarray:
    # language=rst
    """
    Converts RGB image into grayscale.

    :param image: RGB image.
    :return: Gray-scaled image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def crop(image: np.ndarray, x1: int, x2: int, y1: int, y2: int) -> np.ndarray:
    # language=rst
    """
    Crops an image given coordinates of cropping box.

    :param image: 3-dimensional image.
    :param x1: Left x coordinate.
    :param x2: Right x coordinate.
    :param y1: Bottom y coordinate.
    :param y2: Top y coordinate.
    :return: Image cropped using coordinates (x1, x2, y1, y2).
    """
    return image[x1:x2, y1:y2, :]


def binary_image(image: np.ndarray) -> np.ndarray:
    # language=rst
    """
    Converts input image into black and white (binary)

    :param image: Gray-scaled image.
    :return: Black and white image.
    """
    return cv2.threshold(image, 0, 1, cv2.THRESH_BINARY)[1]


def subsample(image: np.ndarray, x: int, y: int) -> np.ndarray:
    # language=rst
    """
    Scale the image to (x, y).

    :param image: Image to be rescaled.
    :param x: Output value for ``image``'s x dimension.
    :param y: Output value for ``image``'s y dimension.
    :return: Re-scaled image.
    """
    return cv2.resize(image, (x, y))


""" Below is the implementation of necessary preprocessing for the dataset ALOV300 """


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


# Copyright (c) 2018 Abhinav Moudgil
