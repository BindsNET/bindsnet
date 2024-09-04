import os
import sys
import time
import warnings
import zipfile
from urllib.request import urlretrieve

import cv2
import numpy as np
from torch.utils.data import Dataset

from bindsnet.datasets.preprocess import (
    BoundingBox,
    Rescale,
    bgr2rgb,
    crop_sample,
    cropPadImage,
)

warnings.filterwarnings("ignore")


class ALOV300(Dataset):
    DATASET_WEB = "http://alov300pp.joomlafree.it/dataset-resources.html"
    VOID_LABEL = 255

    def __init__(self, root, transform, input_size, download=False):
        """
        Class to read the ALOV dataset

        :param root: Path to the ALOV folder that contains JPEGImages,
            annotations, etc. folders.
        :param input_size: The input size of network that is using this data,
            for rescaling.
        :param download: Specify whether to download the dataset if it is not
            present.
        :param num_samples: Number of samples to pass to the batch
        """
        super(ALOV300, self).__init__()

        # Makes a unique path for a given instance of davis
        self.root = root
        self.download = download
        self.img_path = os.path.join(self.root, "JPEGImages")
        self.box_path = os.path.join(self.root, "box/")
        self.frame_path = os.path.join(self.root, "frame/")

        # Check if Davis is installed and download it if necessary
        self._check_directories()

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
        self.exclude = [
            # "01-Light_video00016",
            # "01-Light_video00022",
            # "01-Light_video00023",
            # "02-SurfaceCover_video00012",
            # "03-Specularity_video00003",
            # "03-Specularity_video00012",
            # "10-LowContrast_video00013",
        ]

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
                vid_src = f"{self.frame_path}{env}/{vid}"
                vid_ann = f"{self.box_path}{env}/{vid}.ann"
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
        ann = map(lambda x: float(x), ann.strip().split(" "))
        left = min(ann[1], ann[3], ann[5], ann[7])
        top = min(ann[2], ann[4], ann[6], ann[8])
        right = max(ann[1], ann[3], ann[5], ann[7])
        bottom = max(ann[2], ann[4], ann[6], ann[8])
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
        self.frame_zip_path = os.path.join(self.root, "frame.zip")
        self.text_zip_path = os.path.join(self.root, "text.zip")
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
