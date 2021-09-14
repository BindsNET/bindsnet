import os
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from glob import glob
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class Davis(torch.utils.data.Dataset):
    SUBSET_OPTIONS = ["train", "val", "test-dev", "test-challenge"]
    TASKS = ["semi-supervised", "unsupervised"]
    RESOLUTION_OPTIONS = ["480p", "Full-Resolution"]
    DATASET_WEB = "https://davischallenge.org/davis2017/code.html"
    VOID_LABEL = 255

    def __init__(
        self,
        root,
        task="unsupervised",
        subset="train",
        sequences="all",
        resolution="480p",
        size=(600, 480),
        codalab=False,
        download=False,
        num_samples: int = -1,
    ):
        # language=rst
        """
        Class to read the DAVIS dataset
        :param root: Path to the DAVIS folder that contains JPEGImages, Annotations,
            etc. folders.
        :param task: Task to load the annotations, choose between semi-supervised or
            unsupervised.
        :param subset: Set to load the annotations
        :param sequences: Sequences to consider, 'all' to use all the sequences in a
            set.
        :param resolution: Specify the resolution to use the dataset, choose between
            '480' and 'Full-Resolution'
        :param download: Specify whether to download the dataset if it is not present
        :param num_samples: Number of samples to pass to the batch
        """
        super().__init__()

        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f"Subset should be in {self.SUBSET_OPTIONS}")
        if task not in self.TASKS:
            raise ValueError(f"The only tasks that are supported are {self.TASKS}")
        if resolution not in self.RESOLUTION_OPTIONS:
            raise ValueError(
                f"You may only use one of these resolutions: {self.RESOLUTION_OPTIONS}"
            )

        self.task = task
        self.subset = subset
        self.resolution = resolution
        self.size = size
        self.codalab = codalab

        # Sets the boolean converted if the size of the images must be scaled down
        self.converted = not self.size == (600, 480)

        # Sets a tag for naming the folder containing the dataset
        self.tag = ""
        if self.task == "unsupervised":
            self.tag += "Unsupervised-"
        if self.subset == "train" or self.subset == "val":
            self.tag += "trainval"
        else:
            self.tag += self.subset
        self.tag += "-" + self.resolution

        # Makes a unique path for a given instance of davis
        self.converted_root = os.path.join(
            root, self.tag + "-" + str(self.size[0]) + "x" + str(self.size[1])
        )
        self.root = os.path.join(root, self.tag)
        self.download = download
        self.num_samples = num_samples
        self.zip_path = os.path.join(self.root, "repo.zip")
        self.img_path = os.path.join(self.root, "JPEGImages", resolution)
        annotations_folder = (
            "Annotations" if task == "semi-supervised" else "Annotations_unsupervised"
        )
        self.mask_path = os.path.join(self.root, annotations_folder, resolution)
        year = (
            "2019"
            if task == "unsupervised"
            and (subset == "test-dev" or subset == "test-challenge")
            else "2017"
        )
        self.imagesets_path = os.path.join(self.root, "ImageSets", year)

        # Makes a converted path for scaled images
        if self.converted:
            self.converted_img_path = os.path.join(
                self.converted_root, "JPEGImages", resolution
            )
            self.converted_mask_path = os.path.join(
                self.converted_root, annotations_folder, resolution
            )
            self.converted_imagesets_path = os.path.join(
                self.converted_root, "ImageSets", year
            )

        # Sets seqence_names to the relevant sequences
        if sequences == "all":
            with open(
                os.path.join(self.imagesets_path, f"{self.subset}.txt"), "r"
            ) as f:
                tmp = f.readlines()
            self.sequences_names = [x.strip() for x in tmp]
        else:
            self.sequences_names = (
                sequences if isinstance(sequences, list) else [sequences]
            )
        self.sequences = defaultdict(dict)

        # Check if Davis is installed and download it if necessary
        self._check_directories()

        # Sets the images and masks for each sequence resizing for the given size
        for seq in self.sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, "*.jpg"))).tolist()
            if len(images) == 0 and not self.codalab:
                raise FileNotFoundError(f"Images for sequence {seq} not found.")
            self.sequences[seq]["images"] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, "*.png"))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]["masks"] = masks

        # Creates an enumeration for the sequences for __getitem__
        self.enum_sequences = []
        for seq in self.sequences_names:
            self.enum_sequences.append(self.sequences[seq])

    def __len__(self):
        # language=rst
        """
        Calculates the number of sequences the dataset holds.

        :return: The number of sequences in the dataset.
        """
        return len(self.sequences)

    def _convert_sequences(self):
        # language=rst
        """
        Creates a new root for the dataset to be converted and placed into,
        then copies each image and mask into the given size and stores correctly.
        """
        os.makedirs(os.path.join(self.converted_imagesets_path, f"{self.subset}.txt"))
        os.makedirs(self.converted_img_path)
        os.makedirs(self.converted_mask_path)

        shutil.copy(
            os.path.join(self.imagesets_path, f"{self.subset}.txt"),
            os.path.join(self.converted_imagesets_path, f"{self.subset}.txt"),
        )

        print("Converting sequences to size: {0}".format(self.size))
        for seq in tqdm(self.sequences_names):
            os.makedirs(os.path.join(self.converted_img_path, seq))
            os.makedirs(os.path.join(self.converted_mask_path, seq))
            images = np.sort(glob(os.path.join(self.img_path, seq, "*.jpg"))).tolist()
            if len(images) == 0 and not self.codalab:
                raise FileNotFoundError(f"Images for sequence {seq} not found.")
            for ind, img in enumerate(images):
                im = Image.open(img)
                im.thumbnail(self.size, Image.ANTIALIAS)
                im.save(
                    os.path.join(
                        self.converted_img_path, seq, str(ind).zfill(5) + ".jpg"
                    )
                )
            masks = np.sort(glob(os.path.join(self.mask_path, seq, "*.png"))).tolist()
            for ind, msk in enumerate(masks):
                im = Image.open(msk)
                im.thumbnail(self.size, Image.ANTIALIAS)
                im.convert("RGB").save(
                    os.path.join(
                        self.converted_mask_path, seq, str(ind).zfill(5) + ".png"
                    )
                )

    def _check_directories(self):
        # language=rst
        """
        Verifies that the correct dataset is downloaded; downloads if it isn't and
        ``download=True``.

        :raises: FileNotFoundError if the subset sequence, annotation or root folder is
            missing.
        """
        if not os.path.exists(self.root):
            if self.download:
                self._download()
            else:
                raise FileNotFoundError(
                    "DAVIS not found in the specified directory, download it from "
                    f"{self.DATASET_WEB} or add download=True to your call"
                )
        if not os.path.exists(os.path.join(self.imagesets_path, f"{self.subset}.txt")):
            raise FileNotFoundError(
                f"Subset sequences list for {self.subset} not found, download the "
                f"missing subset for the {self.task} task from {self.DATASET_WEB}"
            )
        if self.subset in ["train", "val"] and not os.path.exists(self.mask_path):
            raise FileNotFoundError(
                f"Annotations folder for the {self.task} task not found, "
                f"download it from {self.DATASET_WEB}"
            )
        if self.converted:
            if not os.path.exists(self.converted_img_path):
                self._convert_sequences()
            self.img_path = self.converted_img_path
            self.mask_path = self.converted_mask_path
            self.imagesets_path = self.converted_imagesets_path

    def get_frames(self, sequence):
        for img, msk in zip(
            self.sequences[sequence]["images"], self.sequences[sequence]["masks"]
        ):
            image = np.array(Image.open(img))
            mask = None if msk is None else np.array(Image.open(msk))
            yield image, mask

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append("".join(obj.split("/")[-1].split(".")[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, "images")

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, "masks")
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = tmp == masks[None, ...]
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq

    def _download(self):
        # language=rst
        """
        Downloads the correct dataset based on the given parameters.

        Relies on ``self.tag`` to determine both the name of the folder created for the
        dataset and for the finding the correct download url.
        """
        os.makedirs(self.root)

        # Grabs the correct zip url based on parameters
        zip_url = f"https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-{self.tag}.zip"

        print("\nDownloading Davis data set from " + zip_url + "\n")

        # Downloads the relevant dataset
        urlretrieve(zip_url, self.zip_path, reporthook=self.progress)

        print("\nDone! \n\nUnzipping and restructuring")

        # Extracts the dataset
        z = zipfile.ZipFile(self.zip_path, "r")
        z.extractall(path=self.root)
        z.close()
        os.remove(self.zip_path)

        temp_folder = os.path.join(self.root, "DAVIS\\")

        # Deletes an unnecessary containing folder "DAVIS" which comes with every download
        for file in os.listdir(temp_folder):
            shutil.move(temp_folder + file, self.root)
        cwd = os.getcwd()
        os.chdir(self.root)
        os.rmdir("DAVIS")
        os.chdir(cwd)

        print("\nDone!\n")

    def __getitem__(self, ind):
        # language=rst
        """
        Gets an item of the ``Dataset`` based on index.

        :param ind: Index of item to take from dataset.
        :return: A sequence which contains a list of images and masks.
        """
        seq = self.enum_sequences[ind]
        return seq

    @staticmethod
    def progress(count, block_size, total_size):
        # language=rst
        """
        Simple progress indicator for the download of the dataset.
        """
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
