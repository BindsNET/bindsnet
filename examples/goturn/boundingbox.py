# from helper import sample_exp_two_sides, sample_rand_uniform
import random
import math


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
