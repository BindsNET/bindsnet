# necessary imports
import os
import sys
import time
import argparse

import torch
import torch.optim as optim
import numpy as np
from bindsnet.models import GoNet

# from torchsummary import summary

from bindsnet.datasets import ALOV300
from bindsnet.datasets.alov300 import (
    Rescale,
    shift_crop_training_sample,
    crop_sample,
    NormalizeToTensor,
)

# constants
cuda = torch.cuda.is_available()
device = "cuda" if cuda else "cpu"
device = "cpu"
input_size = 224
kSaveModel = 20000  # save model after every 20000 steps
batchSize = 50  # number of samples in a batch
kGeneratedExamplesPerImage = 10  # generate 10 synthetic samples per image
transform = NormalizeToTensor()
bb_params = {}
enable_tensorboard = False
if enable_tensorboard:
    from tensorboardX import SummaryWriter

    writer = SummaryWriter()

args = None
parser = argparse.ArgumentParser(description="GOTURN Training")
parser.add_argument(
    "-n",
    "--num-batches",
    default=500000,
    type=int,
    help="number of total batches to run",
)
parser.add_argument(
    "-lr", "--learning-rate", default=1e-5, type=float, help="initial learning rate"
)
parser.add_argument(
    "--gamma", default=0.1, type=float, help="learning rate decay factor"
)
parser.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum")
parser.add_argument(
    "--weight_decay", default=0.0005, type=float, help="weight decay in optimizer"
)
parser.add_argument(
    "--lr-decay-step",
    default=100000,
    type=int,
    help="number of steps after which learning rate decays",
)
parser.add_argument(
    "-d",
    "--data-directory",
    type=str,
    default="../../data/ALOV300",
    help="path to data directory",
)
parser.add_argument(
    "-s",
    "--save-directory",
    type=str,
    default="../saved_checkpoints/goturn/",
    help="path to save directory",
)
parser.add_argument(
    "-lshift",
    "--lambda-shift-frac",
    default=5,
    type=float,
    help="lambda-shift for random cropping",
)
parser.add_argument(
    "-lscale",
    "--lambda-scale-frac",
    default=15,
    type=float,
    help="lambda-scale for random cropping",
)
parser.add_argument(
    "-minsc",
    "--min-scale",
    default=-0.4,
    type=float,
    help="min-scale for random cropping",
)
parser.add_argument(
    "-maxsc",
    "--max-scale",
    default=0.4,
    type=float,
    help="max-scale for random cropping",
)
parser.add_argument(
    "-seed", "--manual-seed", default=800, type=int, help="set manual seed value"
)
parser.add_argument(
    "--resume",
    default="../saved_checkpoints/goturn/model_itr_120000_loss_34.559.pth.tar",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=50,
    type=int,
    help="number of samples in batch (default: 50)",
)
parser.add_argument(
    "--save-freq",
    default=20000,
    type=int,
    help="save checkpoint frequency (default: 20000)",
)


def main():

    global args, batchSize, kSaveModel, bb_params
    args = parser.parse_args()
    print(args)
    batchSize = args.batch_size
    kSaveModel = args.save_freq
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    if cuda:
        torch.cuda.manual_seed_all(args.manual_seed)

    # load bounding box motion model params
    bb_params["lambda_shift_frac"] = args.lambda_shift_frac
    bb_params["lambda_scale_frac"] = args.lambda_scale_frac
    bb_params["min_scale"] = args.min_scale
    bb_params["max_scale"] = args.max_scale

    # load datasets
    alov = ALOV300(
        root=args.data_directory,
        download=True,
        transform=transform,
        input_size=input_size,
    )
    # imagenet = ILSVRC2014_DET_Dataset(os.path.join(args.data_directory,
    #                                   'ILSVRC2014_DET_train/'),
    #                                   os.path.join(args.data_directory,
    #                                   'ILSVRC2014_DET_bbox_train/'),
    #                                   bb_params,
    #                                   transform,
    #                                   input_size)
    # list of datasets to train on
    datasets = [alov]  # removed imagenet from training

    # load model
    net = GoNet()
    # summary(net, [(3, 224, 224), (3, 224, 224)])
    loss_fn = torch.nn.L1Loss(size_average=False)

    # initialize optimizer
    optimizer = optim.SGD(
        net.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    if os.path.exists(args.save_directory):
        print("Directory %s already exists" % (args.save_directory))
    else:
        os.makedirs(args.save_directory)

    # start training
    net = train_model(net, datasets, loss_fn, optimizer)

    # save trained model
    checkpoint = {"state_dict": net.state_dict()}
    path = os.path.join(args.save_directory, "pytorch_goturn.pth.tar")
    torch.save(checkpoint, path)


def get_training_batch(num_running_batch, running_batch, dataset):
    """
    Implements GOTURN batch formation regimen.
    """
    global args, batchSize
    done = False
    N = kGeneratedExamplesPerImage + 1
    train_batch = None
    x1_batch, x2_batch, y_batch = make_transformed_samples(dataset, args)
    assert x1_batch.shape[0] == x2_batch.shape[0] == y_batch.shape[0] == N
    count_in = min(batchSize - num_running_batch, N)
    remain = N - count_in
    running_batch["previmg"][
        num_running_batch : num_running_batch + count_in
    ] = x1_batch[:count_in]
    running_batch["currimg"][
        num_running_batch : num_running_batch + count_in
    ] = x2_batch[:count_in]
    running_batch["currbb"][num_running_batch : num_running_batch + count_in] = y_batch[
        :count_in
    ]
    num_running_batch = num_running_batch + count_in
    if remain > 0:
        done = True
        train_batch = running_batch.copy()
        running_batch["previmg"][:remain] = x1_batch[-remain:]
        running_batch["currimg"][:remain] = x2_batch[-remain:]
        running_batch["currbb"][:remain] = y_batch[-remain:]
        num_running_batch = remain
    return running_batch, train_batch, done, num_running_batch


def make_transformed_samples(dataset, args):
    """
    Given a dataset, it picks a random sample from it and returns a batch
    of (kGeneratedExamplesPerImage+1) samples. The batch contains true sample
    from dataset and kGeneratedExamplesPerImage samples, which are created
    artifically with augmentation by GOTURN smooth motion model.
    """
    idx = np.random.randint(dataset.len, size=1)[0]
    # unscaled original sample (single image and bb)
    orig_sample = dataset.get_orig_sample(idx)
    # cropped scaled sample (two frames and bb)
    true_sample, _ = dataset.get_sample(idx)
    true_tensor = transform(true_sample)
    x1_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 3, input_size, input_size)
    x2_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 3, input_size, input_size)
    y_batch = torch.Tensor(kGeneratedExamplesPerImage + 1, 4)

    # initialize batch with the true sample
    x1_batch[0] = true_tensor["previmg"]
    x2_batch[0] = true_tensor["currimg"]
    y_batch[0] = true_tensor["currbb"]

    scale = Rescale((input_size, input_size))
    for i in range(kGeneratedExamplesPerImage):
        sample = orig_sample
        # unscaled current image crop with box
        curr_sample, opts_curr = shift_crop_training_sample(sample, bb_params)
        # unscaled previous image crop with box
        prev_sample, opts_prev = crop_sample(sample)
        scaled_curr_obj = scale(curr_sample, opts_curr)
        scaled_prev_obj = scale(prev_sample, opts_prev)
        training_sample = {
            "previmg": scaled_prev_obj["image"],
            "currimg": scaled_curr_obj["image"],
            "currbb": scaled_curr_obj["bb"],
        }
        sample = transform(training_sample)
        x1_batch[i + 1] = sample["previmg"]
        x2_batch[i + 1] = sample["currimg"]
        y_batch[i + 1] = sample["currbb"]

    return x1_batch, x2_batch, y_batch


def train_model(model, datasets, criterion, optimizer):

    global args, writer
    since = time.time()
    curr_loss = 0
    lr = args.learning_rate
    flag = False
    start_itr = 0
    num_running_batch = 0
    running_batch = {
        "previmg": torch.Tensor(batchSize, 3, input_size, input_size),
        "currimg": torch.Tensor(batchSize, 3, input_size, input_size),
        "currbb": torch.Tensor(batchSize, 4),
    }
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.gamma
    )

    # resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_itr = checkpoint["itr"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            num_running_batch = checkpoint["num_running_batch"]
            running_batch = checkpoint["running_batch"]
            lr = checkpoint["lr"]
            np.random.set_state(checkpoint["np_rand_state"])
            torch.set_rng_state(checkpoint["torch_rand_state"])
            print(
                "=> loaded checkpoint '{}' (iteration {})".format(
                    args.resume, checkpoint["itr"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if not os.path.isdir(args.save_directory):
        os.makedirs(args.save_directory)

    itr = start_itr
    st = time.time()
    while itr < args.num_batches:

        model.train()
        if (
            args.resume
            and os.path.isfile(args.resume)
            and itr == start_itr
            and (not flag)
        ):
            checkpoint = torch.load(args.resume)
            i = checkpoint["dataset_indx"]
            flag = True
        else:
            i = 0

        # train on datasets
        # usually ALOV and ImageNet
        while i < len(datasets):
            dataset = datasets[i]
            i = i + 1
            (running_batch, train_batch, done, num_running_batch) = get_training_batch(
                num_running_batch, running_batch, dataset
            )
            # print(i, num_running_batch, done)
            if done:
                scheduler.step()
                # load sample
                x1 = train_batch["previmg"].to(device)
                x2 = train_batch["currimg"].to(device)
                y = train_batch["currbb"].requires_grad_(False).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                output = model(x1, x2)
                loss = criterion(output, y)

                # backward + optimize
                loss.backward()
                optimizer.step()

                # statistics
                curr_loss = loss.item()
                end = time.time()
                itr = itr + 1
                print(
                    "[training] step = %d/%d, loss = %f, time = %f"
                    % (itr, args.num_batches, curr_loss, end - st)
                )
                sys.stdout.flush()
                del train_batch
                st = time.time()

                if enable_tensorboard:
                    writer.add_scalar("train/batch_loss", curr_loss, itr)

                if itr > 0 and itr % kSaveModel == 0:
                    path = os.path.join(
                        args.save_directory,
                        "model_itr_"
                        + str(itr)
                        + "_loss_"
                        + str(round(curr_loss, 3))
                        + ".pth.tar",
                    )
                    save_checkpoint(
                        {
                            "itr": itr,
                            "np_rand_state": np.random.get_state(),
                            "torch_rand_state": torch.get_rng_state(),
                            "l1_loss": curr_loss,
                            "state_dict": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "num_running_batch": num_running_batch,
                            "running_batch": running_batch,
                            "lr": lr,
                            "dataset_indx": i,
                        },
                        path,
                    )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    if enable_tensorboard:
        writer.export_scalars_to_json("./all_scalars.json")
        writer.close()
    return model


def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)


if __name__ == "__main__":
    main()
