#!/usr/bin/env python3

"""
Module: main.py
Authors: Christian Bergler
Institution: Friedrich-Alexander-University Erlangen-Nuremberg, Department of Computer Science, Pattern Recognition Lab
Last Access: 06.02.2021
"""

import os
import json
import math
import pathlib
import argparse

import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim

from data.audiodataset import (
    get_audio_files_from_dir,
    get_broken_audio_files,
    DatabaseCsvSplit,
    DefaultSpecDatasetOps,
    Dataset,
)

from trainer import Trainer
from utils.logging import Logger
from collections import OrderedDict

from models.L2Loss import L2Loss
from models.unet_model import UNet

parser = argparse.ArgumentParser()

"""
Convert string to boolean.
"""
def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

parser.add_argument(
    "-d",
    "--debug",
    dest="debug",
    action="store_true",
    help="Log additional training and model information.",
)

""" Directory parameters """
parser.add_argument(
    "--data_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--cache_dir",
    type=str,
    help="The path to the dataset directory.",
)

parser.add_argument(
    "--model_dir",
    type=str,
    help="The directory where the model will be stored.",
)

parser.add_argument(
    "--checkpoint_dir",
    type=str,
    help="The directory where the checkpoints will be stored.",
)

parser.add_argument(
    "--log_dir", type=str, default=None, help="The directory to store the logs."
)

parser.add_argument(
    "--summary_dir",
    type=str,
    help="The directory to store the tensorboard summaries.",
)

parser.add_argument(
    "--noise_dir_train",
    type=str,
    default="",
    help="Path to a directory with noise files for training noise2noise approach using real world noise.",
)

parser.add_argument(
    "--noise_dir_val",
    type=str,
    default="",
    help="Path to a directory with noise files for validation noise2noise approach using real world noise.",
)

parser.add_argument(
    "--noise_dir_test",
    type=str,
    default="",
    help="Path to a directory with noise files for testing noise2noise approach using real world noise.",
)

""" Training parameters """
parser.add_argument(
    "--start_from_scratch",
    dest="start_scratch",
    action="store_true",
    help="Start taining from scratch, i.e. do not use checkpoint to restore.",
)

parser.add_argument(
    "--jit_save",
    dest="jit_save",
    action="store_true",
    help="Save model via torch.jit save functionality.",
)

parser.add_argument(
    "--max_train_epochs", type=int, default=500, help="The number of epochs to train for the classifier."
)

parser.add_argument(
    "--random_val",
    dest="random_val",
    action="store_true",
    help="Select random value intervals for noise2noise and binary mask alternatives also in validation and not only during training.",
)

parser.add_argument(
    "--epochs_per_eval",
    type=int,
    default=2,
    help="The number of batches to run in between evaluations.",
)

parser.add_argument(
    "--batch_size", type=int, default=1, help="The number of images per batch."
)

parser.add_argument(
    "--num_workers", type=int, default=4, help="Number of workers used in data-loading"
)

parser.add_argument(
    "--no_cuda",
    dest="cuda",
    action="store_false",
    help="Do not use cuda to train model.",
)

parser.add_argument(
    "--lr",
    "--learning_rate",
    type=float,
    default=1e-5,
    help="Initial learning rate. Will get multiplied by the batch size.",
)

parser.add_argument(
    "--beta1", type=float, default=0.5, help="beta1 for the adam optimizer."
)

parser.add_argument(
    "--lr_patience_epochs",
    type=int,
    default=8,
    help="Decay the learning rate after N/epochs_per_eval epochs without any improvements on the validation set.",
)

parser.add_argument(
    "--lr_decay_factor",
    type=float,
    default=0.5,
    help="Decay factor to apply to the learning rate.",
)

parser.add_argument(
    "--early_stopping_patience_epochs",
    metavar="N",
    type=int,
    default=20,
    help="Early stopping (stop training) after N/epochs_per_eval epochs without any improvements on the validation set.",
)

""" Input parameters """
parser.add_argument(
    "--filter_broken_audio", action="store_true", help="Filter by a minimum loudness using SoX (Sound exchange) toolkit (option could only be used if SoX is installed)."
)

parser.add_argument(
    "--sequence_len", type=int, default=1280, help="Sequence length in ms."
)

parser.add_argument(
    "--freq_compression",
    type=str,
    default="linear",
    help="Frequency compression to reduce GPU memory usage. "
    "Options: `'linear'` (default), '`mel`', `'mfcc'`",
)

parser.add_argument(
    "--n_freq_bins",
    type=int,
    default=256,
    help="Number of frequency bins after compression.",
)

parser.add_argument(
    "--n_fft",
    type=int,
    default=4096,
    help="FFT size.")

parser.add_argument(
    "--hop_length",
    type=int,
    default=441,
    help="FFT hop length.")

parser.add_argument(
    "--augmentation",
    type=str2bool,
    default=True,
    help="Whether to augment the input data. "
    "Validation and test data will not be augmented.",
)


ARGS = parser.parse_args()
ARGS.cuda = torch.cuda.is_available() and ARGS.cuda
ARGS.device = torch.device("cuda") if ARGS.cuda else torch.device("cpu")

log = Logger("TRAIN", ARGS.debug, ARGS.log_dir)

"""
Get audio all audio files from the given data directory except they are broken.
"""
def get_audio_files():
    audio_files = None
    if input_data.can_load_from_csv():
        log.info("Found csv files in {}".format(ARGS.data_dir))
    else:
        log.debug("Searching for audio files in {}".format(ARGS.data_dir))
        if ARGS.filter_broken_audio:
            data_dir_ = pathlib.Path(ARGS.data_dir)
            audio_files = get_audio_files_from_dir(ARGS.data_dir)
            log.debug("Moving possibly broken audio files to .bkp:")
            broken_files = get_broken_audio_files(audio_files, ARGS.data_dir)
            for f in broken_files:
                log.debug(f)
                bkp_dir = data_dir_.joinpath(f).parent.joinpath(".bkp")
                bkp_dir.mkdir(exist_ok=True)
                f = pathlib.Path(f)
                data_dir_.joinpath(f).rename(bkp_dir.joinpath(f.name))
        audio_files = list(get_audio_files_from_dir(ARGS.data_dir))
        log.info("Found {} audio files for training.".format(len(audio_files)))
        if len(audio_files) == 0:
            log.close()
            exit(1)
    return audio_files

"""
Save the trained model and corresponding options either via torch.jit and/or torch.save.
"""
def save_model(unet, dataOpts, path, model, use_jit=False):
    unet = unet.cpu()
    unet_state_dict = unet.state_dict()
    save_dict = {
        "unetState": unet_state_dict,
        "dataOpts": dataOpts,
    }
    if not os.path.isdir(ARGS.model_dir):
        os.makedirs(ARGS.model_dir)
    if use_jit:
        example = torch.rand(1, 1, 128, 256)
        extra_files = {}
        extra_files['dataOpts'] = dataOpts.__str__()
        model = torch.jit.trace(model, example)
        torch.jit.save(model, path, _extra_files=extra_files)
        log.debug("Model successfully saved via torch jit: " + str(path))
    else:
        torch.save(save_dict, path)
        log.debug("Model successfully saved via torch save: " + str(path))


"""
Main function to compute data preprocessing, network training, evaluation, and saving.
"""
if __name__ == "__main__":

    dataOpts = DefaultSpecDatasetOps

    for arg, value in vars(ARGS).items():
        if arg in dataOpts and value is not None:
            dataOpts[arg] = value

    ARGS.lr *= ARGS.batch_size

    patience_lr = math.ceil(ARGS.lr_patience_epochs / ARGS.epochs_per_eval)

    patience_lr = int(max(1, patience_lr))

    log.debug("dataOpts: " + json.dumps(dataOpts, indent=4))

    sequence_len = int(
        float(ARGS.sequence_len) / 1000 * dataOpts["sr"] / dataOpts["hop_length"]
    )
    log.debug("Training with sequence length: {}".format(sequence_len))
    input_shape = (ARGS.batch_size, 1, dataOpts["n_freq_bins"], sequence_len)

    log.info("Setting up model")

    unet = UNet(n_channels=1, n_classes=1, bilinear=False)

    log.debug("Model: " + str(unet))
    model = nn.Sequential(OrderedDict([("unet", unet)]))

    split_fracs = {"train": .7, "val": .15, "test": .15}
    input_data = DatabaseCsvSplit(
        split_fracs, working_dir=ARGS.data_dir, split_per_dir=True
    )

    audio_files = get_audio_files()

    noise_files_train = [str(p) for p in pathlib.Path(ARGS.noise_dir_train).glob("*.wav")]
    noise_files_val = [str(p) for p in pathlib.Path(ARGS.noise_dir_val).glob("*.wav")]
    noise_files_test = [str(p) for p in pathlib.Path(ARGS.noise_dir_test).glob("*.wav")]

    random_val = ARGS.random_val

    datasets = {
        split: Dataset(
            file_names=input_data.load(split, audio_files),
            working_dir=ARGS.data_dir,
            cache_dir=ARGS.cache_dir,
            sr=dataOpts["sr"],
            n_fft=dataOpts["n_fft"],
            hop_length=dataOpts["hop_length"],
            n_freq_bins=dataOpts["n_freq_bins"],
            freq_compression=dataOpts["freq_compression"],
            f_min=dataOpts["fmin"],
            f_max=dataOpts["fmax"],
            seq_len=sequence_len,
            augmentation=ARGS.augmentation if split == "train" else False,
            noise_files_train=noise_files_train,
            noise_files_val=noise_files_val if split == "val" else False,
            noise_files_test=noise_files_test if split == "test" else False,
            random=True if split == "train" or (split == "val" and random_val) else False,
            dataset_name=split,
        )
        for split in split_fracs.keys()
    }

    dataloaders = {
        split: torch.utils.data.DataLoader(
            datasets[split],
            batch_size=ARGS.batch_size,
            shuffle=True,
            num_workers=ARGS.num_workers,
            drop_last=False if split == "val" or split == "test" else True,
            pin_memory=True,
        )
        for split in split_fracs.keys()
    }

    trainer = Trainer(
        model=model,
        logger=log,
        prefix="denoiser",
        checkpoint_dir=ARGS.checkpoint_dir,
        summary_dir=ARGS.summary_dir,
        n_summaries=4,
        start_scratch=ARGS.start_scratch,
    )

    optimizer = optim.Adam(
        model.parameters(), lr=ARGS.lr, betas=(ARGS.beta1, 0.999)
    )

    metric_mode = "min"
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=metric_mode,
        patience=patience_lr,
        factor=ARGS.lr_decay_factor,
        threshold=1e-3,
        threshold_mode="abs",
    )

    L2Loss = L2Loss(reduction="sum")

    model = trainer.fit(
        dataloaders["train"],
        dataloaders["val"],
        dataloaders["test"],
        loss_fn=L2Loss,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        n_epochs=ARGS.max_train_epochs,
        val_interval=ARGS.epochs_per_eval,
        patience_early_stopping=ARGS.early_stopping_patience_epochs,
        device=ARGS.device,
        val_metric="loss"
    )

    path = os.path.join(ARGS.model_dir, "orca-clean.pk")

    save_model(unet, dataOpts, path, model, use_jit=ARGS.jit_save)

    log.close()
