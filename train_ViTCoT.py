## IMPORTS

import sys
sys.path.append("/home/lpandey/Baby_Research/")


# Handle warnings
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)


# LIBRARIES
from argparse import ArgumentParser
import wandb

from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.metrics import Accuracy
from torchmetrics import Accuracy
# Pytorch modules
import torch
import torch.nn
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
from vit_pytorch import ViT
from transformers import ViTConfig

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from datamodules import ImagePairsDataModule

# model
from models.vit_contrastive import Backbone, ViTConfigExtended, ViTConfig, configuration, LitClassifier

import torchvision.transforms as transforms

# custom iamge transforms
from datamodules.transforms import CenterCropLongDimension


## FLAGS ------

def create_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Max number of epochs to train."
    )
    parser.add_argument(
        "--val_split",
        default=0.15,
        type=float,
        help="Percent (float) of samples to use for the validation split."
    )
    #The action set to store_true will store the argument as True , if present
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporally ordered image pairs."
    )
    parser.add_argument(
        "--window_size",
        default=3,
        type=int,
        help="Size of sliding window for sampling temporally ordered image pairs."
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--seed_val",
        type=int,
        default=0,
        help="SEED VALUE"
    )
    parser.add_argument(
        "--shuffle_frames",
        action="store_true",
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--shuffle_temporalWindows",
        action="store_true",
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--dataloader_shuffle",
        action="store_true",
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--head",
        type=int,
        choices=[1,3,6,9,12],
        default=1,
        help="number of attention heads"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="BATCH SIZE"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="",
        help="dataset directory"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="NUM OF WORKERS"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="num of gpus to use"
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=-1,
        help="num of training samples to use from dataset. -1 = entire dataset"
    )
    parser.add_argument(
        "--transforms",
        type=str,
        choices=['transform_gB', 'transform_grayScale', 'transform_randomCrop', 
                 'transform_randomHFlip', 'transform_cj', 'transform_all', 'transform_resize', 
                 'transform_none', 'transform_cropped_resize'],
        default='None',
        help="data augmentation transform"
    )
    parser.add_argument(
        "--resize_dims",
        type=int,
        choices=[64,224],
        default=64,
        help="resize the image to a desired resolution if transform_resize is selected"
    )
    parser.add_argument(
        "--loss_ver",
        type=str,
        choices=['v0','v1'],
        default='v0',
        help="select btw CLTT loss version 0 and loss version 1. Same objectives but different implementations"
    )


    return parser


def cli_main():

    parser = create_argparser()

    # model args
    args = parser.parse_args()
    args.lars_wrapper = True


    # set seed value
    pl.seed_everything(args.seed_val)
    torch.manual_seed(args.seed_val)


    # assign heads and hidden layers 
    # currently, heads and hidden_layers are same for stability.
    configuration.num_attention_heads = args.head
    configuration.num_hidden_layers = args.head

    print("[INFO] Number of ATTENTION HEADS :: ", configuration.num_attention_heads)
    print("[INFO] Number of HIDDEN LAYERS :: ", configuration.num_hidden_layers)
    print("[INFO] Number of GPUs in use :: ", args.gpus)
    print("[INFO] Temporal Window Size :: ", args.window_size)
    

    # setup model 
    backbone = Backbone('vit', configuration)
    model = LitClassifier(backbone=backbone, window_size=args.window_size, loss_ver=args.loss_ver)

    # experimental
    if args.transforms == 'transform_gB':
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
        ], p=0.5),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_grayScale':
        trans = transforms.Compose([
        transforms.RandomGrayscale(p=0.2),
        #transforms.Resize((args.resize_dims, args.resize_dims)),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_randomCrop':
        trans = transforms.Compose([
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_randomHFlip':
        trans = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_cj':
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.ColorJitter(brightness=[0.19999999999999996, 1.8], contrast=[0.19999999999999996, 1.8], saturation=[0.19999999999999996, 1.8], hue=[-0.2, 0.2])
        ], p=0.8),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_all':
        # gb
        trans = transforms.Compose([
        transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 2.0))
        ], p=0.5),
        # grayscale 
        transforms.RandomGrayscale(p=0.2),
        # crop
        transforms.RandomResizedCrop(size=(64, 64), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=transforms.InterpolationMode.BILINEAR),
        # hflip
        transforms.RandomHorizontalFlip(p=0.5),
        # cj
        transforms.RandomApply([
        transforms.ColorJitter(brightness=[0.19999999999999996, 1.8], contrast=[0.19999999999999996, 1.8], saturation=[0.19999999999999996, 1.8], hue=[-0.2, 0.2])
        ], p=0.8),
        # final transform
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_none':
        trans = transforms.Compose([
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_resize':
        trans = transforms.Compose([
        transforms.Resize((args.resize_dims, args.resize_dims)),
        transforms.ToTensor()
        ])
    elif args.transforms == 'transform_cropped_resize':
        trans = transforms.Compose([
            CenterCropLongDimension(),
            transforms.Resize((args.resize_dims, args.resize_dims)),
            transforms.ToTensor()
        ])


    if args.temporal:
        dm = ImagePairsDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle_frames = args.shuffle_frames,
            shuffle_temporalWindows = args.shuffle_temporalWindows,
            dataloader_shuffle = args.dataloader_shuffle,
            drop_last=False,
            val_split=args.val_split,
            window_size=args.window_size,
            dataset_size=args.dataset_size,
            gpus=args.gpus,
            transform=trans,
        )
    
    print("[INFO] Shuffle (frames) set to :: ", dm.shuffle_frames)

    print("[INFO] Shuffle (temporal windows) set to :: ", dm.shuffle_temporalWindows)

    print("[INFO] Train dataloader shuffle set to :: ", dm.dataloader_shuffle)

    print("[INFO] Passing through transformations :: {}".format(dm.transform))

    print("[INFO] Loss function version :: {}".format(model.loss_ver))


    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger("/data/lpandey/LOGS/VIT_Time", name=f"{args.exp_name}")
   

    # single gpu training - 
    if args.gpus == 1:
        print("[INFO] Single GPU training selected")
        trainer = pl.Trainer(
            devices=1,
            accelerator='gpu', # cpu will be used if not set to gpu
            max_epochs=args.max_epochs,
            logger=logger,
            sync_batchnorm=True if args.gpus > 1 else False,
            callbacks=callbacks,
        )
    # multi gpu training - 
    elif args.gpus>1:
        print("[INFO] Multi GPU training selected")
        trainer = pl.Trainer(
            num_nodes=1, # NUM_OF_SERVERS
            devices=args.gpus, # NUM_OF_GPUS
            max_epochs=args.max_epochs,
            strategy='ddp',  # Faster than 'dp'
            accelerator='gpu',
            logger=logger,
            sync_batchnorm=True if args.gpus > 1 else False,
            callbacks=callbacks,
        )


    
    #print(model)
    trainer.fit(model, datamodule=dm)
    # trainer.validate(model, datamodule=dm)




if __name__ == '__main__':
    cli_main()