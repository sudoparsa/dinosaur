import torch
import argparse
import os
import time
from torch import nn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from pathlib import Path
import einops
import utils
from model import Dinosaur
import json
import datetime


def get_args_parser():
    parser = argparse.ArgumentParser('DINOSAUR', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory.""")
    parser.add_argument('--pretrained_weights', default='.//checkpoint.pth', type=str,
        help='Path to dino pretrained weights')
    parser.add_argument('--checkpoint_key', default='teacher', type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--num_slots', default=2, type=int,
        help='Number of slots in SlotAttention module')
    parser.add_argument('--dim', default=192, type=int,
        help='Dimension of each slot')
    parser.add_argument('--iters', default=3, type=int,
        help='Iterations of SlotAttention')
    parser.add_argument('--epsilon', default=1e-8, type=float,
        help='Epsilon of SlotAttention')
    parser.add_argument('--slot_hdim', default=128, type=int,
        help='Hidden dimension of SlotAttention')

    # Training/Optimization parameters
    parser.add_argument('--batch_size', default=256, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate""")
    parser.add_argument('--weight_decay', type=float, default=0.00001, help="""Initial value of the
        weight decay.""")
    
    # Misc
    parser.add_argument('--data_path', default='/path/to/bcmnist/train/', type=str,
        help='Please specify path to the training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=42, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=2, type=int, help='Number of data loading workers per GPU.')
    return parser


def train_dino(args):
    utils.fix_random_seeds(args.seed)
    f = open('parameters.txt', 'w')
    f.write(' '.join(f'{k}={v}\n' for k, v in vars(args).items()))
    f.close()

    # ============ preparing data ... ============
    # Maybe better to not transform at all
    transform = pth_transforms.Compose([
        pth_transforms.Resize(36, interpolation=3),
        pth_transforms.CenterCrop(32),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform)
    dataset_test = datasets.ImageFolder(os.path.join(args.data_path, 'test'), transform=transform)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train, {len(dataset_val)} val, and {len(dataset_test)} test imgs.")
    # print(f"Data loaded with {len(dataset_train)} train, and {len(dataset_val)} val imgs.")
    # print(f"Data loaded with {len(dataset_train)} train imgs.")

    # ============ building model ... ============
    model = Dinosaur(args).cuda()

    # ============ preparing loss ... ============
    reconstruction_loss = nn.MSELoss()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ============ init schedulers ... ============
    
    start_time = time.time()
    print("Starting DINOSAUR training !")
    for epoch in range(args.epochs):

        # ============ training one epoch of DINO ... ============
        model.train()
        print('Train...')
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
        for it, (images, _) in enumerate(metric_logger.log_every(data_loader_train, 50, header)):
            it = len(data_loader_train) * epoch + it  # global training iteration
            # forward passes + compute reconstruction loss
            output = model(images.cuda())
            decoder_output = einops.rearrange(output['decoder_output']['output'], 'b e w h -> b (w h) e')
            loss = reconstruction_loss(output['dino_output'], decoder_output)
            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # logging
            torch.cuda.synchronize()
            metric_logger.update(loss=loss.item())
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # ============ evaluation ============
        with torch.no_grad():
            model.eval()
            print('Validation...')
            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
            for it, (images, _) in enumerate(metric_logger.log_every(data_loader_val, 50, header)):
                # forward passes + compute reconstruction loss
                output = model(images.cuda())
                decoder_output = einops.rearrange(output['decoder_output']['output'], 'b e w h -> b (w h) e')
                val_loss = reconstruction_loss(output['dino_output'], decoder_output)
                # logging
                torch.cuda.synchronize()
                metric_logger.update(loss=val_loss.item())
            print("Averaged stats:", metric_logger)
            val_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            
            print('Test...')
            metric_logger = utils.MetricLogger(delimiter="  ")
            header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
            for it, (images, _) in enumerate(metric_logger.log_every(data_loader_test, 50, header)):
                # forward passes + compute reconstruction loss
                output = model(images.cuda())
                decoder_output = einops.rearrange(output['decoder_output']['output'], 'b e w h -> b (w h) e')
                test_loss = reconstruction_loss(output['dino_output'], decoder_output)
                # logging
                torch.cuda.synchronize()
                metric_logger.update(loss=test_loss.item())
            print("Averaged stats:", metric_logger)
            test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'train_loss': loss.item(),
            'val_loss': val_loss.item(),
            'test_loss': test_loss.item(),
        }
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINOSAUR', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)