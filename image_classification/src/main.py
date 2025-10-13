"""
muon_paper_validator_tunable_adam.py

A trainer script with a tunable learning rate for the auxiliary AdamW optimizer,
while keeping the main learning rate fixed as requested.
"""


#############################################
#                  Setup                    #
#############################################
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb
import numpy as np
import os
import math
from typing import List, Optional, Tuple, Union
# import timm
from models import *
from optimizers import SingleDeviceMuonWithAuxAdam, ShampooOptimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    MIXED_PRECISION_DTYPE = torch.bfloat16
    print("bfloat16 is supported and will be used for Muon internal computation.")
else:
    MIXED_PRECISION_DTYPE = torch.float16
    print("bfloat16 not supported, falling back to float16 for Muon internal computation.")

#############################################
#               Optimizers                  #
#############################################
# Optimizer implementations are now imported from src/optimizers/



#############################################
#            Network and Data               #
#############################################


def get_model(arch, num_classes=10):
    if arch == "resnet18":
        assert num_classes == 10, "ResNet18 is only supported for CIFAR-10"
        return ResNet18_CIFAR10()
    elif arch == "vgg16_bn" or arch == "vgg16":
        assert num_classes == 100, "VGG16_BN is only supported for CIFAR-100"
        return vgg16_bn()
    elif arch.startswith("vit"):

        # https://huggingface.co/timm/vit_small_patch16_224.augreg_in21k
        # https://github.com/kentaroy47/vision-transformers-cifar10/blob/main/train_cifar10.py
        # For ViT, we use the timm library
        # Example model: vit_tiny_patch16_32
        # return timm.create_model(arch, pretrained=False, num_classes=num_classes, img_size=32)
        net = timm.create_model("vit_small_patch16_224.augreg_in21k", pretrained=True)
        net.head = nn.Linear(net.head.in_features, num_classes)
        return net
    else:
        raise ValueError(
            f"Unknown architecture: {arch}. Supported: simplecnn, resnet18, vgg16_bn, vit_*.")


def get_cifar10_loader(batch_size, img_size = 32, train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616))
        ])

    dataset = datasets.CIFAR10(
        './data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, pin_memory=True, num_workers=4, drop_last=train)


def get_cifar100_loader(batch_size, img_size = 32, train=True):
    if train:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761))
        ])

    dataset = datasets.CIFAR100(
        './data', train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, pin_memory=True, num_workers=4, drop_last=train)


@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for data, target in test_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        output = model(data)
        loss = F.cross_entropy(output, target, reduction='sum')
        total_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    return total_loss / total, 100 * correct / total

#############################################
#                  Main                     #
#############################################


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    effective_batch_size = args.bs_per_gpu
    n_gpu = torch.cuda.device_count()
    print(f"Using {n_gpu} GPUs, effective batch size: {effective_batch_size}")
    config = vars(args)
    config['effective_batch_size'] = effective_batch_size

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "dryrun"

    with wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_expname, config=vars(args)) as run:

        # Determine number of classes based on dataset
        num_classes = 100 if args.dataset == 'cifar100' else 10
        
        model = get_model(args.arch, num_classes=num_classes).to(DEVICE)
        if n_gpu > 1:
            model = nn.DataParallel(model)

        per_step_loader_bs = args.bs_per_gpu * n_gpu

        if args.arch.startswith('vit'):
            img_size = 224
        else:
            img_size = 32

        if args.dataset == 'cifar10':
            train_loader = get_cifar10_loader(per_step_loader_bs, img_size=img_size, train=True)
            test_loader = get_cifar10_loader(per_step_loader_bs, img_size=img_size, train=False)
        elif args.dataset == 'cifar100':
            train_loader = get_cifar100_loader(per_step_loader_bs, img_size=img_size, train=True)
            test_loader = get_cifar100_loader(per_step_loader_bs, img_size=img_size, train=False)
        else:
            raise ValueError(f"Dataset '{args.dataset}' not supported.")


        if args.optimizer.lower() == 'muon':
            model_to_get_params = model.module if n_gpu > 1 else model
            
            if args.arch.startswith('vit'):
                # For ViT, apply Muon to transformer blocks and AdamW to the rest
                # muon_params = list(model_to_get_params.blocks.parameters())

                muon_params = []
                adam_params = []

                for name, param in model_to_get_params.named_parameters():
                    if 'blocks' in name and param.ndim >= 2:
                        muon_params.append(param)
                    else:
                        adam_params.append(param)

            elif args.arch in ['simplecnn', 'resnet18', 'vgg16_bn']:
                # For CNNs, apply Muon to all but the first conv layer
                muon_params = []
                first_conv_seen = False
                for m in model_to_get_params.modules():
                    if isinstance(m, nn.Conv2d):
                        if first_conv_seen:
                            muon_params.extend(list(m.parameters()))
                        else:
                            first_conv_seen = True
            else:
                raise ValueError(f"Muon optimizer logic not defined for arch: {args.arch}")

            muon_id_set = {id(p) for p in muon_params}
            adam_params = [p for p in model_to_get_params.parameters() if id(p) not in muon_id_set]

            assert not (set(adam_params) & set(muon_params)), "param overlap!"

            print(f"Muon params  : {sum(p.numel() for p in muon_params):,}")
            print(f"AdamW params : {sum(p.numel() for p in adam_params):,}")

            param_groups = [
                {
                    'params': muon_params, 'use_muon': True,
                    'lr': args.lr, 'weight_decay': args.weight_decay, 'momentum': args.momentum,
                    'nesterov': args.nesterov, 'ns_steps': 3
                },
                {
                    'params': adam_params, 'use_muon': False,
                    'lr': args.adam_lr,
                    'weight_decay': args.weight_decay,
                    'betas': (0.9, 0.999),
                    'eps': 1e-8
                }
            ]
            optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
            muon_params_to_track = muon_params

        elif args.optimizer.lower() == 'msgd':
            optimizer = optim.SGD(
                model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
            muon_params_to_track = []

        elif args.optimizer.lower() == 'shampoo':
            optimizer = ShampooOptimizer(
                model.parameters(),
                lr=args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                epsilon=args.shampoo_epsilon,
                update_freq=args.shampoo_update_freq,
                inv_freq=1,  # Not exposed as argument for simplicity
                precond_freq=args.shampoo_precond_freq,
                start_precond=args.shampoo_start_precond,
                block_size=args.shampoo_block_size,
                dtype=torch.float32,
                use_nesterov=args.nesterov,
                use_bias_correction=True,
                use_decoupled_weight_decay=True,
                graft_type=args.shampoo_graft_type,
                graft_epsilon=args.shampoo_graft_epsilon,
                graft_beta1=args.shampoo_graft_beta1,
                graft_beta2=args.shampoo_graft_beta2,
            )
            muon_params_to_track = []

        else:  # adamw
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            muon_params_to_track = []

        print(
            f"Starting training for {args.epochs} epochs with optimizer '{args.optimizer}'.")
        global_step = 0
        steps_to_target_train_acc = -1
        steps_to_target_test_acc = -1

        for epoch in range(args.epochs):
            model.train()
            epoch_train_loss, epoch_correct, epoch_total = 0.0, 0, 0
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()

                muon_grad_norm = 0.0
                if args.optimizer.lower() == 'muon':
                    if muon_params_to_track:
                        with torch.no_grad():
                            for p in muon_params_to_track:
                                if p.grad is not None:
                                    muon_grad_norm += p.grad.norm(
                                        p='fro').item()**2
                        muon_grad_norm = np.sqrt(muon_grad_norm)
                    else:
                        muon_grad_norm = 0.0
                else:
                    muon_grad_norm = 0.0

                if np.isnan(muon_grad_norm) or np.isinf(muon_grad_norm) or muon_grad_norm > 1e4:
                    print(
                        f"Divergence detected! Grad norm: {muon_grad_norm}. Stopping run.")
                    run.log({"batch_train_loss": loss.item(),
                            "muon_grad_norm": muon_grad_norm}, step=global_step)
                    run.summary["exp_status"] = "diverged"
                    return

                optimizer.step()

                epoch_train_loss += loss.item()
                with torch.no_grad():
                    _, predicted = torch.max(output.data, 1)
                    epoch_total += target.size(0)
                    epoch_correct += (predicted == target).sum().item()

                if i % 50 == 0:
                    run.log({"batch_train_loss": loss.item(),
                            "muon_grad_norm": muon_grad_norm}, step=global_step)

                global_step += 1

            avg_train_loss = epoch_train_loss/len(train_loader)
            avg_train_accuracy = 100*epoch_correct/epoch_total
            test_loss, test_accuracy = evaluate_model(model, test_loader)
            print(f"Epoch {epoch+1} finished. Train Acc: {avg_train_accuracy:.2f}%, Train Loss: {avg_train_loss:.4f} | Test Acc: {test_accuracy:.2f}%, Test Loss: {test_loss:.4f}")
            log_dict = {
                "epoch": epoch + 1,
                "avg_train_accuracy": avg_train_accuracy,
                "avg_train_loss": avg_train_loss,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
            }

            if steps_to_target_train_acc == -1 and avg_train_accuracy >= args.target_train_accuracy:
                steps_to_target_train_acc = global_step
                sfo_complexity_train = steps_to_target_train_acc * effective_batch_size
                print(f"\n>>>> Target TRAIN accuracy {args.target_train_accuracy}% reached at step {steps_to_target_train_acc}! <<<<")
                print(f"     Train SFO Complexity: {sfo_complexity_train:.2e}")
                run.summary["steps_to_target_train_acc"] = steps_to_target_train_acc
                run.summary["sfo_complexity_train"] = sfo_complexity_train

            if steps_to_target_test_acc == -1 and test_accuracy >= args.target_test_accuracy:
                steps_to_target_test_acc = global_step
                sfo_complexity_test = steps_to_target_test_acc * effective_batch_size
                print(f"\n>>>> Target TEST accuracy {args.target_test_accuracy}% reached at step {steps_to_target_test_acc}! <<<<")
                print(f"     Test SFO Complexity: {sfo_complexity_test:.2e}")
                run.summary["steps_to_target_test_acc"] = steps_to_target_test_acc
                run.summary["sfo_complexity_test"] = sfo_complexity_test

            run.log(log_dict, step=global_step)

            if args.early_stopping and (steps_to_target_train_acc != -1 or steps_to_target_test_acc != -1):
                print(f"\n>>>> Training stopped early! <<<<")
                print(f"     Train target reached: {steps_to_target_train_acc != -1}")
                print(f"     Test target reached: {steps_to_target_test_acc != -1}")
                break

        if steps_to_target_train_acc == -1:
            run.summary["steps_to_target_train_acc"] = float('inf')
            run.summary["sfo_complexity_train"] = float('inf')
        
        if steps_to_target_test_acc == -1:
            run.summary["steps_to_target_test_acc"] = float('inf')
            run.summary["sfo_complexity_test"] = float('inf')
        
        if steps_to_target_train_acc != -1 or steps_to_target_test_acc != -1:
            run.summary["exp_status"] = "completed"
        else:
            run.summary["exp_status"] = "target_not_reached"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Trainer to validate the Muon convergence paper.")
    # General settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arch', type=str, default='simplecnn',
                        help="Model architecture. Supported: simplecnn, resnet18, and any ViT from timm (e.g., vit_tiny_patch16_32).")
    parser.add_argument('--bs_per_gpu', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)

    # Optimizer settings
    parser.add_argument('--optimizer', type=str,
                        required=True, choices=['muon', 'adamw', 'msgd', 'shampoo'],)
    parser.add_argument('--lr', type=float, required=True,
                        help="Learning rate for the main optimizer (Muon, AdamW, or Shampoo)")
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help='Weight decay (lambda in the paper for Muon)')
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--nesterov', action='store_true',
                        help='Use Nesterov momentum for Muon and Shampoo.')

    parser.add_argument('--adam_lr', type=float, default=3e-4,
                        help="Learning rate for the auxiliary AdamW optimizer used with Muon.")
    
    parser.add_argument('--shampoo_epsilon', type=float, default=1e-8,
                        help="Epsilon for numerical stability in Shampoo")
    parser.add_argument('--shampoo_update_freq', type=int, default=1,
                        help="Frequency for updating Shampoo statistics")
    parser.add_argument('--shampoo_precond_freq', type=int, default=1,
                        help="Frequency for updating Shampoo preconditioners")
    parser.add_argument('--shampoo_start_precond', type=int, default=1000,
                        help="Step to start using Shampoo preconditioners")
    parser.add_argument('--shampoo_block_size', type=int, default=8192,
                        help="Block size for Shampoo optimization")
    parser.add_argument('--shampoo_graft_type', type=str, default="none", 
                        choices=["none", "adagrad", "sgd", "rmsprop"],
                        help="Graft type for Shampoo optimizer")
    parser.add_argument('--shampoo_graft_epsilon', type=float, default=1e-8,
                        help="Epsilon for graft optimizer in Shampoo")
    parser.add_argument('--shampoo_graft_beta1', type=float, default=0.9,
                        help="Beta1 for graft optimizer in Shampoo")
    parser.add_argument('--shampoo_graft_beta2', type=float, default=0.999,
                        help="Beta2 for graft optimizer in Shampoo")

    parser.add_argument('--target_test_accuracy', type=float, default=75.0,
                        help='The target test accuracy to measure time-to-accuracy for SFO calculation.')
    parser.add_argument('--target_train_accuracy', type=float, default=90.0,
                        help='The target train accuracy to measure time-to-accuracy for SFO calculation.')

    # Dataset settings
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                        help='Dataset to use for training and testing.')
    
    # Early stopping settings
    parser.add_argument('--early_stopping', action='store_true',
                        help='Stop training when either train or test target accuracy is reached.')

    # W&B settings
    parser.add_argument('--wandb_project', type=str,
                        default='muon-paper-validation')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_expname', type=str, default=None)
    parser.add_argument('--wandb_offline', action = 'store_true')

    args = parser.parse_args()

    # Auto-generate experiment name if not provided
    if args.wandb_expname is None:
        nesterov_str = "nesterov" if args.nesterov else "no_nesterov"
        wd_str = f"wd{args.weight_decay}" if args.weight_decay > 0 else "no_wd"
        adam_lr_str = f"_adamlr{args.adam_lr}" if args.optimizer == 'muon' else ""
        
        # Add Shampoo-specific parameters to experiment name
        shampoo_str = ""
        if args.optimizer == 'shampoo':
            graft_str = f"_graft{args.shampoo_graft_type}" if args.shampoo_graft_type != "none" else ""
            precond_str = f"_precond{args.shampoo_start_precond}" if args.shampoo_start_precond != 1000 else ""
            shampoo_str = f"{graft_str}{precond_str}"
        
        args.wandb_expname = f"{args.dataset}_{args.arch}_{args.optimizer}_{nesterov_str}_{wd_str}_lr{args.lr}{adam_lr_str}{shampoo_str}_bs{args.bs_per_gpu}"

    if args.wandb_entity is None:
        if 'WANDB_ENTITY' in os.environ:
            args.wandb_entity = os.environ['WANDB_ENTITY']
        else:
            raise ValueError(
                "W&B entity must be provided via --wandb_entity argument or WANDB_ENTITY environment variable.")

    main(args)
