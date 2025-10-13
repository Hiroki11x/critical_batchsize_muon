import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import argparse

def get_cifar100_loader(dir_path, batch_size, img_size=32, train=True):
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
    dataset = datasets.CIFAR100(dir_path, train=train, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, pin_memory=True, num_workers=4, drop_last=train)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--per_step_loader_bs", type=int, default=512)
    args.add_argument("--img_size", type=int, default=32)
    args.add_argument("--down_load_dir", type=str, default="./data")

    args = args.parse_args()
    per_step_loader_bs = args.per_step_loader_bs
    img_size = args.img_size
    down_load_dir = args.down_load_dir

    print("start of script")

    train_loader = get_cifar100_loader(down_load_dir, per_step_loader_bs, img_size=img_size, train=True)
    test_loader = get_cifar100_loader(down_load_dir, per_step_loader_bs, img_size=img_size, train=False)

    print(len(train_loader))
    print(len(test_loader))

    print("end of script")
