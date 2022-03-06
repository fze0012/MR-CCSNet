import torchvision
import torch


def data_loader(args):
    kwopt = {'num_workers': 8, 'pin_memory': True}
    trn_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(args.image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
    ])

    # Transformers for set5 and set14
    # In quantitative comparsions, we resize all images into 256*256(Set5,Set14) and 480*320(BSDS).
    # In visual comparisons, we crop all image into a*b where a and b are multiple of 32.
    val_set5 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_set14 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Transformers for BSDS
    val_bsds = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((480, 320)),
        torchvision.transforms.ToTensor(),
    ])
    trn_dataset = torchvision.datasets.ImageFolder('./datasets/train', transform=trn_transforms)

    val_bsds = torchvision.datasets.ImageFolder('./datasets/val', transform=val_bsds)
    val_set5 = torchvision.datasets.ImageFolder('./datasets/set5', transform=val_set5)
    val_set14 = torchvision.datasets.ImageFolder('./datasets/set14', transform=val_set14)

    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, **kwopt,
                                             drop_last=False)
    val_loader_bsds = torch.utils.data.DataLoader(val_bsds, batch_size=1, shuffle=True, **kwopt, drop_last=False)
    val_loader_set5 = torch.utils.data.DataLoader(val_set5, batch_size=1, shuffle=True, **kwopt, drop_last=False)
    val_loader_set14 = torch.utils.data.DataLoader(val_set14, batch_size=1, shuffle=True, **kwopt, drop_last=False)

    return trn_loader, val_loader_bsds, val_loader_set5, val_loader_set14
