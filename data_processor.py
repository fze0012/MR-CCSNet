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
    # In quantitative comparsions, we resize all images into 256*256.
    # In visual comparisons, we crop all image into a*b where a and b are multiple of 32.
    test_set5 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])
    test_set14 = torchvision.transforms.Compose([
        torchvision.transforms.Resize((256, 256)),
        torchvision.transforms.ToTensor(),
    ])

    # Transformers for BSDS
    val_bsds = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((480, 320)),
        torchvision.transforms.ToTensor(),
    ])
    
    trn_dataset = torchvision.datasets.ImageFolder('./BSDS500/train', transform=trn_transforms)
    val_bsds = torchvision.datasets.ImageFolder('./BSDS500/val', transform=val_bsds)
    test_set5 = torchvision.datasets.ImageFolder('./BSDS500/set5', transform=val_set5)
    test_set14 = torchvision.datasets.ImageFolder('./BSDS500/set14', transform=val_set14)

    trn_loader = torch.utils.data.DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, **kwopt, drop_last=False)
    val_loader_bsds = torch.utils.data.DataLoader(val_bsds, batch_size=1, shuffle=True, **kwopt, drop_last=False)
    test_loader_set5 = torch.utils.data.DataLoader(test_set5, batch_size=1, shuffle=True, **kwopt, drop_last=False)
    test_loader_set14 = torch.utils.data.DataLoader(test_set14, batch_size=1, shuffle=True, **kwopt, drop_last=False)

    return trn_loader, val_loader_bsds, test_loader_set5, test_loader_set14
