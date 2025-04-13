import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torchmetrics.segmentation
import torchmetrics.segmentation.mean_iou
from tqdm import tqdm
from pathlib import Path
import timm
import torchmetrics

import matplotlib.pyplot as plt

import torchvision.transforms.v2 as v2
from torchvision import tv_tensors


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")

parser.add_argument("--lr_decay", default="cosine", choices=["cosine", "exponential"], help="Decay type")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay strength.")
parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--lr_final", default=0.0, type=float, help="Final learning rate.")
parser.add_argument("--retrain_backbone", default=False, action="store_true", help="Whether to include the backbone in the training process.")
parser.add_argument("--load_model", default=None, type=str, help="Path to model weights to load.")

parser.add_argument("--mask_count", default=2, type=int, help="Number of masks to predict.")



class EfficientUNet(nn.Module):
    def __init__(self, out_channels=2, retrain_backbone=False):
        super().__init__()

        self.retrain_backbone = retrain_backbone
        # Load EfficientNet-B6 backbone 
        self.backbone: timm.models.efficientnet.EfficientNet = timm.create_model("tf_efficientnet_b5.ns_jft_in1k", pretrained=True, num_classes=0)

        _, features = self.backbone.forward_intermediates(torch.zeros(1, 3, 456, 456))
        for feature in features:
            print(feature.shape)

        # Freeze encoder
        if not retrain_backbone:
            self.backbone.requires_grad_(False)

        old_conv = self.backbone.conv_stem

        # Create a new Conv2d with new input channels but same other params
        #print(self.backbone.conv_stem)
        new_conv = nn.Conv2d(1, 48, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)

        if 1 < old_conv.in_channels:
            new_conv.weight.data.copy_(old_conv.weight.data[:, :1])
        elif 1 > old_conv.in_channels:
            new_conv.weight.data[:, :old_conv.in_channels].copy_(old_conv.weight.data)
            # fill extra channels with mean or zeros
            for i in range(old_conv.in_channels, 1):
                new_conv.weight.data[:, i] = old_conv.weight.data.mean(dim=1)

        self.backbone.conv_stem = new_conv
        #print(self.backbone.conv_stem)

        self.stage1 = torch.nn.Sequential(
            torch.nn.LazyConvTranspose2d(176, 1, 2),    torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage2 = torch.nn.Sequential(
            torch.nn.LazyConv2d(176, 3, 1, 'same'),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConvTranspose2d(64, 1, 2),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage3 = torch.nn.Sequential(
            torch.nn.LazyConv2d(64, 3, 1, 'same'),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConvTranspose2d(40, 2, 2),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage4 = torch.nn.Sequential(
            torch.nn.LazyConv2d(40, 3, 1, 'same'),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConvTranspose2d(24, 2, 2),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage5 = torch.nn.Sequential(
            torch.nn.LazyConv2d(24, 3, 1, 'same'),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConvTranspose2d(16, 2, 2),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage6 = torch.nn.Sequential(
            torch.nn.LazyConv2d(16, 3, 1, 'same'),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConv2d(16, 3, 1, 'same'),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConv2d(out_channels, 3, 1, 'same')
        )

        # Init lazy layer sizes
        self.eval()(torch.zeros(1, 1, 456, 456))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)


    def forward(self, images: torch.Tensor):
        _, features = self.backbone.forward_intermediates(images)
        stage1_out = self.stage1(features[4])
        stage2_out = self.stage2(torch.cat((stage1_out, features[3]), dim=1))
        stage3_out = self.stage3(torch.cat((stage2_out, features[2]), dim=1))
        stage4_out = self.stage4(torch.cat((stage3_out, features[1]), dim=1))
        stage5_out = self.stage5(torch.cat((stage4_out, features[0]), dim=1))
        out = self.stage6(torch.cat((stage5_out, images), dim=1))
        return out
    

    def train(self, mode = True):
        self.backbone.train(mode and self.retrain_backbone)
        self.backbone.conv_stem.train(mode)
        self.stage1.train(mode)
        self.stage2.train(mode)
        self.stage3.train(mode)
        self.stage4.train(mode)
        self.stage5.train(mode)
        self.stage6.train(mode)
        return self


class SegmentationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.patients = sorted(os.listdir(data_dir))
        self.transform = transform

        self.slice_files = []
        self.negative_masks_files = []
        self.positive_masks_files = []

        for patient_dir in self.patients:
            for series_dir in sorted(os.listdir(self.data_dir / patient_dir)):
                series_path = self.data_dir / patient_dir / series_dir
                self.slice_files += [series_path / "slices" / file for file in sorted(os.listdir(series_path / "slices"))]
                self.negative_masks_files += [series_path / "negative" / file for file in sorted(os.listdir(series_path / "negative"))]
                self.positive_masks_files += [series_path / "positive" / file for file in sorted(os.listdir(series_path / "positive"))]

    def __len__(self):
        return len(self.slice_files)

    def __getitem__(self, idx):
        imgs = torch.tensor(np.load(self.slice_files[idx])[np.newaxis]).to(torch.float32)
        masks = torch.tensor(np.array((np.load(self.negative_masks_files[idx]),
                                       np.load(self.positive_masks_files[idx])))).to(torch.float32)
        
        masks = tv_tensors.Mask(masks)
        
        if self.transform:
            imgs, masks = self.transform(imgs, masks)

        return imgs, masks

def get_dataloaders(data_root, batch_size=4):
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
        v2.Resize((456, 456)),
        v2.RandomHorizontalFlip(),
        v2.Pad(30),
        v2.RandomCrop(456),
        v2.ColorJitter(0.05, 0.05)
    ])
    
    train_ds = SegmentationDataset(f'{data_root}/train/', transform)
    dev_ds   = SegmentationDataset(f'{data_root}/dev/', transform)
    test_ds  = SegmentationDataset(f'{data_root}/test/', transform)

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(dev_ds, batch_size=batch_size),
        DataLoader(test_ds, batch_size=batch_size)
    )


def train_model(args: argparse.Namespace, model, train_loader, dev_loader, device='cuda'):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.AdamW([
            {"params": [param for name, param in model.named_parameters() if "bias" in name], "weight_decay": 0.},
            {"params": [param for name, param in model.named_parameters() if "bias" not in name]}],
            weight_decay=args.weight_decay,
            lr=args.lr)
    
    iou_metric = torchmetrics.segmentation.MeanIoU(num_classes=args.mask_count, per_class=True).to(device)

    scheduler = None
    if args.lr_decay == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=(args.lr_final / args.lr) ** (1/(args.epochs)))
        
    elif args.lr_decay == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr_final)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            #print(outputs.shape, outputs.dtype, masks.shape, masks.dtype)
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f}) - Train Loss: {running_loss * 1000 / len(train_loader):.4f}")
        
        model.eval()
        val_loss = 0.0
        iou_metric.reset()
        with torch.no_grad():
            for imgs, masks in dev_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.05)
                iou_metric.update(preds.to(torch.int64), masks.to(torch.int64))
        print(f"Dev Loss: {val_loss * 1000 / len(dev_loader):.4f}, Dev IoU: {iou_metric.compute()}")

        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"model_weights_epoch{epoch}.pts")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EfficientUNet(out_channels=main_args.mask_count, retrain_backbone=main_args.retrain_backbone)
    if main_args.load_model:
        model.load_state_dict(torch.load(main_args.load_model, map_location=device))

    train_loader, dev_loader, test_loader = get_dataloaders('data', batch_size=main_args.batch_size)

    # for imgs, masks in dev_loader:
    #     outputs = model(imgs)
        
    #     preds = (torch.sigmoid(outputs) > 0.05)
    #     for batch in range(main_args.batch_size):
    #         colored_mask1 = np.zeros((*preds[batch, 0].shape, 4))  # Initialize a transparent RGBA image
    #         colored_mask1[preds[batch, 0] == 1] = [1, 0, 0, 0.5]  # Red with 50% opacity
    #         colored_mask2 = np.zeros((*preds[batch, 1].shape, 4))  # Initialize a transparent RGBA image
    #         colored_mask2[preds[batch, 1] == 1] = [0, 1, 0, 0.5]  # Green with 50% opacity

    #         # Plot
    #         plt.imshow(imgs[batch, 0], cmap='gray')
    #         plt.imshow(colored_mask1)
    #         plt.imshow(colored_mask2)

    #         plt.axis('off')  # Optional
    #         plt.show()

    train_model(main_args, model, train_loader, dev_loader, device=device)
