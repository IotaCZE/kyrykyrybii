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


#### Where shit might hit the fan:
# First conv still frozen
# model.train not being set to false in eval for backbone
# Output size being 528x528
# Input mask loading not including multiple masks

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")

parser.add_argument("--lr_decay", default="cosine", choices=["cosine", "exponential"], help="Decay type")
parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay strength.")
parser.add_argument("--lr", default=0.01, type=float, help="Initial learning rate.")
parser.add_argument("--lr_final", default=0.0, type=float, help="Final learning rate.")
parser.add_argument("--retrain_backbone", default=False, action="store_true", help="Whether to include the backbone in the training process.")
parser.add_argument("--load_model", default=None, type=str, help="Path to model weights to load.")

parser.add_argument("--mask_count", default=13, type=int, help="Number of masks to predict.")



class EfficientUNet(nn.Module):
    def __init__(self, out_channels=5, retrain_backbone=False):
        super().__init__()

        self.retrain_backbone = retrain_backbone
        # Load EfficientNet-B6 backbone 
        self.backbone: timm.models.efficientnet.EfficientNet = timm.create_model("tf_efficientnet_b6.ap_in1k", pretrained=True, num_classes=0)

        # Freeze encoder
        if not retrain_backbone:
            self.backbone.requires_grad_(False)

        self.stage1 = torch.nn.Sequential(
            torch.nn.LazyConvTranspose2d(200, 1, 2),    torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage2 = torch.nn.Sequential(
            torch.nn.LazyConv2d(200, 3, 1, 'same'),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConvTranspose2d(72, 2, 2),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage3 = torch.nn.Sequential(
            torch.nn.LazyConv2d(72, 3, 1, 'same'),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConvTranspose2d(40, 2, 2),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage4 = torch.nn.Sequential(
            torch.nn.LazyConv2d(40, 3, 1, 'same'),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConvTranspose2d(32, 2, 2),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage5 = torch.nn.Sequential(
            torch.nn.LazyConv2d(32, 3, 1, 'same'),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(),
            torch.nn.LazyConvTranspose2d(20, 2, 2),     torch.nn.LazyBatchNorm2d(), torch.nn.ReLU()
        )
        self.stage6 = torch.nn.Sequential(
            torch.nn.LazyConv2d(20, 5, 1, padding=0),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(), # 528 -> 524
            torch.nn.LazyConv2d(20, 5, 1, padding=0),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(), # 524 -> 520
            torch.nn.LazyConv2d(20, 5, 1, padding=0),      torch.nn.LazyBatchNorm2d(), torch.nn.ReLU(), # 520 -> 516
            torch.nn.LazyConv2d(out_channels, 5, 1, padding=0)                                          # 516 -> 512
        )

        # Init lazy layer sizes
        self.eval()(torch.zeros(1, 3, 528, 528))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        for lay in self.modules():
            if type(lay) in [torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear]:
                torch.nn.init.xavier_uniform_(lay.weight)


    def forward(self, images: torch.Tensor):
        print(torch.cuda.memory_allocated(0))
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

        self.slices = []
        self.masks = []

        with open('target_contours.txt', 'r') as f:
            target_contours_raw = f.readline().strip()
            target_contours = target_contours_raw.split(sep=',')

        for patient_dir in self.patients:
            slices_dir = self.data_dir / patient_dir / "slices"
            slice_files = sorted(os.listdir(slices_dir))
            contour_files = []
            for contour_name in target_contours:
                contour_files.append(sorted(os.listdir(self.data_dir / patient_dir / contour_name)))
                
            for i in range(1, len(slice_files) - 1):
                slices = np.array((
                    np.load(self.data_dir / patient_dir / "slices" / slice_files[i-1]),
                    np.load(self.data_dir / patient_dir / "slices" / slice_files[i]),
                    np.load(self.data_dir / patient_dir / "slices" / slice_files[i+1])))
                self.slices.append(slices)

                contour_masks = [np.load(self.data_dir / patient_dir / target_contours[contour_i] / contour_files[contour_i][i-1])
                         for contour_i in range(len(contour_files))]
                self.masks.append(np.array(contour_masks))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        imgs = torch.tensor(self.slices[idx]).to(torch.float32) / 255
        masks = torch.tensor(self.masks[idx]).to(torch.float32)
        
        if self.transform:
            imgs = self.transform(imgs)

        return imgs, masks

def get_dataloaders(data_root, batch_size=4):
    transform = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.5], [0.5]),
        v2.Resize((528, 528))
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
        print(f"Epoch {epoch+1}/{args.epochs} (lr={scheduler.get_last_lr()[0]:.6f}) - Train Loss: {running_loss / len(train_loader):.4f}")
        
        model.eval()
        val_loss = 0.0
        iou_metric.reset()
        with torch.no_grad():
            for imgs, masks in dev_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = (outputs > 0)
                iou_metric.update(preds.to(torch.int64), masks.to(torch.int64))
        print(f"Dev Loss: {val_loss / len(dev_loader):.4f}, Dev IoU: {iou_metric.compute()}")

        if epoch % 1 == 0:
            torch.save(model.state_dict(), f"model_weights_epoch{epoch}.pts")


if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EfficientUNet(out_channels=main_args.mask_count, retrain_backbone=main_args.retrain_backbone)
    model.load_state_dict(torch.load("model_weights_epoch13.pts", map_location=device))

    train_loader, dev_loader, test_loader = get_dataloaders('data', batch_size=main_args.batch_size)

    for imgs, masks in dev_loader:
        outputs = model(imgs)
        plt.imshow(torch.sigmoid(outputs))

    train_model(main_args, model, train_loader, dev_loader, device=device)
