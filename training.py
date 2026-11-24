import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
import model as models


# ── Multi-binary losses ────────────────────────────────────────────────────────

class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1).float()
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)


class MultiBinaryCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = BinaryDiceLoss()
        self.pos_weights = torch.FloatTensor(config.BINARY_CLASS_WEIGHTS).to(config.DEVICE)

    def forward(self, pred, target):
        # pred: (B, 3, H, W) logits  |  target: (B, H, W) multiclass labels
        total = 0
        for i in range(pred.shape[1]):
            binary_target = (target == (i + 1)).float()
            bce = nn.BCEWithLogitsLoss(pos_weight=self.pos_weights[i:i+1])
            total += self.dice(pred[:, i], binary_target)
            total += bce(pred[:, i], binary_target)
        return total / pred.shape[1]


class DiceLoss(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_oh = torch.zeros_like(pred)
        target_oh.scatter_(1, target.unsqueeze(1), 1)

        dice = 0
        for c in range(self.num_classes):
            p = pred[:, c].contiguous().view(-1)
            t = target_oh[:, c].contiguous().view(-1)
            intersection = (p * t).sum()
            dice += (2 * intersection + self.smooth) / (p.sum() + t.sum() + self.smooth)

        return 1 - dice / self.num_classes


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.dice = DiceLoss(num_classes)
        weights = torch.FloatTensor(config.CLASS_WEIGHTS).to(config.DEVICE)
        self.ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, pred, target):
        return self.dice(pred, target) + self.ce(pred, target)


def compute_dice_per_class(pred, target, num_classes=config.NUM_CLASSES):
    pred_mask = pred.argmax(dim=1)
    dice_scores = []
    for c in range(num_classes):
        p = (pred_mask == c).float().view(-1)
        t = (target == c).float().view(-1)
        intersection = (p * t).sum()
        score = (2 * intersection) / (p.sum() + t.sum() + 1e-5)
        dice_scores.append(score.item())
    return dice_scores


def compute_binary_dice_per_class(pred, target):
    """pred: (B,3,H,W) logits | target: (B,H,W) multiclass labels"""
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    dice_scores = []
    for i in range(pred.shape[1]):
        p = pred_binary[:, i].reshape(-1)
        t = (target == (i + 1)).float().reshape(-1)
        intersection = (p * t).sum()
        dice_scores.append((2 * intersection / (p.sum() + t.sum() + 1e-5)).item())
    return dice_scores


def train_multibinary(train_gen, val_loader, steps_per_epoch):
    print(f"Using device: {config.DEVICE}")

    net = models.new_model()
    criterion = MultiBinaryCombinedLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_val_dice = 0
    train_losses, val_losses, val_dices = [], [], []

    for epoch in tqdm(range(config.EPOCHS), desc="Training"):
        net.train()
        train_loss = 0
        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1} [train]", leave=False):
            images, masks = next(train_gen)
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= steps_per_epoch
        train_losses.append(train_loss)

        net.eval()
        val_loss = 0
        val_dice = np.zeros(len(config.BINARY_CLASS_NAMES))
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", leave=False):
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                outputs = net(images)
                val_loss += criterion(outputs, masks).item()
                val_dice += np.array(compute_binary_dice_per_class(outputs, masks))
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        mean_dice = val_dice.mean()

        val_losses.append(val_loss)
        val_dices.append(mean_dice)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"Epoch {epoch+1}/{config.EPOCHS} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            tqdm.write(f"  Dice: {' | '.join(f'{n}: {d:.3f}' for n, d in zip(config.BINARY_CLASS_NAMES, val_dice))}")
            tqdm.write(f"  Mean Dice: {mean_dice:.3f}")

        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            torch.save(net.state_dict(), f"{config.MODELS_DIR}/best_model.pth")

    print(f"\nTraining complete. Best mean Dice: {best_val_dice:.3f}")
    return net, train_losses, val_losses, val_dices


def train(train_gen, val_loader, steps_per_epoch):
    print(f"Using device: {config.DEVICE}")

    net = models.new_model()
    criterion = CombinedLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    best_val_dice = 0
    train_losses = []
    val_losses = []
    val_dices = []

    for epoch in tqdm(range(config.EPOCHS), desc="Training"):
        # Train
        net.train()
        train_loss = 0
        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1} [train]", leave=False):
            images, masks = next(train_gen)
            images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= steps_per_epoch
        train_losses.append(train_loss)

        # Validate
        net.eval()
        val_loss = 0
        val_dice = np.zeros(config.NUM_CLASSES)
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch+1} [val]", leave=False):
                images, masks = images.to(config.DEVICE), masks.to(config.DEVICE)
                outputs = net(images)
                val_loss += criterion(outputs, masks).item()
                val_dice += np.array(compute_dice_per_class(outputs, masks))
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        mean_dice = val_dice[1:].mean()

        val_losses.append(val_loss)
        val_dices.append(mean_dice)
        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            tqdm.write(f"Epoch {epoch+1}/{config.EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            tqdm.write(f"  Dice: {' | '.join(f'{n}: {d:.3f}' for n, d in zip(config.CLASS_NAMES, val_dice))}")
            tqdm.write(f"  Mean Dice (excl. bg): {mean_dice:.3f}")

        # Save best model
        if mean_dice > best_val_dice:
            best_val_dice = mean_dice
            save_path = f"{config.MODELS_DIR}/best_model.pth"
            torch.save(net.state_dict(), save_path)

    print(f"\nTraining complete. Best mean Dice: {best_val_dice:.3f}")
    print(f"Model saved to {config.MODELS_DIR}/best_model.pth")

    return net, train_losses, val_losses, val_dices
