from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


class SimpleForgeryCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CNN sahtecilik modeli eğitimi")
    parser.add_argument("--data-dir", type=str, default="dataset/train")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-out", type=str, default="models/deepfake_detector.pth")
    parser.add_argument("--onnx-out", type=str, default="models/deepfake_detector.onnx")
    return parser.parse_args()


def build_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    return running_loss / max(1, len(loader.dataset))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(1, total)


def export_onnx(model: nn.Module, output_path: Path, device: torch.device) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224, device=device)
    model.eval()
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["logit"],
        dynamic_axes={"input": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=13,
    )


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise SystemExit(f"Dataset dizini bulunamadı: {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(data_dir, args.batch_size)

    model = SimpleForgeryCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{args.epochs} - loss: {train_loss:.4f} - val_acc: {val_acc:.4f}")

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_out)
    print(f"PyTorch model kaydedildi: {model_out}")

    export_onnx(model, Path(args.onnx_out), device)
    print(f"ONNX model kaydedildi: {args.onnx_out}")


if __name__ == "__main__":
    main()
