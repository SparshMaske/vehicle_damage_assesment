import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


LABELS = ["Minor", "Moderate", "Severe"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train severity classifier.")
    parser.add_argument(
        "--data-dir",
        default="data/severity",
        help="ImageFolder dataset root with Minor/Moderate/Severe subdirectories.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output", default="models/weights/severity.pth")
    return parser.parse_args()


def build_dataloaders(data_dir: Path, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)
    if not dataset.classes:
        raise RuntimeError(f"No classes found under {data_dir}")

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise RuntimeError("Need at least two images to split train/validation datasets.")
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total if total else 0.0


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Severity dataset not found: {data_dir}. Run data/prepare_severity_dataset.py first."
        )

    train_loader, val_loader = build_dataloaders(data_dir, args.batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        accuracy = evaluate(model, val_loader, device)
        mean_loss = running_loss / max(1, len(train_loader))
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={mean_loss:.4f} val_accuracy={accuracy:.4f}"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Saved severity weights to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
