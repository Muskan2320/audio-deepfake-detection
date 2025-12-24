import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from src.dataset import AudioDataset
from src.model import AudioCNN


def train():
    # ------------------
    # Config
    # ------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3

    # ------------------
    # Dataset
    # ------------------
    dataset = AudioDataset("data/real", "data/fake")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------------------
    # Model
    # ------------------
    model = AudioCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # ------------------
    # Training Loop
    # ------------------
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ------------------
        # Validation Loop
        # ------------------
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%}"
        )

    # ------------------
    # Save Model
    # ------------------
    torch.save(model.state_dict(), "audio_deepfake_model.pth")
    print("✅ Model saved as audio_deepfake_model.pth")


if __name__ == "__main__":
    train()
