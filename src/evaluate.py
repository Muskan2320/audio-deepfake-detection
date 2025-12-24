import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.dataset import AudioDataset
from src.model import AudioCNN


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------
    # Load dataset
    # ------------------
    dataset = AudioDataset("test_data/real", "test_data/fake")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size])

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # ------------------
    # Load trained model
    # ------------------
    model = AudioCNN().to(device)
    model.load_state_dict(
        torch.load("audio_deepfake_model.pth", map_location=device)
    )
    model.eval()

    y_true, y_pred = [], []

    # ------------------
    # Evaluation loop
    # ------------------
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)
            preds = outputs.argmax(dim=1)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # ------------------
    # Metrics
    # ------------------
    print("Evaluation Metrics")
    print("==================")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")


if __name__ == "__main__":
    evaluate()
