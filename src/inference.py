import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

from src.dataset import AudioDataset
from src.model import AudioCNN


class SingleAudioDataset(Dataset):
    """
    Reuses AudioDataset preprocessing for ONE audio file
    """
    def __init__(self, audio_path: Path):
        self.audio_path = audio_path

        # Create a normal AudioDataset instance
        self.dataset = AudioDataset(
            real_dir=audio_path.parent,
            fake_dir=audio_path.parent
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # Temporarily replace internal path list
        original_real = getattr(self.dataset, "real_files", None)
        original_fake = getattr(self.dataset, "fake_files", None)

        # Force dataset to process ONLY this file
        self.dataset.real_files = [str(self.audio_path)]
        self.dataset.fake_files = []

        x, y = self.dataset[0]  # y is dummy

        # Restore state
        if original_real is not None:
            self.dataset.real_files = original_real
        if original_fake is not None:
            self.dataset.fake_files = original_fake

        return x


def predict(audio_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    audio_path = Path(audio_path)

    # ------------------
    # Load model
    # ------------------
    model = AudioCNN().to(device)
    model.load_state_dict(
        torch.load("audio_deepfake_model.pth", map_location=device)
    )
    model.eval()

    # ------------------
    # Load audio via dataset
    # ------------------
    dataset = SingleAudioDataset(audio_path)
    x = dataset[0].unsqueeze(0).to(device)  # add batch dim

    # ------------------
    # Inference
    # ------------------
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        confidence, prediction = torch.max(probs, dim=1)

    label = "FAKE" if prediction.item() == 1 else "REAL"

    print("=================================")
    print(f"Audio file : {audio_path}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence.item() * 100:.2f}%")
    print("=================================")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m src.inference <audio.wav>")
        exit(1)

    predict(sys.argv[1])
