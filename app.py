import gradio as gr
import torch
import torch.nn.functional as F

from src.model import AudioCNN
from src.preprocess import extract_mfcc


# ------------------
# Load model
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioCNN().to(device)
model.load_state_dict(
    torch.load("audio_deepfake_model.pth", map_location=device)
)
model.eval()


# ------------------
# Prediction function
# ------------------
def predict(audio_path):
    mfcc = extract_mfcc(audio_path)
    mfcc = torch.tensor(mfcc).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(mfcc)
        probs = F.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    label = "FAKE AUDIO" if pred.item() == 1 else "REAL AUDIO"
    return f"{label} (Confidence: {confidence.item() * 100:.2f}%)"


# ------------------
# Gradio Interface
# ------------------
demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Upload Audio (.wav)"),
    outputs=gr.Textbox(label="Prediction"),
    title="Audio Deepfake Detection",
    description=(
        "Detect whether an audio sample is REAL or AI-GENERATED "
        "using MFCC features and a lightweight CNN model."
    ),
)

if __name__ == "__main__":
    demo.launch()
