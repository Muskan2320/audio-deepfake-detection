import librosa
import numpy as np

def extract_mfcc(
    file_path: str,
    n_mfcc: int = 40,
    max_len: int = 300
):
    """
    Extract MFCC features from an audio file.
    """
    audio, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate to fixed length
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
    else:
        mfcc = mfcc[:, :max_len]

    return mfcc
