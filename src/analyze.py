import numpy as np
import librosa

from features import extract_basic_features
from visualize import (plot_waveform_compare, plot_spectrogram_compare)
from interpret import interpret


def load_wav(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    y = y / (np.max(np.abs(y)) + 1e-6)
    return y, sr


def main():
    # 入力
    y_normal, sr = load_wav("data/norma.wav")
    y_target, _ = load_wav("data/target.wav")

    # 可視化
    plot_waveform_compare(y_normal, y_target, sr)
    plot_spectrogram_compare(y_normal, y_target, sr)

    # 特徴量
    f_normal = extract_basic_features(y_normal, sr)
    f_target = extract_basic_features(y_target, sr)

    diff = {k: f_target[k] - f_normal[k] for k in f_normal}

    print("=== Feature Difference ===")
    for k, v in diff.items():
        print(f"{k}: {v:.3f}")

    # 解釈
    comments = interpret(diff)

    print("\n=== Interpretation ===")
    for c in comments:
        print("・", c)


if __name__ == "__main__":
    main()
