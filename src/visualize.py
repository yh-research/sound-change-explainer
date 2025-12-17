import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display


def plot_waveform_compare(y1, y2, sr):

    plt.figure(figsize=(10, 4))
    plt.plot(y1, label="Normal", alpha=0.7)
    plt.plot(y2, label="Target", alpha=0.7)
    plt.legend()
    plt.title("Waveform Comparison")
    plt.tight_layout()
    plt.show()


def plot_spectrogram_compare(y1, y2, sr):
    """
    2つの音声信号(通常音・比較音)のスペクトログラムを並べて表示し,
    周波数構造の違いを視覚的に比較するための関数.

    本関数の目的は,単なる可視化ではなく,
    「どの周波数帯で、どのような変化が起きているか」を
    人が直感的に理解できる形で提示することにある.

    Parameters
    ----------
    y1 : np.ndarray
        通常状態を表すモノラル音声信号
    y2 : np.ndarray
        比較対象となるモノラル音声信号
    sr : int
        サンプリング周波数 [Hz]

    Notes
    -----
    - 両スペクトログラムは必ず同一のカラースケールで描画する.
      これにより、色の違いがそのままエネルギー差を意味する.
    - 対数振幅(dB)表示を用いることで,
      人の聴覚特性に近い感覚で変化を捉えられる.
    """
    # === STFT → 対数振幅（dB）変換 ===
    # 短時間フーリエ変換（STFT）により、
    # 音声信号を時間×周波数の成分に分解する。
    # 振幅をdBスケールに変換することで、
    # 小さな異音や高周波成分の変化を強調する。
    S1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
    S2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
    # === 同一カラースケールの設定（比較の肝） ===
    # 左右で異なるカラースケールを使うと,
    # 見た目上の差が実際のエネルギー差と一致しなくなるため,
    # 両者の最小値・最大値を共有する.
    vmin = min(S1.min(), S2.min())
    vmax = max(S1.max(), S2.max())

    # === 描画 ===
    plt.figure(figsize=(12, 5))
    # 通常音のスペクトログラム
    # 周期構造や支配的な周波数帯が
    # 安定して存在しているかを確認する.
    plt.subplot(1, 2, 1)
    librosa.display.specshow(S1, sr=sr, x_axis="time",
                             y_axis="hz", vmin=vmin, vmax=vmax)
    plt.title("Normal")
    plt.colorbar(format="%+2.0f dB")

    # 比較音のスペクトログラム
    # 高周波成分の増加、帯域の出現,
    # 構造の乱れなどを視覚的に捉える.
    plt.subplot(1, 2, 2)
    librosa.display.specshow(S2, sr=sr, x_axis="time",
                             y_axis="hz", vmin=vmin, vmax=vmax)
    plt.title("Target")
    plt.colorbar(format="%+2.0f dB")

    # 図全体のタイトル
    plt.suptitle("Spectrogram Comparison", fontsize=14)
    plt.tight_layout()
    plt.show()
