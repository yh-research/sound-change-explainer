import numpy as np
import librosa


def extract_basic_features(y, sr):
    """
    音声信号から、変化の兆候を捉えるための基本的な特徴量を抽出する。

    本関数は異常検知を目的とせず、
    「通常音と比較したときに、何がどう変わったか」を説明するための
    解釈可能な指標のみを算出する。

    Parameters
    ----------
    y : np.ndarray
        モノラル音声信号（振幅は正規化済みを想定）
    sr : int
        サンプリング周波数 [Hz]

    Returns
    -------
    dict
        以下の特徴量を含む辞書
        - rms        : 音の平均エネルギー（音量・負荷の指標）
        - centroid   : スペクトル重心（音質の高音化・低音化の指標）
        - hf_ratio   : 高周波/低周波エネルギー比（摩耗・異音傾向の指標）
    """
    # RMS(Root Mean Square)
    # 音全体のエネルギー量を表す指標.
    # 通常音と比較して増加している場合,
    # ・負荷の増加
    # ・接触, 摩擦の増加
    # ・全体的な騒音レベルの上昇
    # などが考えられる.
    rms = float(np.mean(librosa.feature.rms(y=y)))

    # スペクトル重心(Spectral Centroid)
    # 周波数成分のエネルギー分布の「重心」を表す.
    # 値が大きいほど高周波成分が支配的で,
    # 摩擦や表面状態の変化, 金属的な音への変化を示唆する.
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

    # 短時間フーリエ変換(STFT)
    # 音声を時間x周波数の成分に分解し,
    # 周波数帯域ごとのエネルギー分布を取得する.
    S = np.abs(librosa.stft(y))

    # 周波数軸(Hz)
    freeqs = librosa.fft_frequencies(sr=sr)

    # 低周波帯(例: 1kHz未満)
    # 回転音や周期的な構造音が主に含まれる帯域.
    lf = float(S[freeqs < 1000].mean())

    # 高周波帯（例：3kHz超）
    # 摩耗,衝突,ガタつきなどの微小な異音が現れやすい帯域.
    hf = float(S[freeqs > 3000].mean())

    # 高周波／低周波エネルギー比
    # 高周波成分が相対的に増加しているかを示す指標。
    # 比率の上昇は、異音や摩耗の初期兆候を示唆する。
    hf_ratio = hf / (lf + 1e-6)

    # 特徴量を辞書として返す。
    # 通常音との差分を取ることで、
    # 音の変化を定量的かつ言語的に説明する材料となる。
    return {
        "rms": rms,
        "centroid": centroid,
        "hf_ratio": hf_ratio
    }
