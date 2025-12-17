# Sound Change Explainer

設備や環境音の「通常音」と「比較音」を入力し、
数値差分とその意味を人が理解できる言葉で説明するツールです。

## コンセプト
- 異常検知ではなく「変化の説明」
- ブラックボックスAIを使わない
- 現場判断の補助を目的とする

## 構成
- analyze_sound.py : 音解析と比較
- interpret.py     : 数値→意味の解釈ルール

## 使い方
1. data/ に normal.wav と target.wav を配置
2. `python analyze_sound.py`

※ data/*.wav はリポジトリには含めません
