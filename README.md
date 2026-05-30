# Kamehameha Live

Windows のローカルカメラ映像を SSH トンネル経由でサーバへ送り、サーバ側でリアルタイム姿勢/手認識を行って、かめはめ波風の派手なエフェクトを重ねるアプリです。

## 構成

- `app/server.py`: FastAPI サーバ。Windows クライアントから JPEG フレームを受け取り、加工済み MJPEG を配信します。
- `app/detector.py`: MediaPipe による姿勢/手ランドマーク検出と発射判定。
- `app/effects.py`: OpenCV/NumPy によるビーム、オーラ、粒子、発光合成。
- `windows_camera_client.py`: Windows 側で実行するカメラ送信クライアント。
- `static/index.html`: ブラウザ表示用ビューア。

## サーバ側セットアップ

```bash
uv sync
uv run uvicorn app.server:app --host 127.0.0.1 --port 8000
```

LAN 内で直接見たい場合は `--host 0.0.0.0` にできます。SSH トンネル前提なら `127.0.0.1` のままで十分です。

## Windows カメラを SSH 経由で使う

Windows の PowerShell で、まずサーバへのローカルフォワードを張ります。

```powershell
ssh -N -L 8000:127.0.0.1:8000 <server-user>@<server-host>
```

別の PowerShell で Windows 側依存関係を入れて、カメラ送信を開始します。

```powershell
py -m pip install opencv-python websockets
py windows_camera_client.py --server ws://127.0.0.1:8000/ws/camera --camera 0 --width 960 --height 540 --fps 24 --quality 78
```

表示は Windows のブラウザで開きます。

```text
http://127.0.0.1:8000/
```

## 撃ち方

両手首が近い状態で胸や腰の前に構え、そこから左右どちらかの方向へ腕を伸ばすと発射状態になります。横向き、斜め向きでも、肩から手首への方向ベクトルを使うため発射方向が追従します。

二人で撃ち合う場合は、画面の左側と右側に一人ずつ入って向かい合ってください。サーバは左右の領域を少し重ねて二人分の姿勢推定を行います。二つのビームが向かい合って交差、または十分近づくと、衝突地点で爆発エフェクトが出ます。

ゲームモードでは各プレイヤーに HP と気があり、相手の胸付近の円にビームが当たると一定間隔でダメージが入ります。ビームは一定時間チャージし、気を消費してから発射でき、発射後も一定時間で尽きます。脇を締めて踏ん張るようなポーズを取ると気が回復しますが、その間は受けるダメージが増えます。気が足りている状態で両手を上げるとスーパーサイヤ人化し、与ダメージ、気の回復速度、被ダメージが変化します。ビーム同士が先にぶつかっている場合は、そこで遮られるため胸ダメージにはなりません。ブラウザ上の `Reset` で HP、気、ビーム状態、スーパーサイヤ人状態を戻せます。

## 調整

サーバ起動時の環境変数で負荷を調整できます。

```bash
KAME_WIDTH=960 KAME_PLAYERS=2 KAME_DETECTION=0.55 KAME_TRACKING=0.55 uv run uvicorn app.server:app --host 127.0.0.1 --port 8000
```

- `KAME_WIDTH`: サーバ内部処理解像度。軽くしたい場合は `720`、画質重視なら `1280`。
- `KAME_DETECTION`: 検出信頼度。上げると誤検出が減り、検出漏れが増えます。
- `KAME_TRACKING`: 追跡信頼度。上げると安定寄りになります。
- `KAME_PLAYERS`: `2` で左右二人モード、`1` で従来の一人モード。二人モードは Pose を2本走らせるため、一人モードより重くなります。
- `KAME_MAX_HP`: 初期 HP。デフォルトは `100`。
- `KAME_DAMAGE`: 胸ヒット1回あたりのダメージ。デフォルトは `6`。
- `KAME_HIT_COOLDOWN`: 連続ダメージ間隔秒。デフォルトは `0.38`。
- `KAME_BEAM_CHARGE`: 発射に必要なチャージ秒数。デフォルトは `1.15`。
- `KAME_BEAM_DURATION`: ビームを出し続けられる秒数。デフォルトは `2.4`。
- `KAME_KI_INITIAL`: 初期の気。デフォルトは `50`。
- `KAME_KI_MAX`: 気の最大値。デフォルトは `100`。
- `KAME_BEAM_KI_COST`: かめはめ波1回あたりの気消費。デフォルトは `25`。
- `KAME_KI_CHARGE_PER_SEC`: 気溜め中の毎秒回復量。デフォルトは `5`。
- `KAME_KI_CHARGE_DAMAGE_BONUS`: 気溜め中に受ける追加ダメージ。デフォルトは `2`。
- `KAME_SUPER_KI_COST`: スーパーサイヤ人化に必要な気。デフォルトは `75`。
- `KAME_SUPER_DAMAGE_MULTIPLIER`: スーパーサイヤ人時の与ダメージ倍率。デフォルトは `2`。
- `KAME_SUPER_KI_CHARGE_MULTIPLIER`: スーパーサイヤ人時の気溜め速度倍率。デフォルトは `2`。
- `KAME_SUPER_DAMAGE_REDUCTION`: スーパーサイヤ人時に受けるダメージ軽減。デフォルトは `1`。
- `KAME_SUPER_KI_DRAIN_PER_SEC`: スーパーサイヤ人時の毎秒気消費。デフォルトは `1`。

16GB VRAM の範囲に収めるため、初期実装は GPU メモリを大きく使う深層推論ではなく MediaPipe の軽量モデルを採用しています。効果描画は CPU/OpenCV で完結します。
