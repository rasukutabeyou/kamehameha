# Beam Live

Windows のローカルカメラ映像を SSH トンネル経由でサーバへ送り、サーバ側でリアルタイム姿勢/手認識を行って、ビーム風の派手なエフェクトを重ねるアプリです。

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

ゲームモードでは各プレイヤーに HP とエネルギーがあり、相手の胸付近の円にビームが当たると一定間隔でダメージが入ります。ビームは一定時間チャージし、エネルギーを消費してから発射でき、発射後も一定時間で尽きます。脇を締めて踏ん張るようなポーズを取るとエネルギーが回復しますが、その間は受けるダメージが増えます。エネルギーが足りている状態で両手を上げるとウルトラ化し、与ダメージ、エネルギーの回復速度、被ダメージが変化します。ビーム同士が先にぶつかっている場合は、そこで遮られるため胸ダメージにはなりません。ブラウザ上の `Reset` で HP、エネルギー、ビーム状態、ウルトラ状態を戻せます。

## 調整

サーバ起動時の環境変数で負荷を調整できます。

```bash
BEAM_WIDTH=960 BEAM_PLAYERS=2 BEAM_DETECTION=0.55 BEAM_TRACKING=0.55 uv run uvicorn app.server:app --host 127.0.0.1 --port 8000
```

- `BEAM_WIDTH`: サーバ内部処理解像度。軽くしたい場合は `720`、画質重視なら `1280`。
- `BEAM_DETECTION`: 検出信頼度。上げると誤検出が減り、検出漏れが増えます。
- `BEAM_TRACKING`: 追跡信頼度。上げると安定寄りになります。
- `BEAM_PLAYERS`: `2` で左右二人モード、`1` で一人モード。`BEAM_PLAYERS=1` ではデフォルトで右側にNPCが出ます。二人モードは Pose を2本走らせるため、一人モードより重くなります。
- `BEAM_NPC`: `1` で一人モードのNPCを有効化、`0` で無効化。デフォルトは `1`。NPC戦はブラウザの `Battle Start` を押してから5秒後に始まります。
- `BEAM_NPC_DIFFICULTY`: NPC難易度。`easy`、`normal`、`hard`。デフォルトは `easy`。`hard` は開始時にNPCがウルトラ状態になります。
- `BEAM_BATTLE_START_DELAY`: `Battle Start` 後にNPCが動き始めるまでの秒数。デフォルトは `5`。
- `BEAM_NPC_COOLDOWN`: NPCの待機秒数。デフォルトは `1.25`。
- `BEAM_NPC_CHARGE`: NPCのチャージ秒数。デフォルトは `1.25`。
- `BEAM_NPC_ATTACK`: NPCの攻撃秒数。デフォルトは `1.15`。
- `BEAM_MAX_HP`: 初期 HP。デフォルトは `100`。
- `BEAM_DAMAGE`: 胸ヒット1回あたりのダメージ。デフォルトは `6`。
- `BEAM_HIT_COOLDOWN`: 連続ダメージ間隔秒。デフォルトは `0.38`。
- `BEAM_CHARGE`: 発射に必要なチャージ秒数。デフォルトは `1.15`。
- `BEAM_DURATION`: ビームを出し続けられる秒数。デフォルトは `2.4`。
- `BEAM_ENERGY_INITIAL`: 初期のエネルギー。デフォルトは `50`。
- `BEAM_ENERGY_MAX`: エネルギーの最大値。デフォルトは `100`。
- `BEAM_ENERGY_COST`: ビーム1回あたりのエネルギー消費。デフォルトは `25`。
- `BEAM_ENERGY_CHARGE_PER_SEC`: エネルギー溜め中の毎秒回復量。デフォルトは `5`。
- `BEAM_ENERGY_CHARGE_DAMAGE_BONUS`: エネルギー溜め中に受ける追加ダメージ。デフォルトは `2`。
- `BEAM_GUARD_DAMAGE_MULTIPLIER`: ガード中に受けるダメージ倍率。`0.45` なら通常の45%。デフォルトは `0.45`。
- `BEAM_ULTRA_ENERGY_COST`: ウルトラ化に必要なエネルギー。デフォルトは `75`。
- `BEAM_ULTRA_DAMAGE_MULTIPLIER`: ウルトラ時の与ダメージ倍率。デフォルトは `2`。
- `BEAM_ULTRA_ENERGY_CHARGE_MULTIPLIER`: ウルトラ時のエネルギー溜め速度倍率。デフォルトは `2`。
- `BEAM_ULTRA_DAMAGE_REDUCTION`: ウルトラ時に受けるダメージ軽減。デフォルトは `1`。
- `BEAM_ULTRA_ENERGY_DRAIN_PER_SEC`: ウルトラ時の毎秒エネルギー消費。デフォルトは `1`。

16GB VRAM の範囲に収めるため、初期実装は GPU メモリを大きく使う深層推論ではなく MediaPipe の軽量モデルを採用しています。効果描画は CPU/OpenCV で完結します。
