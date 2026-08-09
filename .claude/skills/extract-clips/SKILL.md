---
name: extract-clips
description: Step 1 of the bird-cam comment-based shorts pipeline (extract -> cut_clips.py -> compile_shorts.py). Read a day's merged chat log (marugoto/MMDD_output.txt) and pick short-clip candidates, writing marugoto/MMDD_clips.json for review before cutting. Use whenever the user asks to make/build a short (compilation) video for a date from the bird-cam comments, e.g. "0807のショート結合動画を作って", "0806のコメントからクリップ候補を作って", "0807の切り抜き作って", or "/extract-clips 0806" -- this is the required first step even if the user only asked for the final combined short, since cut_clips.py and compile_shorts.py both depend on the MMDD_clips.json this produces.
---

# コメントログからのクリップ候補抽出（ショート生成パイプラインの①）

このリポジトリには、鳥の巣ライブカメラのチャットコメントから見どころを切り抜き、
字幕焼き込み→スワイプ転換で結合した「ショート」動画を作る3ステップのパイプラインがある
（詳細は `readme.md` の「🎬 コメントベースのショート自動生成」参照）。

```
① このスキル（extract-clips）  marugoto/MMDD_output.txt → marugoto/MMDD_clips.json
② python cut_clips.py MMDD     marugoto/MMDD_clips.json → marugoto/shorts/MMDD_<id>.mp4
③ python compile_shorts.py MMDD marugoto/shorts/MMDD_<id>.mp4 → marugoto/shorts/MMDD_short.mp4
```

ユーザーが「MMDDのショート（結合）動画を作って」のように**最終成果物だけ**を頼んできた場合でも、
`marugoto/MMDD_clips.json` が無ければ必ずこのスキルを先に実行し、そのあと②③をこの順で実行すること
（②③はコマンドを打つだけで判断は不要）。`marugoto/MMDD_output.txt` が無い場合
（＝その日の配信がまだ終わっていない/`smart_bird_pipeline.py`が未処理）は、それを先にユーザーに伝える。

このスキル自体が担当するのは①の「文章の意味判断」だけ。
判断が要る部分（挨拶かどうか、範囲か単発か、余白をどれくらい取るか、返信が同じ出来事を指しているか）
はここで人間の代わりに行い、実際のファイル・秒数へのマッピングと動画の切り出し・字幕焼き込みは
`cut_clips.py`（決定的スクリプト、別ステップ）に任せる。

引数として日付 MMDD を受け取ります（例: `0806`）。引数が無ければユーザーに尋ねてください。

## 入力

`marugoto/{MMDD}_output.txt`
各行は `[HH:MM:SS] @author: message` 形式。`[HH:MM:SS]` はコメント投稿時刻（壁時計時刻）。

## タスク

各行を読み、以下のルールで `clips` / `skipped` に振り分けてください。

### スキップ対象

挨拶・相槌・意味内容のないコメント（おはようございます、こんにちは、どうも、絵文字のみ、空メッセージ等）は
`skipped` に理由付きで入れる。本文中に時刻表現が無くても具体的な実況内容があるコメントは
スキップせず、後述「時刻表現の無い実況コメント」のルールで拾うこと（**なるべく多めに拾う**方針）。

### 採用対象の判定

message 本文中に時刻表現（`H:MM`, `H:MM:SS`, `H時MM分` など）が含まれるコメントを対象にする。
`[HH:MM:SS]` の投稿時刻ではなく、**本文中に書かれた時刻**を使うこと
（投稿時刻は本文中の時刻の数分後になることが多い＝視聴者が「今の」出来事を報告している）。

1. **秒単位の単発時刻**（例: `14:45:45 おちりのアップ`）
   → `confidence: high`。前後に余白を付けて `cut_start`/`cut_end` を決める。
   デフォルトは前3秒・後8秒。ただし文意が「ちらっと」「一瞬」など瞬間的なら短め（前2秒・後4秒）、
   「運び出し」「アピール」「給餌」など動作が続きそうな語なら長め（後10〜15秒）に調整してよい。

2. **範囲指定**（区切り記号: `〜`, `～`, `ー`, "から〜まで", "の間"）
   例: `10:44:58～45:53 バッタさん持ち帰り`（終端の"45:53"は始端と同じ時なので10:45:53と解釈）
   → `confidence: high`。範囲の前後に2秒ずつ余白を足す。
   範囲が15分を超えるなど明らかに長すぎる場合は `low` + noteで理由を書き、
   `cut_end` は `cut_start` + 30秒程度に丸める。
   `〜`の後に終端時刻が書かれていない場合（例: `15:33:39～雛ちゃんチラリ`）は範囲ではなく単発として扱う。

3. **並列した複数時刻**（区切り: "と"、読点で繋がれた別々の時刻）
   例: `16:12:00と16:41からお口見えたよ!`
   → 1コメントから複数の `clips` エントリを作る（それぞれ単発時刻として1.のルールを適用）。

4. **分単位まで**（秒の記載がない、例: `9:36 入口に来てくれた`）
   → 書かれた分をそのまま`:00`（時報として発言された時刻）とみなし、前5秒・後25秒（計30秒）を切り出す。
   例: `9:36` なら `cut_start=9:35:55`, `cut_end=9:36:25`。1分まるごと(60秒)は長すぎるため避ける。
   `confidence: low`。

### 時刻表現の無い実況コメント

本文に時刻が書かれていなくても、「今見ている様子」を具体的に描写した実況コメント
（例: `頭の栗饅頭の黒い部分が出来てきたね`、`おちりはモフモフになってる`、
`暗闇から雛パッキン`、`あご乗せきゃわわ〜ですね` のように、見えている特徴・仕草・状態の変化を
具体的に指しているもの）は積極的に拾う。**ここは既存クリップの「返信」候補になれなかった
（数分以内の反応ではない等）というだけの理由で切り捨てないこと** — 単独の実況コメントとして
このルールの対象になる。
単なる相槌・感想のみ（「かわいい」「見えた」「ねてる」だけ等、1〜2語で何が起きているか
具体的に分からないもの）は対象外。

→ `confidence: low` 固定。投稿時刻ブラケット `[HH:MM:SS]` から**12秒引いた時刻**を推定タイミングとし、
そこに1.と同じ前後余白（デフォルト前3秒・後8秒）を適用する。
**注意**: このブラケット自体、動画ファイルの切り出しと同じ「配信開始時刻＋オフセット」の単純計算で
生成されており、配信中の見えない停止/再接続の影響で数秒〜まれに分単位でズレることが実測で分かっている。
そのため必ず `confidence: low` とし、noteに「ブラケット由来のため誤差の可能性あり」と明記すること。

### 返信のマージ（AIでないと判断できない部分）

採用したコメントの後、数分以内に**同じ出来事について言及している**返信・反応コメントがあれば
（例: 「時報ありがとうございます」に続けてその出来事への補足、「まーめさん時報ありがとうございます」等）、
それを独立した `clips`/`skipped` エントリにはせず、元クリップの `replies` 配列に含める。
単なる相槌（「見えました」「かわいい」だけ等、内容が薄いもの）や無関係な話題への移行は含めない。
判断に迷ったら含めない（`skipped`に理由を書く）方を選ぶ。

## caption の生成

- `caption_line1`: 元コメントの `"@author:"` の部分をそのまま使う
- `caption_line2`: message本文をそのまま使う（時刻表記が含まれていてもよい、削らない）
- `replies[].author` / `replies[].text`: 返信コメントの author と本文
- `replies[].reason`: なぜこれを同じ出来事への返信と判断したか

## title（オープニング用タイトル）の生成

`marugoto/{MMDD}_output.txt` の全コメント（`skipped`含む）の中から、その日いちばん際立った・映える
出来事を1つ選び、10文字以内の短いタイトルにする（例: 「おちりモフモフ」「パタパタ練習中」）。
挨拶等の意味のないコメントは対象外。長い説明文ではなく、キャッチーな体言止め・短句にすること。
このタイトルは②③でオープニング動画に自動で焼き込まれる（`compile_shorts.py`が`clips.json`の
`title`フィールドを読む）。

## 出力フォーマット

`marugoto/{MMDD}_clips.json` に以下のスキーマで書き出す。

```json
{
  "date": "MMDD",
  "title": "10文字以内のタイトル",
  "clips": [
    {
      "id": "MMDD-01",
      "source_comment": "[HH:MM:SS] @author: 元コメント全文",
      "cut_start": "HH:MM:SS",
      "cut_end": "HH:MM:SS",
      "confidence": "high|medium|low",
      "note": "判断理由。low/mediumの場合は特に理由を書く",
      "caption_line1": "@author:",
      "caption_line2": "元コメントの本文",
      "replies": [
        {
          "source_comment": "[HH:MM:SS] @author2: 返信コメント全文",
          "author": "@author2",
          "text": "返信本文",
          "reason": "同じ出来事への返信と判断した理由"
        }
      ]
    }
  ],
  "skipped": [
    {"source_comment": "[HH:MM:SS] @author: 本文", "reason": "スキップ理由"}
  ]
}
```

`replies` は該当が無ければキー自体を省略してよい。

不明点や判断に迷ったコメントは `confidence: low` + `note` に理由を書き、
削除はせず必ず `clips` か `skipped` のどちらかに入れること（黙って無視しない）。

## 完了後

書き出したら、採用した`clips`を「コメント → 切り出し範囲」の表で簡潔に報告し（`skipped`件数もひとこと添える）、
`confidence: low` が付いたものが分かるようにしてください。

- ユーザーの依頼が「クリップ候補を作って」など**①だけ**なら、ここで止めてレビューしてもらう。
- ユーザーの依頼が「ショート（結合）動画を作って」など**最終成果物**なら、報告した上でそのまま
  `python cut_clips.py {MMDD}` → `python compile_shorts.py {MMDD}` を続けて実行し、
  最後に `marugoto/shorts/{MMDD}_short.mp4` ができたことを報告する（許可待ちで止まらない）。
  ルールは会話を重ねて調整済みなので、通常はこの一気通貫で問題ない。
