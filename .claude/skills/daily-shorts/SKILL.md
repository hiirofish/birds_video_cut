---
name: daily-shorts
description: Run the whole bird-cam daily workflow end to end -- smart_bird_pipeline.py (download + motion detection + chat log), the comment-based shorts pipeline (extract-clips -> cut_clips.py -> compile_shorts.py), and finally upload_videos.py, which uploads the day's two videos to YouTube as PUBLIC and adds each to its playlist. Also picks up earlier days that were left waiting for their chat replay. This is the user's STANDING DAILY ROUTINE -- invoke it immediately and run to completion WITHOUT asking for confirmation, the public upload at the end included, and including when the request is vague, elliptical, or phrased as a question. Imperative triggers: "本日分をやってみて", "今日のやつやって", "0809の作業して", "今日の分お願い", "溜まってる日付を処理して". Equally valid triggers, do NOT treat these as yes/no questions: "今日の取り分は？", "今日のある？", "0826の動画は作れる？", "今日どう？" -- 作れる？/できる？/ある？ means DO IT and report the result, never answer with a status report plus a request for permission. Missing input files just mean the day is unprocessed; that is what step ⓪ is for, not a reason to stop and ask. Use for any request that starts before marugoto/MMDD_output.txt exists. If the chat log already exists and the user only wants the short video, use the extract-clips skill instead.
---

# 鳥カメラ 1日分の全自動処理（⓪→①→②→③→④）

「本日分をやってみて」の一言で、配信のDLから完成動画のYouTube公開までを通す手順書。

**このスキルはユーザーの許可待ちで止まらずに最後まで実行すること**（ルールは会話を重ねて調整済み）。
**④の公開アップロードも確認を取らない**（2026-09-11 にユーザーがルーチンへの組み込みを指示）。
毎日のルーチンなので依頼は曖昧なことが多い。「今日の取り分は？」「0826の動画は作れる？」のような
**疑問形・省略形も実行依頼**であって可否の質問ではない。可否を答えるのではなく、やってから結果を報告する。
素材が無いのは「未処理」というだけ（それを解決するのが⓪）で、着手しない理由にならない。
**着手前に止まってよいのは、⓪が `⏭ スキップ(配信未終了)` を返したときだけ。**
①〜④を回さずに止めてよいのは「①〜④の対象日とチャットの確認」の節に当てはまる日だけ。

**作業ディレクトリは必ずリポジトリのルート** (`/home/yoshi/private_dev/birds_video_cut`)。
スクリプトのパスは全部そこからの相対パス。

## 全体像

```
⓪ python smart_bird_pipeline.py     YouTube配信 → marugoto/MMDD_output.mp4（動体検知ダイジェスト＝まるごと版）
                                              → marugoto/MMDD_output.txt（統合チャットログ）
        │
        ├─ 対象日を決め、チャットが揃っているか確認（配信当日の日はここで翌日に持ち越す）
        │
① extract-clips スキル               marugoto/MMDD_output.txt → marugoto/MMDD_clips.json
        │                            （コメントの意味判断＝AIの担当）
② python cut_clips.py MMDD           marugoto/MMDD_clips.json → marugoto/shorts/MMDD_<id>.mp4
        │                            （OCRで時刻補正して切り出し＋字幕焼き込み）
③ python compile_shorts.py MMDD      marugoto/shorts/MMDD_<id>.mp4 → marugoto/MMDD_short_DAY<n>_<title>.mp4
        │                            （OP/EDつき・スワイプ転換で結合。3分超なら_part1_/_part2_の2本）
④ python upload_videos.py MMDD --privacy public
                                     まるごと版 → 再生リスト「【全編記録】動体検知で見る…」
                                     ショート版 → 再生リスト「【2026年・2年目】帰ってきたコシアカツバメ…」（こちらが本動画）
```

チャットリプレイは配信の翌日にならないと取れないので、**毎晩のルーチンは「今日の分は⓪まで、
昨日の分を①〜④」という1日遅れの形になる**のが普通。

`marugoto/` は`.gitignore`済み。成果物は`marugoto/`直下、②の個別クリップは`marugoto/shorts/`（テンポラリ）。
③の出力名にはDAY数と①が付けたタイトルが入る。3分を超える日は自動で2本に分かれ、
後半には`clips.json`の`title_part2`が使われる。④の記録は`marugoto/MMDD_upload.json`。

---

## ⓪ DL＋動体検知＋チャットログ取得

```bash
python smart_bird_pipeline.py
```

**引数は不要**。過去7日以内の「未処理の日付」をAPIで自動検出し、古い順に全部処理する。
だからユーザーが「本日分」と言っても日付を指定する必要はなく、溜まっていれば一緒に片付く。

### 実行上の注意（重要）

- **時間がかかる**（映像の長さ次第。1日2枠・計15時間の映像なら全体1時間前後。2026年9月の実測では
  1日3時間強・2枠で8〜10分だった）。短く済む日でもフォアグラウンドで実行するとツールのタイムアウトに
  引っかかるので、**必ずバックグラウンドで起動し、プロセスの終了を待つ**こと。

  ```bash
  # 起動（ログはスクラッチパッドへ）。PIDを控えておく
  nohup python smart_bird_pipeline.py > <scratchpad>/pipeline.log 2>&1 &
  echo $! > <scratchpad>/pipeline.pid

  # 終了待ち（別途バックグラウンドで）。**PIDで待つこと**
  while kill -0 $(cat <scratchpad>/pipeline.pid) 2>/dev/null; do sleep 30; done
  ```

  待っている間は手を止めずに、持ち越し分の①〜④（下の「対象日」の節）を先に進めてよい。
  `pgrep -f "smart_bird_pipeline.py"` で待ってはいけない。**待機ループ自身のコマンドラインに
  その文字列が含まれるので自分自身にマッチし、パイプラインが終わっても永久に回り続ける**
  （実際に17時間残り続け、次のセッションで「パイプライン実行中」と誤認する原因になった）。
- 進捗は上記ログか `logs/MMDD_pipeline.log` で確認できる。
- 途中で失敗してもDL済みファイルはスキップされるので、**再実行すれば続きから再開**する。

### 結果の読み方

最後に必ずサマリが出る。

- `✅ 成功: ['0809']` → 「①〜④の対象日とチャットの確認」へ進む。
- `⏭ スキップ(配信未終了): ['0809']` → **その日の配信がまだ終わっていない**。
  その日は①以降に進めないので、ユーザーに「配信終了後にもう一度実行が必要」と伝える。
  （配信は夕方〜夜に終わることが多い。朝や日中に「本日分」と言われた場合はこれになりやすい）
  持ち越し分があればそちらは進めてよい。
- `🟢 過去7日分はすべて処理済みです` → ⓪はやることなし。「①〜④の対象日とチャットの確認」で持ち越し分を探す。
- `❌ 失敗: [...]` → `logs/MMDD_pipeline.log`を見て原因を報告する。手動DL用のURLも出力されている。
  - `検証で不合格: 尺が短い` で止まった場合は、配信直後でYouTube側のアーカイブがまだ確定していないことがある
    （0910は当日夜に 1:08:07 / 期待 1:09:39 で落ち、翌日の再実行で通った）。DL済みの枠はスキップされるので、
    次に⓪を回せば続きから処理される。

### 当日分を朝のうちに回したときの落とし穴

未処理判定は `marugoto/MMDD_output.mp4` の有無だけで決まる。だから**朝枠しか終わっていない
当日分を処理すると、その1枠だけで`_output.mp4`が出来てしまい、以降その日付は永久に
「処理済み」として飛ばされる**。チャットはほぼ夜枠にしか付かない（朝枠は`live_chat`トラック
自体が無い日が多い）ので、これをやると**その日のコメントが丸ごと失われる**。

処理後に枠数を確認し、YouTube側の枠数より少なければ`_output.mp4`を退避して、
夜枠終了後にもう一度⓪を回すこと（DL済みの朝枠は自動でスキップされる）。

```bash
ls input/MMDD/          # 実際にDLされた枠
mv marugoto/MMDD_output.mp4 <scratchpad>/backup/   # 足りなければ退避して作り直し
```

---

## ①〜④の対象日とチャットの確認

⓪が終わったら（`🟢` で何もしなかったときも）、①〜④を回す日を決める。
**対象は「`marugoto/MMDD_output.mp4` があるのに `marugoto/MMDD_upload.json` が無い、0910以降の日」**。
⓪で今回処理した日も、前回チャット待ちで止めた日（持ち越し分）も、これで拾える。
0909以前はユーザーが手動でアップ済みなので対象外。

```bash
for f in marugoto/*_output.mp4; do d=$(basename "$f" _output.mp4); [[ "$d" > 0909 && ! -f marugoto/${d}_upload.json ]] && echo "$d"; done
```

YouTubeのチャットリプレイは**配信終了から丸1日近く経たないと取れない**
（夜枠は終了3時間後では取れず、翌日の同じ時間帯なら取れた）。配信当日に⓪を回した日は、
動画は正しく出来てもコメントが0〜数件しか入らない（0件だと `_output.txt` 自体が作られない）。
そのまま進めると**コメントが欠けたショートが公開されてしまう**（公開後の作り直しは削除と上げ直しになる）。
対象日ごとに、チャットログを作った日時を見て判定する。

```bash
ls -l --time-style=+%m/%d_%H:%M marugoto/MMDD_output.txt   # 無ければ「無い」
```

| 状態 | やること |
|---|---|
| 対象日が**今日** | ①〜④に進まない。「チャットが取れる明日の夜に①〜④をやる」と報告して、その日はここまで |
| `_output.txt` が無い、または**対象日の翌日19時より前**に作られている | 今が翌日19時以降なら `python extract_daily_chat.py MMDD` でチャットを取り直してから①へ。まだ19時前なら今回は飛ばして報告 |
| 対象日の翌日19時以降に作られている | そのまま①へ |

- 19時の根拠：夜枠は18時前後に始まって19時過ぎに終わり、終了から約24時間後（0910分を0911の19:12に取得）で取れた。
- 取り直すときは、古い `marugoto/MMDD_clips.json` や `marugoto/MMDD_short_DAY*.mp4` が残っていれば
  `<scratchpad>/backup/` へ退避してから作り直す（タイトルが変わると別名で残るため）。
- 翌日19時以降に取り直しても**コメントが0件**なら、その日はコメントが無かったものとして①〜③を飛ばし、
  ④でまるごと版だけ上げる。

## ① クリップ候補の抽出（extract-clips スキル）

`marugoto/MMDD_output.txt`（チャットログ）が揃ったら、**extract-clips スキルを呼んで**
`marugoto/MMDD_clips.json` を作る。判断ルール（挨拶のスキップ、時刻表現の解釈、ブラケットと映像のズレの実測、
余白の取り方、返信のマージ、オープニング用タイトルの生成）は全部そちらに書いてあるので、ここでは繰り返さない。

このステップだけが「意味の判断」で、AIがやる価値がある部分。②③④は機械的な処理なのでコードに任せる。

## ②③ 切り出しと結合

```bash
python cut_clips.py MMDD        # 数分（クリップ数×OCR探索）
python compile_shorts.py MMDD   # 1分未満
```

判断は不要でコマンドを打つだけ。②で一部のクリップが「時刻を特定できませんでした」と失敗しても、
2本以上残っていれば③は動く。失敗したクリップは件数と理由を報告に含めること。

## ④ YouTubeへアップロード

```bash
python upload_videos.py MMDD --privacy public
```

**`--privacy public` を必ず付ける**（付けないと非公開で上がる。非公開はテスト用）。数十MBなので1〜2分で終わる。

- 上がるもの：
  - まるごと版 `marugoto/MMDD_output.mp4` → 「動体検知」の再生リスト
  - ショート版 `marugoto/MMDD_short[_partN]_DAY<n>_<title>.mp4` → 「2年目」（帰ってきたコシアカツバメ）の再生リスト。part分割の日は2本
- タイトル・概要欄・タグは、これまでの手動投稿の書式をそのまま再現する。文面は
  `sozai/description_short.txt` / `sozai/description_full.txt` / `sozai/upload_tags.txt`。
  ショートかどうかは長さでYouTubeが勝手に判定するので何も指定しない。
- 再生リストは公開日の新しい順で自動並べ替えなので、公開で上げれば先頭に並ぶ。
- **何度実行しても二重投稿にならない**。`marugoto/MMDD_upload.json` に記録して再実行時は残りの手順だけ行い、
  同じタイトルの動画がもう再生リストにあれば（手動でアップ済みの日など）アップしない。
- ショート版が無い日（②で残ったクリップが2本未満など）は警告を出してまるごと版だけ上げる。
  後からショート版を作って再実行すれば、ショート版だけ上がる。
- 先に中身だけ見たいときは `--dry-run`。
- **アップ後に必ず読み直して確認する。** `videos.list` で公開状態、`playlistItems.list` で再生リストの何番目かを見る。
  公開で上げれば1番目に並ぶので、末尾のままなら公開になっていない。非公開ロックされていないかの確認も兼ねる。
  過去にアップした分も、この機会に公開のままか見ておくと早く気づける。

### 認証が切れていたら

`uploader_token.json` が無効だと、スクリプトは認証URLを表示して**ブラウザでの許可を待ち続ける**。
バックグラウンドで起動してログからURLを拾い、ユーザーに開いてもらう（許可はユーザーにしかできない）。

- 待っている間に **`curl localhost:<ポート>` などで様子を見てはいけない**。受け口は1回きりなので、
  それで消費されて `MismatchingStateError` で落ちる（実際にやってしまい、URLを出し直した）。
- ユーザーのブラウザが「このサイトにアクセスできません」になったら、アドレス欄のURL
  （`http://localhost:<ポート>/?state=…&code=…`）を貼ってもらい、そのURLをそのまま `curl` で叩けば続きが進む。

### 注意

- YouTube APIの監査はまだ通っていない。ドキュメント上は「未監査プロジェクトからのアップロードは非公開にロック」だが、
  2026-09-11 の実測ではロックされず公開できた。**上げた動画が「非公開（ロック）」になっていたら報告する**。
- エラー時はスクリプトが理由と対処を表示する（`quotaExceeded` など）。そのまま報告に含める。

---

## 完了報告

最後に以下をまとめて報告する。

- ⓪ 処理した日付、映像の長さ、かかった時間
- ① 採用したクリップ数／スキップ数、AIが付けたタイトル
- ②③ 成功したクリップ数、完成したファイル名と尺（2本に分割された場合はpart1/part2それぞれ）
- ④ アップした動画のURLと入れた再生リスト。上げなかったもの（アップ済み・同じタイトルあり・ショート版無し）はその理由
- ①〜④に進まなかった日とその理由（チャット待ちで明日に持ち越し、など）

複数日が対象だった場合は、日付ごとに①〜④を回して、日付ごとに報告する。
