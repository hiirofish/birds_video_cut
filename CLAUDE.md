# birds_video_cut

コシアカツバメの巣のライブ配信（YouTube `@Take1bit`）を録画・編集・投稿する自動化リポジトリ。

## いちばん多い依頼：その日の分を処理する

`daily-shorts` スキルを使うこと。配信のDLから完成動画のYouTube公開まで一気通貫で処理する手順書になっている。

**毎日のルーチンなので依頼は曖昧でよく、確認を取らずに最後まで実行すること（④の公開アップロードも含む）。**
次はすべて同じ依頼。**疑問形・省略形も「可否の質問」ではなく実行依頼**として扱う。

- 「本日分をやってみて」「今日の分お願い」「0809の作業して」「溜まってる日付を処理して」
- 「今日の取り分は？」「今日のある？」「0826の動画は作れる？」「今日どう？」

「作れる？」「できる？」と聞かれたら、**やってから結果を報告する**のが答え。
素材が無いのは「未処理」というだけで、着手しない理由にならない。
⓪に1時間かかることも、配信がまだ終わっていない可能性も、確認を取る理由にならない
（`smart_bird_pipeline.py` が未処理日の自動検出も配信終了検知も持っている）。
**着手前に止まってよいのは、⓪が `⏭ スキップ(配信未終了)` を返したときだけ。**

```
⓪ python smart_bird_pipeline.py                  配信DL＋動体検知＋チャットログ統合（引数不要・未処理日を自動検出）
① extract-clips スキル                            チャットログ → クリップ候補 JSON（コメントの意味判断＝AI）
② python cut_clips.py MMDD                        切り出し＋字幕焼き込み
③ python compile_shorts.py MMDD                   OP/EDつきで結合 → marugoto/MMDD_short_DAY<n>_<title>.mp4
④ python upload_videos.py MMDD --privacy public   まるごと版・ショート版を公開し、それぞれの再生リストに追加（通知はショート版のみ）
```

⓪は**1時間近くかかる**ので必ずバックグラウンド実行＋終了待ちにする（詳細は `daily-shorts` スキル）。
**チャットが揃っていない日（配信当日に⓪を回した日）は①〜④に進まない**。コメントが欠けたショートが公開されてしまう（詳細はスキル）。
チャットログ（`marugoto/MMDD_output.txt`）が既にあって、ショート動画だけ作る場合は `extract-clips` スキル。

## スクリプトの役割

| ファイル | 役割 |
|---|---|
| `smart_bird_pipeline.py` | 全自動パイプライン（推奨）。配信終了検知→DL→動体検知→結合→チャットログ統合 |
| `fast_bird_pipeline.py` | 手動DL済みの場合の本番用。再エンコードなしで高速 |
| `motion_detector.py` | 旧版（切り出しのみ） |
| `cut_clips.py` / `compile_shorts.py` | コメントベースのショート生成（②③） |
| `upload_videos.py` | YouTubeへアップロードして再生リストに追加（④）。まるごと版→「動体検知」、ショート版→「2年目」 |
| `extract_daily_chat.py` | チャットログ抽出 |

## ディレクトリ

`input/`, `work/`, `marugoto/`, `logs/` は**すべて`.gitignore`済み**（動画・ログの実体はコミットしない）。

- `input/MMDD/` … DLした元動画
- `marugoto/MMDD_output.mp4` / `_output.txt` … 動体検知ダイジェスト（＝まるごと版）／統合チャットログ
- `marugoto/MMDD_clips.json` … ①のクリップ候補（人がレビュー・修正できる）
- `marugoto/MMDD_short_DAY<n>_<title>.mp4` … **成果物**のショート動画。
  3分を超える日は `_part1_` / `_part2_` の2本に自動分割（後半のタイトルは`clips.json`の`title_part2`）
- `marugoto/MMDD_upload.json` … ④の記録（再実行しても二重投稿しないため）
- `marugoto/shorts/` … ②が作る個別クリップ（テンポラリ）
- `sozai/` … OP/ED用の素材、`date_list.txt`（日付→DAY数の対応表）、④の概要欄・タグのテンプレート
  （`description_short.txt` / `description_full.txt` / `upload_tags.txt`）
- `uploader_credentials.json`（OAuthクライアント）/ `uploader_token.json`（④の認証トークン）… リポジトリ直下、`.gitignore`済み

## 設計方針

**「意味の判断が要る部分」と「機械的正確さが要る部分」を分ける。**
LLMに秒数計算をさせるとミスるので、AIは「どのコメントを採用するか・余白をどれくらい取るか・
返信が同じ出来事か・タイトルは何か」だけを判断し、ファイルと秒数へのマッピングやffmpeg実行は
スクリプトに任せる。

## ハマりどころ

- **配信の時刻はズレる。** ログの開始時刻は分精度な上、配信中の見えない停止/再接続で映像時間と
  実時間が数分ズレる（実測で約8分の例あり）。`cut_clips.py`は映像に焼き込まれた日時オーバーレイを
  `tesseract`でOCRして補正している。ログの時刻を信じて計算しないこと。
- **チャットリプレイは配信終了から丸1日近く経たないと取れない。** 当日に⓪を回した日はコメントが欠けるので、
  翌日 `python extract_daily_chat.py MMDD` で取り直してから①〜④。
- **シーズンが変わったら `upload_videos.py` の `SEASON` / `PLAYLIST_KEYWORDS` を変える。**
  再生リストはタイトルのキーワードで探すので、そのままだと「2年目」のリストに入り続ける。
- 日本語フォントは `/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc` 決め打ち。
- OCRの切り抜き座標は720x720の映像に合わせた決め打ち。
