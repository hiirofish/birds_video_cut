import os
import re
import sys
import subprocess
import requests
import json
import glob
import argparse
from datetime import datetime, timedelta, timezone

# ==================== 設定 ====================
CHANNEL_HANDLE = "@Take1bit"
FINAL_OUT_DIR = "marugoto"
INPUT_ROOT = "input"
# ==============================================

JST = timezone(timedelta(hours=9))

def load_api_key(filename="config.txt"):
    if not os.path.exists(filename):
        print(f"❌ {filename} が見つかりません。")
        return None
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_channel_id(api_key, handle):
    url = "https://www.googleapis.com/youtube/v3/channels"
    params = {"part": "id", "forHandle": handle, "key": api_key}
    response = requests.get(url, params=params).json()
    items = response.get("items", [])
    return items[0]["id"] if items else None

def utc_to_jst(utc_str):
    utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(timezone(timedelta(hours=9)))

def get_uploads_playlist_id(api_key, channel_id):
    url = "https://www.googleapis.com/youtube/v3/channels"
    params = {"part": "contentDetails", "id": channel_id, "key": api_key}
    items = requests.get(url, params=params, timeout=30).json().get("items", [])
    if not items:
        return None
    return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]


def list_recent_video_ids(api_key, channel_id, limit=120):
    """Return the newest `limit` uploads of the channel, newest first.

    search.list is not dependable for this: its index lags behind, and on
    2026-09-18 it returned only nine videos for this channel, which silently
    dropped 0914's morning slot (that day was then built from the evening
    archive alone). The uploads playlist is exhaustive, and it draws on the
    10,000-unit/day shared pool at 1 unit per page rather than search.list's
    own 100-calls-a-day bucket.
    """
    playlist_id = get_uploads_playlist_id(api_key, channel_id)
    if not playlist_id:
        return []
    url = "https://www.googleapis.com/youtube/v3/playlistItems"
    video_ids, token = [], None
    while len(video_ids) < limit:
        params = {"part": "contentDetails", "playlistId": playlist_id,
                  "maxResults": 50, "key": api_key}
        if token:
            params["pageToken"] = token
        res = requests.get(url, params=params, timeout=30).json()
        video_ids.extend(i["contentDetails"]["videoId"] for i in res.get("items", []))
        token = res.get("nextPageToken")
        if not token:
            break
    return video_ids[:limit]


def find_target_videos(api_key, channel_id, target_mmdd):
    """指定された日付(MMDD)の全動画をAPIで自動検索する"""
    video_ids = list_recent_video_ids(api_key, channel_id, limit=200)
    if not video_ids: return []

    # videos.list caps at 50 ids per request.
    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_items = []
    for i in range(0, len(video_ids), 50):
        video_params = {
            "part": "snippet,liveStreamingDetails,contentDetails",
            "id": ",".join(video_ids[i:i + 50]), "key": api_key
        }
        video_items.extend(requests.get(video_url, params=video_params).json().get("items", []))

    target_videos = []
    for item in video_items:
        if "liveStreamingDetails" not in item: continue

        start_time_utc = (item["liveStreamingDetails"].get("actualStartTime")
                          or item["liveStreamingDetails"].get("scheduledStartTime"))
        if not start_time_utc: continue

        jst_dt = utc_to_jst(start_time_utc)
        mmdd = jst_dt.strftime("%m%d")

        if mmdd == target_mmdd:
            target_videos.append({
                "id": item["id"],
                "title": item["snippet"]["title"],
                "start_time": jst_dt,
                "duration_sec": parse_iso8601_duration(
                    item.get("contentDetails", {}).get("duration", "")),
                "url": f"https://www.youtube.com/watch?v={item['id']}"
            })

    # 古い順にソート
    target_videos.sort(key=lambda x: x["start_time"])
    return target_videos

def parse_iso8601_duration(text):
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", text or "")
    if not m:
        return 0
    d, h, mi, s = (int(g or 0) for g in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + s

# ---------------------------------------------------------------- 動画コメント
# スマホからだとチャット欄が見つけにくく、ライブ中でも動画ページの「コメント」に
# 書き込む人がいる。これはチャットリプレイには入らないので別に取る。
# 投稿時刻は「書き込んだ時刻」で場面の時刻ではない（数時間〜翌日のこともある）ため、
# 使えるのは本文に時刻が書かれたものだけ。区別できるよう _output.txt の末尾に別枠で書く。

VIDEO_COMMENT_HEADER = ("===== 動画コメント（配信アーカイブの動画ページへの書き込み。"
                        "[ ]内は書き込んだ日時で、場面の時刻ではない） =====")

# 「38:25」「1:05:10」のような時刻表記。全角コロンも拾う
TIME_TOKEN_RE = re.compile(r"(?<![\d:：])(\d{1,2})[:：](\d{2})(?:[:：](\d{2}))?(?![\d:：])")

# 「06:16:50～19:20」の終端は 06:19:20 の省略なので経過時間と読まない
RANGE_END_RE = re.compile(r"[\d:：]\s*[〜～~ー\-]\s*$")

# 時計の時刻として読むときの、枠の前後の許容幅
WALL_CLOCK_SLACK_SEC = 10 * 60


def fetch_video_comments(api_key, video_id):
    """[(published_jst, author, text, is_reply), ...]（古いスレッドから、返信は親の直後）"""
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    threads, token = [], None
    while True:
        params = {"part": "snippet,replies", "videoId": video_id, "maxResults": 100,
                  "textFormat": "plainText", "order": "time", "key": api_key}
        if token:
            params["pageToken"] = token
        res = requests.get(url, params=params, timeout=30).json()
        if "error" in res:
            # コメント無効の動画など。チャットと同じく、取れなくても処理は止めない
            raise RuntimeError(res["error"].get("message", "commentThreads error"))
        for thread in res.get("items", []):
            top = thread["snippet"]["topLevelComment"]
            replies = thread.get("replies", {}).get("comments", [])
            # replies は最大5件しか同梱されないので、足りなければ取り直す
            if thread["snippet"].get("totalReplyCount", 0) > len(replies):
                replies = fetch_all_replies(api_key, top["id"])
            thread_comments = []
            for cm, is_reply in [(top, False)] + [(r, True) for r in replies]:
                s = cm["snippet"]
                thread_comments.append((utc_to_jst(s["publishedAt"]), s["authorDisplayName"],
                                        s["textDisplay"], is_reply))
            threads.append(thread_comments)
        token = res.get("nextPageToken")
        if not token:
            break
    threads.sort(key=lambda t: t[0][0])   # order=time は新しい順で返る
    return [c for t in threads for c in t]


def fetch_all_replies(api_key, parent_id):
    url = "https://www.googleapis.com/youtube/v3/comments"
    replies, token = [], None
    while True:
        params = {"part": "snippet", "parentId": parent_id, "maxResults": 100,
                  "textFormat": "plainText", "key": api_key}
        if token:
            params["pageToken"] = token
        res = requests.get(url, params=params, timeout=30).json()
        replies.extend(res.get("items", []))
        token = res.get("nextPageToken")
        if not token:
            break
    replies.sort(key=lambda r: r["snippet"]["publishedAt"])
    return replies


def elapsed_stamps(text, start_time, duration_sec):
    """本文中の時刻表記のうち、動画の経過時間（例: 夕枠の「38:25」）と読むべきものを
    [(表記, 経過秒), ...] で返す。時計の時刻として枠内に収まるものはそちらを優先する
    （「17:37」は夕枠なら17時37分）。"""
    start_sod = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
    lo = start_sod - WALL_CLOCK_SLACK_SEC
    hi = start_sod + duration_sec + WALL_CLOCK_SLACK_SEC
    found = []
    for m in TIME_TOKEN_RE.finditer(text):
        if RANGE_END_RE.search(text[:m.start()]):
            continue
        a, b, c = int(m.group(1)), int(m.group(2)), m.group(3)
        if c is not None:
            wall, elapsed = a * 3600 + b * 60 + int(c), a * 3600 + b * 60 + int(c)
        else:
            wall, elapsed = a * 3600 + b * 60, a * 60 + b   # H:MM / MM:SS
        if a < 24 and lo <= wall <= hi:
            continue
        if b < 60 and elapsed <= duration_sec:
            found.append((m.group(0), elapsed))
    return found


def elapsed_to_wall(mmdd, idx, video, elapsed):
    """経過時間の位置に焼き込まれた時刻をOCRで読む。読めなければ開始時刻＋経過で概算。"""
    path = os.path.join(INPUT_ROOT, mmdd, f"{mmdd}-{idx}.mp4")
    if os.path.exists(path):
        try:
            import cut_clips  # pytesseract を要するので必要なときだけ
            sod = cut_clips.grab_overlay_time(path, elapsed)
            if sod is not None:
                return f"{sod // 3600:02d}:{sod % 3600 // 60:02d}:{sod % 60:02d}", "映像の焼き込み時刻"
        except Exception:
            pass
    approx = video["start_time"] + timedelta(seconds=elapsed)
    return approx.strftime("%H:%M:%S"), "概算・開始時刻＋経過"


def video_comment_lines(api_key, mmdd, videos, log=print):
    """その日の各枠の動画コメントを _output.txt 用の行にして返す。"""
    lines = []
    for idx, video in enumerate(videos, 1):
        try:
            comments = fetch_video_comments(api_key, video["id"])
        except Exception as exc:
            log(f"  ⚠️ [{idx}/{len(videos)}] 動画コメントを取得できませんでした: {exc}")
            continue
        log(f"  💬 [{idx}/{len(videos)}] 動画コメント {len(comments)} 件")
        for published, author, text, is_reply in comments:
            text = " ".join(text.split())   # 改行を詰めて1行に
            notes = []
            for token, elapsed in elapsed_stamps(text, video["start_time"],
                                                 video.get("duration_sec", 0)):
                wall, how = elapsed_to_wall(mmdd, idx, video, elapsed)
                notes.append(f"{token}={wall}（{how}）")
            note = f" 《動画の経過時間→時刻: {', '.join(notes)}》" if notes else ""
            kind = "返信" if is_reply else "動画コメント"
            lines.append(f"[{kind} {idx}枠目 {published:%m/%d %H:%M}] {author}: {text}{note}")
    return lines


def write_output_txt(out_dir, mmdd, chat_texts, comment_lines):
    """チャット（時刻順）→ 動画コメント の順で書く。どちらも無ければ作らない。"""
    if not chat_texts and not comment_lines:
        return None
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{mmdd}_output.txt")
    with open(path, "w", encoding="utf-8") as wf:
        for text in chat_texts:
            wf.write(text + "\n")
        if comment_lines:
            wf.write("\n" + VIDEO_COMMENT_HEADER + "\n")
            for line in comment_lines:
                wf.write(line + "\n")
    return path

def extract_chat_messages(chat_path, start_time):
    """JSONファイルからチャットメッセージを抽出し、(実際の時刻, 整形テキスト)のリストで返す"""
    messages_list = []
    if not os.path.exists(chat_path): return messages_list

    with open(chat_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                action_wrapper = data.get("replayChatItemAction", {})
                offset_msec = int(action_wrapper.get("videoOffsetTimeMsec", 0))
                
                # 実際の時刻を計算
                real_time = start_time + timedelta(milliseconds=offset_msec)
                time_str = real_time.strftime("%H:%M:%S")
                
                actions = action_wrapper.get("actions", [])
                if not actions: continue
                
                item = actions[0].get("addChatItemAction", {}).get("item", {})
                renderer = item.get("liveChatTextMessageRenderer", {}) or item.get("liveChatPaidMessageRenderer", {})
                
                if renderer:
                    author = renderer.get("authorName", {}).get("simpleText", "Unknown")
                    message_runs = renderer.get("message", {}).get("runs", [])
                    message = "".join([run.get("text", "") for run in message_runs])
                    
                    formatted_text = f"[{time_str}] {author}: {message}"
                    messages_list.append((real_time, formatted_text))
            except:
                continue
    return messages_list

def main():
    parser = argparse.ArgumentParser(description='日別チャットデータ統合・テキスト化スクリプト')
    parser.add_argument('date', help='処理対象の日付（例: 0722）')
    args = parser.parse_args()

    mmdd = args.date.strip('/')
    
    print(f"🦅 日別チャット統合システムを起動しました。対象日: {mmdd}")
    print("-" * 65)

    api_key = load_api_key()
    if not api_key: return

    channel_id = get_channel_id(api_key, CHANNEL_HANDLE)
    if not channel_id: return

    videos = find_target_videos(api_key, channel_id, mmdd)
    if not videos:
        print(f"❌ 【{mmdd}】の動画が見つかりませんでした。")
        return

    print(f"✅ 【{mmdd}】の動画枠を {len(videos)} 本検出しました。チャットデータを収集します。")

    all_messages = []

    for idx, video in enumerate(videos, 1):
        print(f"\n🎬 [{idx}/{len(videos)}] 枠のチャットをDL中: {video['title']}")
        
        # 一時ファイル名
        temp_base = f"temp_chat_{mmdd}_{idx}"
        
        command = [
            "yt-dlp", "--remote-components", "ejs:github",
            "--skip-download",
            "--write-subs",
            "--sub-langs", "live_chat",
            "-o", temp_base,
            video["url"]
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"  ❌ チャットのダウンロードに失敗しました。YouTube側の処理待ちかもしれません。")
            continue

        found_files = glob.glob(f"{temp_base}.*live_chat*.json")
        if not found_files:
            print("  ❌ チャットファイルが生成されませんでした。")
            continue
            
        chat_file = found_files[0]
        
        # メッセージを抽出して全枠ブレンド用のリストに格納
        messages = extract_chat_messages(chat_file, video["start_time"])
        all_messages.extend(messages)
        print(f"  ⚡ {len(messages)} 件のコメントを回収しました。")
        
        # 一時ファイルの削除
        if os.path.exists(chat_file):
            os.remove(chat_file)

    print("\n📝 動画ページのコメントを収集します。")
    comment_lines = video_comment_lines(api_key, mmdd, videos)

    # ★ここがミソ：複数枠のコメントを「実際の時計時刻」で一列にソートする
    all_messages.sort(key=lambda x: x[0])

    # 最終出力ファイル（動画が 0722_output.mp4 なので 0722_output.txt にする）
    output_txt_path = write_output_txt(FINAL_OUT_DIR, mmdd,
                                       [text for _, text in all_messages], comment_lines)
    if not output_txt_path:
        print("\n❌ コメントが1件も取得できなかったため、テキストファイルは作成しませんでした。")
        return

    print("\n" + "=" * 65)
    print(f"🎉 統合完了！動画ファイルと完全に並ぶ名前で保存しました。")
    print(f"💾 成果物 -> {output_txt_path} （チャット: {len(all_messages)}件 / "
          f"動画コメント: {len(comment_lines)}件）")

if __name__ == "__main__":
    main()
