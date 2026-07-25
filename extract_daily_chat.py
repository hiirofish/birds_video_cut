import os
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
# ==============================================

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

def find_target_videos(api_key, channel_id, target_mmdd):
    """指定された日付(MMDD)の全動画をAPIで自動検索する"""
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "id", "channelId": channel_id, "order": "date",
        "type": "video", "maxResults": 30, "key": api_key
    }
    search_res = requests.get(search_url, params=search_params).json()
    video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]

    if not video_ids: return []

    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_params = {
        "part": "snippet,liveStreamingDetails",
        "id": ",".join(video_ids), "key": api_key
    }
    video_res = requests.get(video_url, params=video_params).json()

    target_videos = []
    for item in video_res.get("items", []):
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
                "url": f"https://www.youtube.com/watch?v={item['id']}"
            })

    # 古い順にソート
    target_videos.sort(key=lambda x: x["start_time"])
    return target_videos

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

    if not all_messages:
        print("\n❌ コメントが1件も取得できなかったため、テキストファイルは作成しませんでした。")
        return

    # ★ここがミソ：複数枠のコメントを「実際の時計時刻」で一列にソートする
    all_messages.sort(key=lambda x: x[0])

    # 最終出力ファイル（動画が 0722_output.mp4 なので 0722_output.txt にする）
    os.makedirs(FINAL_OUT_DIR, exist_ok=True)
    output_txt_path = os.path.join(FINAL_OUT_DIR, f"{mmdd}_output.txt")

    with open(output_txt_path, "w", encoding="utf-8") as wf:
        for _, text in all_messages:
            wf.write(text + "\n")

    print("\n" + "=" * 65)
    print(f"🎉 統合完了！動画ファイルと完全に並ぶ名前で保存しました。")
    print(f"💾 成果物 -> {output_txt_path} （総コメント数: {len(all_messages)}件）")

if __name__ == "__main__":
    main()
