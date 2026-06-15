import os
import sys
import subprocess
import time
import requests
from datetime import datetime, timedelta, timezone

# ==================== Settings ====================
CHANNEL_HANDLE = "@Take1bit"
FAST_BIRD_PIPELINE = "fast_bird_pipeline.py"

# Force H.264 (avc1) at 720p max to avoid AV1/VP9 codec mismatch in concat
YTDLP_FORMAT = "bv*[vcodec^=avc][height<=720]+ba[ext=m4a]/bv*[vcodec^=avc]+ba[ext=m4a]/b[ext=mp4]"
# ====================================================

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

def find_target_date_and_videos(api_key, channel_id):
    """直近のライブ配信を解析し、次に処理すべき『日付』と『動画リスト』を自動決定する"""
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "id", "channelId": channel_id, "order": "date",
        "type": "video", "maxResults": 15, "key": api_key
    }
    search_res = requests.get(search_url, params=search_params).json()
    video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]

    if not video_ids:
        return None, []

    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_params = {
        "part": "snippet,liveStreamingDetails",
        "id": ",".join(video_ids), "key": api_key
    }
    video_res = requests.get(video_url, params=video_params).json()

    # 動画を【実際の配信日(MMDD)】ごとにグループ化する
    grouped_livestreams = {}
    
    for item in video_res.get("items", []):
        if "liveStreamingDetails" not in item:
            continue
            
        start_time_utc = item["liveStreamingDetails"].get("actualStartTime") or item["liveStreamingDetails"].get("scheduledStartTime")
        if not start_time_utc:
            continue
            
        jst_dt = utc_to_jst(start_time_utc)
        mmdd = jst_dt.strftime("%m%d")
        
        if mmdd not in grouped_livestreams:
            grouped_livestreams[mmdd] = []
            
        grouped_livestreams[mmdd].append({
            "id": item["id"],
            "title": item["snippet"]["title"],
            "status": item["snippet"]["liveBroadcastContent"],
            "start_time": jst_dt,
            "url": f"https://www.youtube.com/watch?v={item['id']}"
        })

    # 日付の新しい順（今日 ➔ 昨日 ➔ 一昨日...）に未処理のターゲットを探す
    sorted_dates = sorted(grouped_livestreams.keys(), reverse=True)
    
    for mmdd in sorted_dates:
        # すべての動画を配信開始時間が古い順（朝➔昼）にソート
        grouped_livestreams[mmdd].sort(key=lambda x: x["start_time"])
        
        # 成果物（marugoto/MMDD_output.mp4）がすでに存在するかチェック
        final_output = f"marugoto/{mmdd}_output.mp4"
        if os.path.exists(final_output):
            # 既に1本の動画になっている日付は「処理済み」として完全スルー
            continue
            
        # 成果物がまだ無い日付を見つけたら、それを「今回の処理ターゲット」に決定！
        return mmdd, grouped_livestreams[mmdd]
        
    return None, []

def preflight_check(video_list, target_dir, mmdd):
    """Check all video formats BEFORE downloading anything. Returns True if all OK."""
    print(f"\n  🔍 全動画のフォーマットを事前確認中...")
    all_ok = True
    for idx, video in enumerate(video_list, 1):
        expected_filepath = os.path.join(target_dir, f"{mmdd}-{idx}.mp4")
        if os.path.exists(expected_filepath):
            print(f"  ✅ [{idx}/{len(video_list)}] 保存済み（チェック不要）")
            continue
        preflight_cmd = [
            "yt-dlp", "-f", YTDLP_FORMAT, "--print",
            "%(format_id)s | %(vcodec)s | %(height)sp",
            video["url"]
        ]
        try:
            result = subprocess.run(preflight_cmd, capture_output=True, text=True, check=True)
            fmt_info = result.stdout.strip()
            if "av01" in fmt_info or "vp9" in fmt_info or "vp09" in fmt_info:
                print(f"  ❌ [{idx}/{len(video_list)}] {fmt_info} ← H.264以外が選択されました！")
                all_ok = False
            else:
                print(f"  ✅ [{idx}/{len(video_list)}] {fmt_info}")
        except subprocess.CalledProcessError:
            print(f"  ❌ [{idx}/{len(video_list)}] フォーマット確認失敗")
            all_ok = False
    return all_ok

def main():
    api_key = load_api_key()
    if not api_key: return
    
    channel_id = get_channel_id(api_key, CHANNEL_HANDLE)
    if not channel_id: return

    print("🦅 インテリジェンス・自動化パイプラインを起動しました。")
    print("※実行時刻に関わらず、未処理の日付を自動検知して処理します。")
    print("-" * 75)

    while True:
        # 1. 次に処理すべき日付と動画リストを自動判定
        mmdd, video_list = find_target_date_and_videos(api_key, channel_id)
        
        if not mmdd:
            print("🟢 すべての日付の動画が処理済み（marugotoに存在）です。やることがありません。")
            break
            
        print(f"\n🎯 ターゲット日程を自動検出: 【{mmdd[:2]}月{mmdd[2:]}日】 (未処理枠数: {len(video_list)}本)")
        
        # 2. ターゲット日の配信が「すべて終了しているか」をチェック
        all_finished = True
        for idx, video in enumerate(video_list, 1):
            print(f"  {idx}本目 ({video['start_time'].strftime('%H:%M')}開始): 状況={video['status']}")
            if video["status"] != "none":
                all_finished = False

        # もし1本でも配信中（live）や配信前（upcoming）があれば、終了するまで5分待機
        if not all_finished:
            print(f"⏳ 【{mmdd[:2]}月{mmdd[2:]}日】の配信がまだ継続中、またはYouTube側で処理中です。")
            print("   配信が完全に終了してアーカイブ化されるまで、5分間待機（スリープ）します...")
            time.sleep(300)
            continue
            
        # ====================================================
        # ★ターゲット日のすべての動画がDL可能（none）になったら進む
        # ====================================================
        print(f"\n🚀 【{mmdd[:2]}月{mmdd[2:]}日】の全動画がDL可能になりました。一気に処理します。")
        
        # 動画の配信日に基づいた正確なフォルダを作成 (例: input/0614)
        target_dir = f"input/{mmdd}"
        os.makedirs(target_dir, exist_ok=True)
        
        # 3. Phase 1: Preflight - verify all formats are H.264 before downloading
        if not preflight_check(video_list, target_dir, mmdd):
            print(f"\n  🛑 H.264以外のコーデックが検出されたため、処理を中止します。")
            print(f"     YouTube側の処理が完了していない可能性があります。時間を置いて再実行してください。")
            break

        # 4. Phase 2: All clear -> download
        for idx, video in enumerate(video_list, 1):
            expected_filename = f"{mmdd}-{idx}.mp4"
            expected_filepath = os.path.join(target_dir, expected_filename)
            
            if os.path.exists(expected_filepath):
                continue
                
            print(f"  📥 ダウンロード開始 [{idx}/{len(video_list)}]: {expected_filepath}")
            command = [
                "yt-dlp",
                "-f", YTDLP_FORMAT,
                "-o", expected_filepath,
                video["url"]
            ]
            try:
                subprocess.run(command, check=True)
                print(f"  ✅ ダウンロード完了: {expected_filename}")
            except subprocess.CalledProcessError:
                print(f"  ❌ {expected_filename} のDL中にエラーが発生しました。処理を中止します。")
                break

        # 5. 既存の動体検知・結合スクリプトを自動キック
        print(f"\n🎬 すべての動画ファイルが揃いました。")
        print(f"🚀 {FAST_BIRD_PIPELINE} {mmdd} を実行します。CPUフルパワー解放...")
        print("=" * 75)
        
        try:
            subprocess.run([sys.executable, FAST_BIRD_PIPELINE, mmdd], check=True)
            print("\n" + "=" * 75)
            print(f"🏁 🎉 【{mmdd[:2]}月{mmdd[2:]}日】の処理が完全自動で終了しました！")
            print(f"成果物 ➔ marugoto/{mmdd}_output.mp4")
        except subprocess.CalledProcessError:
            print(f"\n❌ {FAST_BIRD_PIPELINE} の実行中にエラーが発生しました。")
            
        # 1日分の処理が終わったら安全のために一度終了（また次の日に手動で叩けば、その日の分をやってくれます）
        break

if __name__ == "__main__":
    main()