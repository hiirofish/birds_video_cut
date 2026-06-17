import cv2
import numpy as np
import datetime
import os
import time
import argparse
import shutil
import glob
from collections import deque
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import queue
import threading
import signal
import sys
import re
import tempfile

# ==========================================
# 共通設定＆ユーティリティ
# ==========================================
SOZAI_DIR = "sozai"
OPENING_VIDEO = os.path.join(SOZAI_DIR, "opening_converted.mp4")
ENDING_VIDEO = os.path.join(SOZAI_DIR, "ending_converted.mp4")
DATE_LIST_FILE = os.path.join(SOZAI_DIR, "date_list.txt")
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

def log_message(message, show=True, log_file=None):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{timestamp}] {message}"
    if show:
        print(msg, flush=True)
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
                f.flush()
        except:
            pass

def format_video_time(frame_number, fps):
    total_seconds = int(frame_number / fps) if fps > 0 else 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}_{minutes:02d}_{seconds:02d}"

def get_filename_without_extension(file_path):
    return os.path.splitext(os.path.basename(file_path))[0]

def get_video_files(input_dir):
    exts = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    return sorted(files)

# ==========================================
# 1. 動体検知 ＆ 超高速切り出し
# ==========================================
def process_frame_batch(frames, backSub, threshold, min_area, scale_factor=0.5):
    results = []
    for frame in frames:
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        fgMask = backSub.apply(gray)
        _, thresh = cv2.threshold(fgMask, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=1)
        thresh = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        largest_area = 0
        largest_contour = None

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area:
                largest_area = area
                largest_contour = contour

        effective_min_area = min_area
        if avg_brightness < 15:
            motion_detected = False
        elif 15 <= avg_brightness < 50:
            effective_min_area = min_area * 0.7
            if largest_area > effective_min_area:
                motion_detected = True
        else:
            if largest_area > effective_min_area:
                motion_detected = True

        results.append({
            "motion_detected": motion_detected,
            "area": largest_area,
            "brightness": avg_brightness,
            "largest_contour": largest_contour
        })
    return results

def save_video_with_audio_async(video_path, event, fps, width, height, cap_lock, output_queue):
    try:
        start_time_str = format_video_time(event["start_frame"], fps)
        base_filename = get_filename_without_extension(video_path)
        video_dir = event["video_dir"]

        out_filename = f"{start_time_str}_{base_filename}_motion.mp4"
        out_path = os.path.join(video_dir, out_filename)

        start_seconds = event["start_frame"] / fps
        duration = (event["end_frame"] - event["start_frame"]) / fps

        # -ss を -i の前に置いて input seek（高速）
        # -avoid_negative_ts / -reset_timestamps / +genpts でタイムスタンプを0起点にリセット
        # → concat 時に「OP→本編→交互再生」となるバグを防ぐ
        ffmpeg_cmd = [
            "ffmpeg", "-y", "-nostdin",
            "-ss", f"{start_seconds:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            "-fflags", "+genpts",
            "-reset_timestamps", "1",
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-loglevel", "error",
            out_path
        ]

        with subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, start_new_session=True
        ) as process:
            try:
                stdout, stderr = process.communicate(timeout=60)
                if process.returncode == 0:
                    output_queue.put(f"✅ 切り出し完了: {out_filename}")
                else:
                    output_queue.put(f"❌ 切り出しエラー: {out_filename}\n{stderr.decode(errors='ignore')}")
            except subprocess.TimeoutExpired:
                process.kill()
                output_queue.put(f"⚠️ FFmpegタイムアウト: {out_filename}")
    except Exception as e:
        output_queue.put(f"動画保存エラー: {str(e)}")

def detect_motion_optimized(video_path, detection_dir, video_dir, params):
    os.makedirs(detection_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    log_file = os.path.join(detection_dir, "detection_log.txt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        skip_frames = int(fps * params['skip_initial_seconds'])
        effective_start_frame = skip_frames
        if effective_start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, effective_start_frame)

        backSub = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=24, detectShadows=False)
        frame_count = effective_start_frame
        
        motion_events = []
        current_event = None
        batch_frames = []
        batch_frame_numbers = []

        video_save_executor = ThreadPoolExecutor(max_workers=2)
        output_queue = queue.Queue()
        cap_lock = threading.Lock()

        # 進捗表示用の変数（復活）
        last_progress_percent = -1
        start_process_time = time.time()
        motion_frames = 0
        file_name = os.path.basename(video_path)

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            if frame_count % params['frame_skip'] != 0: continue

            batch_frames.append(frame)
            batch_frame_numbers.append(frame_count)

            if len(batch_frames) >= params['batch_size'] or frame_count >= total_frames - 1:
                results = process_frame_batch(batch_frames, backSub, params['threshold'], params['min_area'])

                for i, result in enumerate(results):
                    current_frame_count = batch_frame_numbers[i]
                    
                    # 進捗の計算と表示（復活）
                    current_progress_percent = int((current_frame_count - effective_start_frame) / (total_frames - effective_start_frame) * 100)
                    if current_progress_percent != last_progress_percent:
                        elapsed = time.time() - start_process_time
                        speed = (current_frame_count - effective_start_frame) / elapsed if elapsed > 0 else 0
                        remaining = (total_frames - current_frame_count) / speed if speed > 0 else 0
                        
                        log_message(f"[{file_name}] 進捗: {current_progress_percent}% ({current_frame_count}/{total_frames}), "
                                    f"速度: {speed:.1f}fps, 残り: {remaining:.1f}秒, 検出: {motion_frames}件", True, log_file)
                        last_progress_percent = current_progress_percent

                    if result["motion_detected"]:
                        motion_frames += 1
                        if current_event is None:
                            current_event = {
                                "start_frame": max(current_frame_count - int(fps * params['pre_seconds']), effective_start_frame),
                                "end_frame": current_frame_count + int(fps * params['post_seconds'])
                            }
                        else:
                            current_event["end_frame"] = current_frame_count + int(fps * params['post_seconds'])

                    if current_event and current_frame_count >= current_event["end_frame"]:
                        motion_events.append(current_event)
                        current_event = None

                batch_frames = []
                batch_frame_numbers = []

            while not output_queue.empty():
                log_message(output_queue.get(), True, log_file)

        if current_event:
            motion_events.append(current_event)

        # イベントのマージ
        merged_events = []
        if motion_events:
            current_merged = motion_events[0]
            for i in range(1, len(motion_events)):
                if motion_events[i]["start_frame"] - current_merged["end_frame"] <= fps * params['merge_threshold']:
                    current_merged["end_frame"] = motion_events[i]["end_frame"]
                else:
                    merged_events.append(current_merged)
                    current_merged = motion_events[i]
            merged_events.append(current_merged)

        # 非同期で動画保存
        save_futures = []
        for event in merged_events:
            event["video_dir"] = video_dir
            save_futures.append(video_save_executor.submit(
                save_video_with_audio_async, video_path, event, fps, width, height, cap_lock, output_queue
            ))
        for future in save_futures: future.result()
        while not output_queue.empty():
            log_message(output_queue.get(), True, log_file)
            
        video_save_executor.shutdown()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return True

def process_video_wrapper(args):
    video_path, params, out_root = args
    import sys
    sys.stdin = open(os.devnull, 'r')
    
    # 各プロセス内でOpenCVが多スレッド化して競合するのを防ぐ
    # （プロセス4 × スレッド2 = 8コア使用、残りはI/O待ち用）
    cv2.setNumThreads(2)
    
    base_name = get_filename_without_extension(video_path)
    detection_dir = os.path.join(out_root, f"detection_{base_name}")
    video_dir = os.path.join(out_root, f"output_{base_name}")
    
    try:
        success = detect_motion_optimized(video_path, detection_dir, video_dir, params)
        return {'status': 'success', 'video': video_path}
    except Exception as e:
        return {'status': 'error', 'video': video_path, 'error': str(e)}

# ==========================================
# 2. 結合処理（コーデック完全一致対応）
# ==========================================
def load_date_list(mmdd):
    mm, dd = int(mmdd[:2]), int(mmdd[2:])
    WIDE = str.maketrans("0123456789", "０１２３４５６７８９")
    date_jp = f"{mm}月{dd}日".translate(WIDE)
    if not os.path.exists(DATE_LIST_FILE): return date_jp, "DAY ??"
    with open(DATE_LIST_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == date_jp: return parts[0], parts[1]
    return date_jp, "DAY ??"

def get_cut_videos(target_dir):
    videos = []
    for root, _, files in os.walk(target_dir):
        if "output_" not in os.path.basename(root): continue
        for f in files:
            if f.endswith(".mp4") and "Zone.Identifier" not in f:
                videos.append(os.path.join(root, f))

    def sort_key(fpath):
        # Primary key: source video name from parent dir "output_0610-1" -> "0610-1"
        # This keeps all clips from video 1 before video 2
        source_name = os.path.basename(os.path.dirname(fpath)).replace("output_", "")
        # Secondary key: timestamp within that source video
        fname = os.path.basename(fpath)
        m = re.match(r'^(\d{2})_(\d{2})_(\d{2})_', fname)
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) if m else 999999
        return (source_name, t)

    videos.sort(key=sort_key)
    return videos

def get_video_info(video_path):
    """動画の解像度とFPSを取得する"""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

# ==========================================
# 2. 結合処理（ED事前作成・最速版）
# ==========================================
def merge_videos(mmdd, detect_out_root, final_out_dir):
    print("\n" + "="*40 + "\n  結合処理(merge) 開始\n" + "="*40)
    
    # 事前作成した「真・エンディング」を使う
    ENDING_MATCHED = os.path.join(SOZAI_DIR, "ending_matched.mp4")
    
    if not os.path.exists(OPENING_VIDEO) or not os.path.exists(ENDING_MATCHED):
        print(f"⚠️ sozaiフォルダに素材がありません。")
        print(f"※事前に以下のコマンドでEDを作成してください:")
        print(f"ffmpeg -i {ENDING_VIDEO} -c:v libx264 -pix_fmt yuv420p -s 720x720 -r 30 -c:a aac -ar 44100 {ENDING_MATCHED}")
        return

    date_text, day_text = load_date_list(mmdd)
    cut_videos = get_cut_videos(detect_out_root)
    
    if not cut_videos:
        print("⚠️ 結合する切り出し動画がありませんでした。")
        return
        
    print(f"🔗 結合対象動画: {len(cut_videos)}本")

    # メイン動画のフォーマットを取得（念のため確認）
    width, height, fps = get_video_info(cut_videos[0])
    fps = round(fps) if fps > 0 else 30
    print(f"📏 メイン動画フォーマット検知: {width}x{height}, {fps}fps")

    os.makedirs(final_out_dir, exist_ok=True)
    output_mp4 = os.path.join(final_out_dir, f"{mmdd}_output.mp4")

    with tempfile.TemporaryDirectory() as tmp_dir:
        opening_with_text = os.path.join(tmp_dir, "opening_with_text.mp4")
        
        esc = lambda s: s.replace("\\", "\\\\").replace("'", "\u2019").replace(":", "\\:")
        line1 = f"{date_text}　{day_text}"
        line2 = "～コシアカツバメの一日記録～"
        
        vf = (f"drawtext=fontfile='{FONT_PATH}':text='{esc(line1)}':fontsize=64:fontcolor=white:borderw=4:bordercolor=black:x=(w-text_w)/2:y=(h/2)-80,"
              f"drawtext=fontfile='{FONT_PATH}':text='{esc(line2)}':fontsize=36:fontcolor=white:borderw=3:bordercolor=black:x=(w-text_w)/2:y=(h/2)+10")
        
        # 1. オープニングは文字焼き込みが必須なため、ここでエンコードしつつメイン動画の規格に合わせる
        print(f"🎬 オープニング作成中（テキスト合成＆規格統一）... ({line1})")
        subprocess.run([
            "ffmpeg", "-y", "-i", OPENING_VIDEO, "-vf", vf, 
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", str(fps),
            "-c:a", "aac", "-ar", "44100", 
            opening_with_text
        ], capture_output=True)

        # 2. 全結合（EDのエンコード処理は削除！）
        list_path = os.path.join(tmp_dir, "concat_list.txt")
        all_files = [opening_with_text] + cut_videos + [ENDING_MATCHED]
        
        with open(list_path, "w", encoding="utf-8") as f:
            for fpath in all_files:
                f.write(f"file '{os.path.abspath(fpath).replace('\'', '\\\'')}'\n")

        print("🎬 全動画を結合中... (-c copy なので高速です)")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_mp4], 
                       capture_output=True)
        
        print(f"🎉 結合完了！ -> {output_mp4}")
def normalize_video(input_path):
    output_path = input_path.replace(".mp4", "_fixed.mp4")
    # コマンドをより堅牢に定義
    cmd = [
        "ffmpeg", "-y", 
        "-i", input_path,
        "-c", "copy",
        "-video_track_timescale", "15360",
        output_path
    ]
    try:
        # check=Trueでエラー時に例外を投げる
        subprocess.run(cmd, capture_output=True, check=True)
        # 成功したらファイルを置き換え
        os.replace(output_path, input_path)
        print(f"✅ 正規化成功: {os.path.basename(input_path)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 正規化失敗: {os.path.basename(input_path)}")
        print(e.stderr.decode())
        return False
    
# ==========================================
# メイン実行処理
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='超高速版 統合テストスクリプト')
    parser.add_argument('input_dir', help='処理対象のフォルダ名（例: 0515）')
    args = parser.parse_args()

    mmdd = os.path.basename(args.input_dir.strip('/'))
    input_dir = os.path.join("input", mmdd)
    
    detect_out_root = os.path.join("work", f"{mmdd}-detection")
    final_out_dir = "marugoto"

    # Clean previous detection results to prevent stale clip contamination
    if os.path.exists(detect_out_root):
        print(f"🧹 前回の検知結果を削除: {detect_out_root}")
        shutil.rmtree(detect_out_root)
    
    print(f"📂 出力先: {detect_out_root}, {final_out_dir}")    
    videos = get_video_files(input_dir)
    if not videos:
        print(f"❌ {input_dir} フォルダに動画がありません。")
        sys.exit(1)

  # ★ここに配置：動体検知処理へ進む前に全動画を正規化する
    print(f"\n⚙️ 全動画のタイムベース(tbn)を正規化中...")
    for vp in videos:
        normalize_video(vp)
    # ★ここまで

    params = {
        'threshold': 50, 'min_area': 300, 'frame_skip': 20,
        'pre_seconds': 2, 'post_seconds': 2, 'merge_threshold': 10,
        'skip_initial_seconds': 60, 'batch_size': 30
    }
    print("\n" + "="*40 + f"\n  動体検知＆切り出し 開始 ({len(videos)}本)\n" + "="*40)
    start_time = time.time()
    
    tasks = [(vp, params, detect_out_root) for vp in videos]
    # 20コアCPUを活用：外側プロセス数を増やす（I/O競合を避けつつ4程度が安全圏）
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('spawn')) as executor:
        for result in executor.map(process_video_wrapper, tasks):
            print(f"[{result['status']}] {os.path.basename(result['video'])}")
            
    print(f"\n⏱️ 検知処理 完了 (所要時間: {time.time() - start_time:.1f}秒)")

    merge_videos(mmdd, detect_out_root, final_out_dir)