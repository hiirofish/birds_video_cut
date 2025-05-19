import cv2
import numpy as np
import datetime
import os
import time
import argparse
from collections import deque

def format_video_time(frame_number, fps):
    """フレーム番号から時:分:秒形式の文字列に変換する関数"""
    total_seconds = int(frame_number / fps) if fps > 0 else 0
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}_{minutes:02d}_{seconds:02d}"

def get_frame_number_from_time(time_str, fps):
    """時:分:秒形式の文字列からフレーム番号を計算する関数"""
    if not time_str:
        return 0
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = map(int, parts)
        else:
            hours = 0
            minutes = 0
            seconds = int(parts[0])
        
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return int(total_seconds * fps)
    except Exception as e:
        print(f"時間形式エラー: {time_str}, {e}")
        return 0

def log_message(message, show=True, log_file=None):
    """ログメッセージを表示/保存する関数"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    
    if show:
        print(log_message)
    
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + "\n")

def process_frame(frame, backSub, threshold, min_area, debug=False):
    """フレームの処理と動き検出を行う関数"""
    start_time = time.time()
    
    # 処理速度向上のためにフレームサイズを縮小
    scale_factor = 0.5
    small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    
    # グレースケール変換（処理を高速化）
    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    # 明るさのチェック
    avg_brightness = np.mean(gray)
    
    # ノイズ除去（ガウシアンブラー）
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    # 背景減算
    fgMask = backSub.apply(gray)
    
    # 二値化
    _, thresh = cv2.threshold(fgMask, threshold, 255, cv2.THRESH_BINARY)
    
    # ノイズ除去
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # 動きの検出（輪郭を検出）
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 一定サイズ以上の輪郭があれば動きとみなす
    motion_detected = False
    largest_area = 0
    largest_contour = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > largest_area:
            largest_area = area
            largest_contour = contour
    
    # 明るさに応じた閾値調整
    effective_min_area = min_area
    
    # 極端に暗い場合
    if avg_brightness < 15:
        motion_detected = False
    # 夕方の薄暗い場合は感度を調整
    elif 15 <= avg_brightness < 50:
        # 夕方用に感度を上げる（面積閾値を下げる）
        effective_min_area = min_area * 0.7
        if largest_area > effective_min_area:
            motion_detected = True
    # 通常の明るさ
    else:
        if largest_area > effective_min_area:
            motion_detected = True
    
    total_time = time.time() - start_time
    
    result = {
        "motion_detected": motion_detected,
        "area": largest_area,
        "brightness": avg_brightness,
        "processing_time": total_time,
        "largest_contour": largest_contour,
        "threshold_used": effective_min_area
    }
    
    return result

def is_cuda_available():
    """GPUの利用可否を確認する関数"""
    return cv2.cuda.getCudaEnabledDeviceCount() > 0

def detect_motion(video_path, detection_dir, video_dir, threshold=25, min_area=300, 
                 frame_skip=10, debug=False, start_time="", end_time="", 
                 pre_seconds=10, post_seconds=10, merge_threshold=30):
    """動画から動きを検出し、検出部分を新しい動画として保存する関数"""
    # ログファイルの設定
    log_file = os.path.join(detection_dir, "detection_log.txt")
    
    log_message(f"処理開始: {video_path}", True, log_file)
    start_process_time = time.time()
    
    # ビデオの読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_message(f"エラー: 動画ファイルを開けませんでした: {video_path}", True, log_file)
        return False
    
    # 動画情報の取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    has_audio = cap.get(cv2.CAP_PROP_AUDIO_STREAM) > 0
    
    log_message(f"動画情報: {width}x{height}, {fps}fps, {total_frames}フレーム, {duration:.1f}秒, 音声: {'あり' if has_audio else 'なし'}", True, log_file)
    
    # 開始・終了フレームの計算
    start_frame = get_frame_number_from_time(start_time, fps) if start_time else 0
    end_frame = get_frame_number_from_time(end_time, fps) if end_time else total_frames
    
    if start_frame > 0 or end_frame < total_frames:
        log_message(f"指定時間範囲: {start_time or '開始'} から {end_time or '終了'} (フレーム {start_frame} から {end_frame})", True, log_file)
    
    # 指定範囲外のフレームを読み飛ばす
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 背景減算器の初期化
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=120,
        varThreshold=24,
        detectShadows=False
    )
    
    # 検出情報を保存するリスト
    detections = []
    
    # 処理用の変数
    frame_count = start_frame
    processed_count = 0
    motion_frames = 0
    last_progress_percent = -1
    
    # 同じフレーム時間のカウンター（衝突回避用）
    time_collision_counter = {}
    
    # フレームバッファ（動画保存用）
    frame_buffer = deque(maxlen=int(fps * pre_seconds))
    
    # 動きの検出イベント管理
    motion_events = []
    current_event = None
    
    # GPUの確認
    use_gpu = is_cuda_available()
    if use_gpu:
        log_message("GPUが利用可能です。動画エンコードにGPUを使用します。", True, log_file)
    else:
        log_message("GPUが利用できません。CPUで処理します。", True, log_file)
    
    # フレームの読み込みと処理
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= end_frame:
            break
            
        frame_count += 1
        
        # フレームをバッファに追加
        frame_buffer.append(frame.copy())
        
        # フレームスキップ（処理速度向上）
        if frame_count % frame_skip != 0 and frame_count > start_frame + 1:
            continue
            
        processed_count += 1
        
        # フレーム処理
        result = process_frame(frame, backSub, threshold, min_area, debug)
        
        # 進捗表示（1%単位）
        current_progress_percent = int((frame_count - start_frame) / (end_frame - start_frame) * 100)
        if current_progress_percent != last_progress_percent:
            elapsed = time.time() - start_process_time
            speed = frame_count / elapsed if elapsed > 0 else 0
            remaining = (end_frame - frame_count) / speed if speed > 0 else 0
            
            log_message(f"進捗: {current_progress_percent}% ({frame_count}/{end_frame}), 速度: {speed:.1f}fps, 残り: {remaining:.1f}秒, 検出: {motion_frames}件", True, log_file)
            last_progress_percent = current_progress_percent
        
        if result["motion_detected"]:
            motion_frames += 1
            
            # 動画時間をhh_mm_ss形式に変換
            video_time = format_video_time(frame_count, fps)
            
            # 同じ時間のファイルがある場合の衝突回避
            if video_time in time_collision_counter:
                time_collision_counter[video_time] += 1
                file_suffix = f"_{time_collision_counter[video_time]}"
            else:
                time_collision_counter[video_time] = 0
                file_suffix = ""
            
            # ハイライト画像のみ保存
            highlight_filename = f"motion_{video_time}{file_suffix}_highlight.jpg"
            
            # 検出領域を強調した画像
            if result["largest_contour"] is not None:
                highlight_frame = frame.copy()
                # 縮小したフレームでの輪郭を元のサイズに戻す
                scaled_contour = result["largest_contour"] * (1.0/0.5)
                scaled_contour = scaled_contour.astype(np.int32)
                cv2.drawContours(highlight_frame, [scaled_contour], -1, (0, 255, 0), 2)
                cv2.imwrite(os.path.join(detection_dir, highlight_filename), highlight_frame)
            
            # 検出情報をリストに追加
            detection_info = {
                "frame": frame_count,
                "video_time": video_time,
                "area": result["area"],
                "brightness": result["brightness"],
                "highlight_file": highlight_filename
            }
            detections.append(detection_info)
            
            log_message(f"検出: {highlight_filename} (時間: {video_time}, 面積: {result['area']:.1f}, 輝度: {result['brightness']:.1f})", True, log_file)
            
            # 動画イベント管理
            if current_event is None:
                # 新しいイベントの開始
                current_event = {
                    "start_frame": max(frame_count - int(fps * pre_seconds), start_frame),
                    "frames": list(frame_buffer),
                    "end_frame": frame_count + int(fps * post_seconds),
                    "first_detection": detection_info
                }
            else:
                # 既存のイベントの延長
                current_event["end_frame"] = frame_count + int(fps * post_seconds)
                current_event["frames"].append(frame.copy())
        
        # 現在のイベントが終了したかチェック
        if current_event and frame_count >= current_event["end_frame"]:
            motion_events.append(current_event)
            current_event = None
    
    # 最後のイベントを追加
    if current_event:
        motion_events.append(current_event)
    
    # イベントのマージ処理
    merged_events = []
    
    if motion_events:
        current_merged_event = motion_events[0]
        
        for i in range(1, len(motion_events)):
            # 次のイベントとの間隔が閾値以内なら結合
            if motion_events[i]["start_frame"] - current_merged_event["end_frame"] <= fps * merge_threshold:
                current_merged_event["end_frame"] = motion_events[i]["end_frame"]
                current_merged_event["frames"].extend(motion_events[i]["frames"])
            else:
                merged_events.append(current_merged_event)
                current_merged_event = motion_events[i]
        
        merged_events.append(current_merged_event)
    
    # 動画ファイルの保存
    for idx, event in enumerate(merged_events):
        start_time_str = format_video_time(event["start_frame"], fps)
        end_time_str = format_video_time(event["end_frame"], fps)
        out_filename = f"motion_{start_time_str}_to_{end_time_str}.mp4"
        out_path = os.path.join(video_dir, out_filename)
        
        # 動画の保存設定
        if use_gpu:
            # GPU使用時の設定
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264コーデック
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        else:
            # CPU使用時の設定
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG-4コーデック
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        
        # 元の動画から指定範囲のフレームを抽出して保存
        cap.set(cv2.CAP_PROP_POS_FRAMES, event["start_frame"])
        frame_count = event["start_frame"]
        
        while frame_count <= event["end_frame"] and frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
        
        out.release()
        
        # FFmpegでオーディオを追加（オーディオがある場合）
        if True:
            temp_file = out_path + ".temp.mp4"
            os.rename(out_path, temp_file)
            
            start_seconds = event["start_frame"] / fps
            duration = (event["end_frame"] - event["start_frame"]) / fps
            
            # FFmpegコマンドを構築
            ffmpeg_cmd = f"ffmpeg -y -i \"{temp_file}\" -ss {start_seconds} -t {duration} -i \"{video_path}\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 \"{out_path}\""
            
            # コマンド実行
            log_message(f"音声を追加中: {out_filename}", True, log_file)
            os.system(ffmpeg_cmd)
            
            # 一時ファイルを削除
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        log_message(f"動画保存完了: {out_filename} ({start_time_str}から{end_time_str}まで)", True, log_file)
    
    # 処理結果のサマリー
    elapsed = time.time() - start_process_time
    speed = (frame_count - start_frame) / elapsed if elapsed > 0 else 0
    
    log_message(f"処理完了: 総フレーム数={frame_count - start_frame}, 処理フレーム数={processed_count}, "
              f"動き検出={motion_frames}箇所, イベント数={len(motion_events)}件, マージ後={len(merged_events)}件, "
              f"処理時間={elapsed:.1f}秒, 処理速度={speed:.1f}fps", True, log_file)
    
    # 検出情報をCSVに出力
    if detections:
        import csv
        csv_path = os.path.join(detection_dir, "detections.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['frame', 'video_time', 'area', 'brightness', 'highlight_file']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for detection in detections:
                writer.writerow(detection)
        log_message(f"検出情報をCSVに保存: {csv_path}", True, log_file)
    
    # リソースの解放
    cap.release()
    
    return motion_frames > 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='鳥の巣観測動画から動きを検出するツール')
    parser.add_argument('video_path', help='入力動画のパス')
    parser.add_argument('--detection_dir', default='output_detection', help='検出結果の出力ディレクトリ')
    parser.add_argument('--video_dir', default='output_video', help='動画出力ディレクトリ')
    parser.add_argument('--threshold', '-t', type=int, default=25, help='動き検出の閾値')
    parser.add_argument('--min_area', '-a', type=int, default=300, help='動きと判断する最小領域サイズ')
    parser.add_argument('--frame_skip', '-s', type=int, default=10, help='フレームスキップ数')
    parser.add_argument('--start', help='分析開始時間 (H:M:S形式、例: 1:30:00)')
    parser.add_argument('--end', help='分析終了時間 (H:M:S形式、例: 2:00:00)')
    parser.add_argument('--pre', type=int, default=10, help='動き検出前の秒数')
    parser.add_argument('--post', type=int, default=10, help='動き検出後の秒数')
    parser.add_argument('--merge', type=int, default=30, help='イベントをマージする秒数の閾値')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグ情報を表示')
    
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.detection_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)
    
    detect_motion(
        args.video_path, 
        args.detection_dir, 
        args.video_dir, 
        threshold=args.threshold, 
        min_area=args.min_area,
        frame_skip=args.frame_skip,
        debug=args.debug,
        start_time=args.start,
        end_time=args.end,
        pre_seconds=args.pre,
        post_seconds=args.post,
        merge_threshold=args.merge
    )