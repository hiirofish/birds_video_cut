import cv2
import numpy as np
import datetime
import os
import time
import argparse
import shutil
import glob
from collections import deque

def clear_output_directories(detection_dir, video_dir, log_file=None):
    """出力ディレクトリの中身をクリアする関数"""
    try:
        if os.path.exists(detection_dir):
            shutil.rmtree(detection_dir)
        os.makedirs(detection_dir, exist_ok=True)
        
        if os.path.exists(video_dir):
            shutil.rmtree(video_dir)
        os.makedirs(video_dir, exist_ok=True)
        
        log_message("出力ディレクトリを初期化しました", True, log_file)
    except Exception as e:
        log_message(f"ディレクトリ初期化エラー: {e}", True, log_file)

def get_filename_without_extension(file_path):
    """ファイルパスから拡張子を除いたファイル名を取得する関数"""
    return os.path.splitext(os.path.basename(file_path))[0]

def get_output_directories(video_path, base_detection_dir="output_detection", base_video_dir="output_video"):
    """動画ファイル名に基づいて出力ディレクトリ名を生成する関数"""
    filename = get_filename_without_extension(video_path)
    detection_dir = f"{base_detection_dir}_{filename}"
    video_dir = f"{base_video_dir}_{filename}"
    return detection_dir, video_dir

def get_video_files(input_dir):
    """inputディレクトリから動画ファイルを取得する関数"""
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv', '*.flv', '*.webm']
    video_files = []
    
    for extension in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, extension)))
        # 大文字小文字の両方をチェック
        video_files.extend(glob.glob(os.path.join(input_dir, extension.upper())))
    
    return sorted(video_files)

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
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + "\n")
        except:
            pass  # ログファイル書き込みエラーは無視

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

def detect_motion(video_path, detection_dir, video_dir, threshold=25, min_area=300, 
                 frame_skip=10, debug=False, start_time="", end_time="", 
                 pre_seconds=10, post_seconds=10, merge_threshold=30, skip_initial_seconds=60):
    """動画から動きを検出し、検出部分を新しい動画として保存する関数"""
    # 出力ディレクトリを初期化
    clear_output_directories(detection_dir, video_dir)
    
    # ログファイルの設定
    log_file = os.path.join(detection_dir, "detection_log.txt")
    
    log_message(f"処理開始: {video_path}", True, log_file)
    log_message(f"初期検出スキップ時間: {skip_initial_seconds}秒", True, log_file)
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
    
    log_message(f"動画情報: {width}x{height}, {fps}fps, {total_frames}フレーム, {duration:.1f}秒", True, log_file)
    
    # 開始・終了フレームの計算
    start_frame = get_frame_number_from_time(start_time, fps) if start_time else 0
    end_frame = get_frame_number_from_time(end_time, fps) if end_time else total_frames
    
    # 初期スキップフレームの計算
    skip_frames = int(fps * skip_initial_seconds)
    effective_start_frame = max(start_frame, skip_frames)
    
    if start_frame > 0 or end_frame < total_frames or skip_initial_seconds > 0:
        log_message(f"処理範囲: フレーム {effective_start_frame} から {end_frame} (初期{skip_initial_seconds}秒スキップ)", True, log_file)
    
    # 指定範囲外のフレームを読み飛ばす
    if effective_start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, effective_start_frame)
    
    # 背景減算器の初期化
    backSub = cv2.createBackgroundSubtractorMOG2(
        history=120,
        varThreshold=24,
        detectShadows=False
    )
    
    # 検出情報を保存するリスト
    detections = []
    
    # 処理用の変数
    frame_count = effective_start_frame
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
    
    # メモリ管理用の変数
    gc_counter = 0
    
    # 音声処理は常に有効にする（OpenCVの検出に頼らない）
    has_audio = True
    log_message("音声処理を有効にしました", True, log_file)
    
    # フレームの読み込みと処理
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= end_frame:
            break
            
        frame_count += 1
        
        # フレームをバッファに追加（メモリ使用量を抑制）
        if len(frame_buffer) >= frame_buffer.maxlen:
            frame_buffer.popleft()  # 古いフレームを削除
        frame_buffer.append(frame.copy())
        
        # フレームスキップ（処理速度向上）
        if frame_count % frame_skip != 0 and frame_count > effective_start_frame + 1:
            continue
            
        processed_count += 1
        
        # フレーム処理
        result = process_frame(frame, backSub, threshold, min_area, debug)
        
        # 進捗表示（1%単位）
        current_progress_percent = int((frame_count - effective_start_frame) / (end_frame - effective_start_frame) * 100)
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
            
            # ファイル名を取得してハイライト画像名に含める
            base_filename = get_filename_without_extension(video_path)
            highlight_filename = f"{base_filename}_motion_{video_time}{file_suffix}_highlight.jpg"
            
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
                    "start_frame": max(frame_count - int(fps * pre_seconds), effective_start_frame),
                    "end_frame": frame_count + int(fps * post_seconds),
                    "first_detection": detection_info
                }
            else:
                # 既存のイベントの延長
                current_event["end_frame"] = frame_count + int(fps * post_seconds)
        
        # 現在のイベントが終了したかチェック
        if current_event and frame_count >= current_event["end_frame"]:
            motion_events.append(current_event)
            current_event = None
        
        # メモリ管理（1000フレームごとにガベージコレクション）
        gc_counter += 1
        if gc_counter >= 1000:
            import gc
            gc.collect()
            gc_counter = 0
    
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
            else:
                merged_events.append(current_merged_event)
                current_merged_event = motion_events[i]
        
        merged_events.append(current_merged_event)
    
    # 動画ファイルの保存
    base_filename = get_filename_without_extension(video_path)
    for idx, event in enumerate(merged_events):
        start_time_str = format_video_time(event["start_frame"], fps)
        end_time_str = format_video_time(event["end_frame"], fps)
        out_filename = f"{base_filename}_motion_{start_time_str}_to_{end_time_str}.mp4"
        out_path = os.path.join(video_dir, out_filename)
        
        # 動画の保存設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        temp_file = out_path + ".temp.mp4"
        out = cv2.VideoWriter(temp_file, fourcc, fps, (width, height))
        
        # 元の動画から指定範囲のフレームを抽出して保存
        cap.set(cv2.CAP_PROP_POS_FRAMES, event["start_frame"])
        temp_frame_count = event["start_frame"]
        
        while temp_frame_count <= event["end_frame"] and temp_frame_count < end_frame:
            ret, temp_frame = cap.read()
            if not ret:
                break
            
            out.write(temp_frame)
            temp_frame_count += 1
        
        out.release()
        
        # FFmpegを使って音声を追加
        if has_audio:
            start_seconds = event["start_frame"] / fps
            duration = (event["end_frame"] - event["start_frame"]) / fps
            
            # FFmpegコマンドを構築
            ffmpeg_cmd = f"ffmpeg -y -i \"{temp_file}\" -ss {start_seconds:.3f} -t {duration:.3f} -i \"{video_path}\" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0? \"{out_path}\" 2>/dev/null"
            
            # コマンド実行
            log_message(f"音声付き動画を作成中: {out_filename}", True, log_file)
            result = os.system(ffmpeg_cmd)
            
            if result != 0:
                # FFmpegコマンドが失敗した場合は、音声なしで処理
                log_message(f"音声処理に失敗、音声なしで保存: {out_filename}", True, log_file)
                os.rename(temp_file, out_path)
            else:
                log_message(f"音声付き動画保存完了: {out_filename}", True, log_file)
        else:
            # 音声なしの場合は一時ファイルをリネーム
            os.rename(temp_file, out_path)
        
        # 一時ファイルを削除
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        log_message(f"動画保存完了: {out_filename} ({start_time_str}から{end_time_str}まで)", True, log_file)
    
    # 処理結果のサマリー
    elapsed = time.time() - start_process_time
    speed = (frame_count - effective_start_frame) / elapsed if elapsed > 0 else 0
    
    log_message(f"処理完了: 総フレーム数={frame_count - effective_start_frame}, 処理フレーム数={processed_count}, "
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

def process_all_videos(input_dir, base_detection_dir="output_detection", base_video_dir="output_video", 
                      threshold=25, min_area=300, frame_skip=10, debug=False, start_time="", end_time="", 
                      pre_seconds=10, post_seconds=10, merge_threshold=30, skip_initial_seconds=60):
    """inputディレクトリ内のすべての動画ファイルを処理する関数"""
    # inputディレクトリの存在確認
    if not os.path.exists(input_dir):
        print(f"エラー: inputディレクトリが見つかりません: {input_dir}")
        return
    
    # 動画ファイルの取得
    video_files = get_video_files(input_dir)
    
    if not video_files:
        print(f"エラー: inputディレクトリに動画ファイルが見つかりません: {input_dir}")
        return
    
    print(f"処理対象の動画ファイル数: {len(video_files)}")
    for i, video_file in enumerate(video_files, 1):
        print(f"{i}. {os.path.basename(video_file)}")
    
    # 各動画ファイルを処理
    total_start_time = time.time()
    successful_count = 0
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"処理中 ({i}/{len(video_files)}): {os.path.basename(video_path)}")
        print(f"{'='*50}")
        
        # 出力ディレクトリ名を生成
        detection_dir, video_dir = get_output_directories(video_path, base_detection_dir, base_video_dir)
        
        # 出力ディレクトリの作成
        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        
        try:
            # 動画処理を実行
            success = detect_motion(
                video_path, 
                detection_dir, 
                video_dir, 
                threshold=threshold, 
                min_area=min_area,
                frame_skip=frame_skip,
                debug=debug,
                start_time=start_time,
                end_time=end_time,
                pre_seconds=pre_seconds,
                post_seconds=post_seconds,
                merge_threshold=merge_threshold,
                skip_initial_seconds=skip_initial_seconds
            )
            
            if success:
                successful_count += 1
                print(f"✓ 処理完了: {os.path.basename(video_path)}")
            else:
                print(f"⚠ 動きが検出されませんでした: {os.path.basename(video_path)}")
                
        except Exception as e:
            print(f"✗ 処理エラー: {os.path.basename(video_path)} - {str(e)}")
    
    # 全体の処理結果を表示
    total_elapsed = time.time() - total_start_time
    print(f"\n{'='*50}")
    print(f"全体の処理結果")
    print(f"{'='*50}")
    print(f"処理済み動画: {successful_count}/{len(video_files)}")
    print(f"総処理時間: {total_elapsed:.1f}秒")
    print(f"平均処理時間: {total_elapsed/len(video_files):.1f}秒/動画")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='鳥の巣観測動画から動きを検出するツール')
    parser.add_argument('input_path', nargs='?', default='input', 
                       help='入力動画のパスまたはinputディレクトリ（デフォルト: input）')
    parser.add_argument('--detection_dir', default='output_detection', help='検出結果の出力ディレクトリベース名')
    parser.add_argument('--video_dir', default='output_video', help='動画出力ディレクトリベース名')
    parser.add_argument('--threshold', '-t', type=int, default=50, help='動き検出の閾値')
    parser.add_argument('--min_area', '-a', type=int, default=300, help='動きと判断する最小領域サイズ')
    parser.add_argument('--frame_skip', '-s', type=int, default=20, help='フレームスキップ数')
    parser.add_argument('--start', help='分析開始時間 (H:M:S形式、例: 1:30:00)')
    parser.add_argument('--end', help='分析終了時間 (H:M:S形式、例: 2:00:00)')
    parser.add_argument('--pre', type=int, default=3, help='動き検出前の秒数')
    parser.add_argument('--post', type=int, default=3, help='動き検出後の秒数')
    parser.add_argument('--merge', type=int, default=10, help='イベントをマージする秒数の閾値')
    parser.add_argument('--skip_initial', type=int, default=60, help='最初にスキップする秒数（誤検出防止）')
    parser.add_argument('--debug', '-d', action='store_true', help='デバッグ情報を表示')
    parser.add_argument('--single', action='store_true', help='単一ファイル処理モード（従来の動作）')
    
    args = parser.parse_args()
    
    # 単一ファイル処理モードまたは直接ファイルが指定された場合
    if args.single or (os.path.isfile(args.input_path) and not os.path.isdir(args.input_path)):
        # 従来の単一ファイル処理
        video_path = args.input_path
        detection_dir, video_dir = get_output_directories(video_path, args.detection_dir, args.video_dir)
        
        # 出力ディレクトリの作成
        os.makedirs(detection_dir, exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)
        
        detect_motion(
            video_path, 
            detection_dir, 
            video_dir, 
            threshold=args.threshold, 
            min_area=args.min_area,
            frame_skip=args.frame_skip,
            debug=args.debug,
            start_time=args.start,
            end_time=args.end,
            pre_seconds=args.pre,
            post_seconds=args.post,
            merge_threshold=args.merge,
            skip_initial_seconds=args.skip_initial
        )
    else:
        # 複数ファイル処理（inputディレクトリ内のすべての動画）
        process_all_videos(
            args.input_path,
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
            merge_threshold=args.merge,
            skip_initial_seconds=args.skip_initial
        )