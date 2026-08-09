"""Combine the day's individual clips (marugoto/shorts/MMDD_MMDD-NN.mp4, made
by cut_clips.py) into one compilation with swipe-style transitions between
them, in the same order as marugoto/MMDD_clips.json. An opening (date/DAY +
title card, from sozai/opening_short.mp4) is prepended, and a subscribe/thanks
overlay is faded in over the last few seconds -- no extra runtime added.

Usage: python compile_shorts.py MMDD
Output: marugoto/MMDD_short.mp4
"""
import os
import sys
import json
import tempfile
import subprocess

CLIP_DIR = os.path.join("marugoto", "shorts")  # per-comment temp clips (cut_clips.py output)
OUT_DIR = "marugoto"  # final combined short goes here, not mixed in with the temp clips
SOZAI_DIR = "sozai"
OPENING_VIDEO = os.path.join(SOZAI_DIR, "opening_short.mp4")
DATE_LIST_FILE = os.path.join(SOZAI_DIR, "date_list.txt")
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

# xfade transition name -- "swipe"-like screen change between clips.
# See `ffmpeg -h filter=xfade` for the full list (fade, wipeleft, slideup, ...).
TRANSITION = "slideleft"
TRANSITION_DUR = 0.5  # seconds of overlap between consecutive clips

# Last N seconds of the finished video where the subscribe/thanks overlay
# fades in. Does not extend the video -- it's composited over existing tail.
ENDING_OVERLAY_DUR = 3.0
ENDING_FADE_DUR = 0.5


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:")


def get_duration(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def get_video_info(path):
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True)
    width, height, rate = result.stdout.strip().split(",")
    num, den = rate.split("/")
    fps = round(float(num) / float(den)) if float(den) else 30
    return int(width), int(height), fps


def load_date_list(mmdd):
    """(date_text, day_text) e.g. ('８月７日', 'ＤＡＹ１０２'), read from
    sozai/date_list.txt (same file/format fast_bird_pipeline.py uses for its
    motion-detection-version opening)."""
    mm, dd = int(mmdd[:2]), int(mmdd[2:])
    wide = str.maketrans("0123456789", "０１２３４５６７８９")
    date_jp = f"{mm}月{dd}日".translate(wide)
    if not os.path.exists(DATE_LIST_FILE):
        return date_jp, "DAY ??"
    with open(DATE_LIST_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] == date_jp:
                return parts[0], parts[1]
    return date_jp, "DAY ??"


def build_opening(mmdd, title, width, height, fps, tmp_dir):
    """sozai/opening_short.mp4 + date/DAY (white, same position as
    fast_bird_pipeline.py's motion-detection opening) + title (yellow, right
    below it) burned in, re-encoded to match the main clips' format so it can
    sit in the same xfade chain."""
    date_text, day_text = load_date_list(mmdd)
    line1 = f"{date_text}　{day_text}"

    vf = (f"drawtext=fontfile='{FONT_PATH}':text='{esc(line1)}':fontsize=64:"
          f"fontcolor=white:borderw=4:bordercolor=black:box=1:boxcolor=black@0.35:boxborderw=14:"
          f"x=(w-text_w)/2:y=(h/2)-80")
    if title:
        vf += (f",drawtext=fontfile='{FONT_PATH}':text='{esc(title)}':fontsize=54:"
               f"fontcolor=yellow:borderw=4:bordercolor=black:box=1:boxcolor=black@0.35:boxborderw=14:"
               f"x=(w-text_w)/2:y=(h/2)+10")

    out_path = os.path.join(tmp_dir, "opening_with_text.mp4")
    cmd = [
        "ffmpeg", "-y", "-i", OPENING_VIDEO, "-vf", vf,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-s", f"{width}x{height}", "-r", str(fps),
        "-c:a", "aac", "-ar", "44100",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"❌ オープニング作成失敗:\n{result.stderr.decode()[-1500:]}")
        sys.exit(1)
    print(f"🎬 オープニング作成 ({line1}{' / ' + title if title else ''})")
    return out_path


def build_ending_overlay_filter(label, out_label, total_dur, width, height):
    """Fade+slide in 'YouTube登録お願いします！' (above center) and
    'いつもコメントありがとうございます' (below center) during the last
    ENDING_OVERLAY_DUR seconds -- transparent text over whatever's already
    playing (the nest sits roughly center in the source footage), no extra
    runtime. Top line appears first, bottom line follows shortly after."""
    appear_top = max(0.0, total_dur - ENDING_OVERLAY_DUR)
    appear_bottom = appear_top + 0.3
    fd = ENDING_FADE_DUR

    def alpha_expr(appear):
        return (f"if(lt(t,{appear:.2f}),0,"
                f"if(lt(t,{appear + fd:.2f}),(t-{appear:.2f})/{fd},1))")

    def slide_y_expr(appear, y_target, y_from):
        return (f"if(lt(t,{appear:.2f}),{y_from},"
                f"if(lt(t,{appear + fd:.2f}),{y_target}+({y_from}-{y_target})*"
                f"(1-(t-{appear:.2f})/{fd}),{y_target}))")

    y_top_target = round(height * 0.12)
    y_bottom_target = round(height * 0.62)

    top_text = esc("YouTube登録お願いします！")
    bottom_text = esc("いつもコメントありがとうございます")

    top = (f"drawtext=fontfile='{FONT_PATH}':text='{top_text}':fontsize=46:"
           f"fontcolor=white:borderw=4:bordercolor=red:box=1:boxcolor=red@0.55:boxborderw=16:"
           f"x=(w-text_w)/2:y='{slide_y_expr(appear_top, y_top_target, y_top_target - 40)}':"
           f"alpha='{alpha_expr(appear_top)}'")
    bottom = (f"drawtext=fontfile='{FONT_PATH}':text='{bottom_text}':fontsize=38:"
              f"fontcolor=yellow:borderw=3:bordercolor=black:box=1:boxcolor=black@0.35:boxborderw=12:"
              f"x=(w-text_w)/2:y='{slide_y_expr(appear_bottom, y_bottom_target, y_bottom_target + 40)}':"
              f"alpha='{alpha_expr(appear_bottom)}'")

    return f"[{label}]{top},{bottom}[{out_label}]"


def main():
    if len(sys.argv) != 2:
        print("Usage: python compile_shorts.py MMDD")
        sys.exit(1)
    mmdd = sys.argv[1]

    clips_path = os.path.join("marugoto", f"{mmdd}_clips.json")
    with open(clips_path, encoding="utf-8") as f:
        data = json.load(f)
    title = data.get("title", "")

    clip_files = []
    for clip in data["clips"]:
        path = os.path.join(CLIP_DIR, f"{mmdd}_{clip['id']}.mp4")
        if os.path.exists(path):
            clip_files.append(path)
        else:
            print(f"⚠️  {path} が無いのでスキップします（cut_clips.pyで失敗/未生成？）")

    if len(clip_files) < 2:
        print("❌ 結合には最低2本のクリップが必要です。")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        width, height, fps = get_video_info(clip_files[0])
        opening_path = build_opening(mmdd, title, width, height, fps, tmp_dir)
        clip_files = [opening_path] + clip_files

        print(f"🔗 結合対象: {len(clip_files)}本 (transition={TRANSITION}, {TRANSITION_DUR}s)")
        durations = [get_duration(p) for p in clip_files]

        inputs = []
        for p in clip_files:
            inputs += ["-i", p]

        filter_parts = []
        cum = durations[0]
        prev_v, prev_a = "0:v", "0:a"
        for i in range(1, len(clip_files)):
            # Transitions need a minimum clip length to overlap into; guard
            # against a clip shorter than the transition itself.
            dur = min(TRANSITION_DUR, durations[i - 1] - 0.1, durations[i] - 0.1)
            dur = max(dur, 0.1)
            offset = cum - dur
            vout, aout = f"v{i}", f"a{i}"
            filter_parts.append(
                f"[{prev_v}][{i}:v]xfade=transition={TRANSITION}:duration={dur:.2f}:"
                f"offset={offset:.2f}[{vout}]")
            filter_parts.append(f"[{prev_a}][{i}:a]acrossfade=d={dur:.2f}[{aout}]")
            prev_v, prev_a = vout, aout
            cum += durations[i] - dur

        filter_parts.append(build_ending_overlay_filter(prev_v, "vfinal", cum, width, height))
        prev_v = "vfinal"

        filter_complex = ";".join(filter_parts)
        out_path = os.path.join(OUT_DIR, f"{mmdd}_short.mp4")

        cmd = ["ffmpeg", "-y"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", f"[{prev_v}]", "-map", f"[{prev_a}]",
            # xfade negotiates its own internal pixel format (observed: yuv444p,
            # High 4:4:4 Predictive profile) unless told otherwise, which plays
            # in ffplay/VLC but common consumer editors (e.g. PowerDirector)
            # reject it. Force standard 8-bit 4:2:0 to match cut_clips.py's output.
            "-pix_fmt", "yuv420p",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-profile:v", "high",
            "-c:a", "aac",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(f"❌ ffmpeg失敗:\n{result.stderr.decode()[-1500:]}")
            sys.exit(1)

    print(f"✅ {out_path} ({cum:.1f}秒)")


if __name__ == "__main__":
    main()
