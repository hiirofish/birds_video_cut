"""Cut short highlight clips from chat-derived timestamps and burn in captions.

Usage: python cut_clips.py MMDD

Reads marugoto/MMDD_clips.json (produced by the chat-extraction step: an LLM
reads marugoto/MMDD_output.txt and picks candidate moments -- see that JSON
file for the schema). This script does the deterministic part: map each
clip's wall-clock cut_start/cut_end to a source file + offset, cut it, and
burn in the two-line caption.

Why OCR instead of just "log start time + ffprobe duration":
The pipeline log only records each video's start time to minute precision,
and -- worse -- a day's archive can contain hidden stalls/reconnects where
video-time and real wall-clock time drift out of a simple 1:1 relationship
by several minutes (confirmed empirically: one 0730 segment was off by
~8 minutes despite duration matching exactly). The camera overlay burns the
real wall-clock (and temperature/humidity) into every frame, so it's used
here as ground truth: get a rough offset from the log/duration, then
iteratively correct it by OCR-reading the overlay until it matches the
target time.
"""
import os
import re
import sys
import json
import subprocess
import tempfile
from datetime import datetime

import pytesseract
from PIL import Image

INPUT_ROOT = "input"
LOG_ROOT = "logs"
OUTPUT_DIR = os.path.join("marugoto", "shorts")
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

# Crop box (w:h:x:y) for the burned-in "YYYY-MM-DD HH:MM:SS" overlay, bottom
# right of a 720x720 frame. Re-calibrate (see scratch notes) if the source
# resolution or overlay layout ever changes.
OVERLAY_CROP = "crop=330:34:390:686"

# How far the naive (log start-time + ffprobe duration) estimate is allowed
# to be from a source file's nominal bounds before we stop considering that
# file a candidate at all. Generous because we've observed ~8 min of hidden
# drift within a single file; this is just for picking WHICH file, the exact
# offset within it is then nailed down by OCR search.
DRIFT_MARGIN_SEC = 1200

OCR_MAX_ITER = 5
OCR_TOL_SEC = 2


def hms_to_sec(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def get_sources(mmdd):
    """[{file, start_sec, duration_sec}, ...] for MMDD-1.mp4, MMDD-2.mp4, ...
    start_sec (minute precision, from the pipeline log) is only a rough
    estimate used to pick a candidate file -- see OCR search below for the
    real offset."""
    log_path = os.path.join(LOG_ROOT, f"{mmdd}_pipeline.log")
    starts = {}
    if os.path.exists(log_path):
        with open(log_path, encoding="utf-8") as f:
            for line in f:
                m = re.search(r"(\d+)本目 \((\d{2}):(\d{2})開始\)", line)
                if m:
                    idx = int(m.group(1))
                    starts[idx] = int(m.group(2)) * 3600 + int(m.group(3)) * 60

    sources = []
    idx = 1
    while True:
        path = os.path.join(INPUT_ROOT, mmdd, f"{mmdd}-{idx}.mp4")
        if not os.path.exists(path):
            break
        if idx not in starts:
            print(f"⚠️  {path} の開始時刻がログに見つかりません。スキップします。")
            idx += 1
            continue
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, check=True)
        sources.append({
            "file": path,
            "start_sec": starts[idx],
            "duration_sec": float(result.stdout.strip()),
        })
        idx += 1
    return sources


def grab_overlay_time(video_path, offset_sec):
    """OCR the burned-in wall-clock overlay at offset_sec. Returns seconds
    since midnight, or None if OCR couldn't parse a timestamp."""
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        png_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(max(0.0, offset_sec)), "-i", video_path,
             "-frames:v", "1", "-vf", OVERLAY_CROP, png_path],
            capture_output=True)
        if not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            return None
        try:
            im = Image.open(png_path).convert("L")
            im = im.resize((im.width * 3, im.height * 3))
        except Exception:
            return None
        text = pytesseract.image_to_string(im, config="--psm 7")
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", text)
        if not m:
            return None
        _, _, _, h, mi, s = (int(g) for g in m.groups())
        return h * 3600 + mi * 60 + s
    finally:
        if os.path.exists(png_path):
            os.unlink(png_path)


def locate_in_file(video_path, duration_sec, target_sec, rough_offset):
    """Iteratively correct rough_offset until the overlay clock there matches
    target_sec (assumes video-time ~ real-time locally, which held in every
    segment we sampled even though the global relationship can jump). Offset
    is kept half a second clear of the very end of the file, since seeking
    past the last frame yields no output at all."""
    offset = min(max(rough_offset, 0.0), duration_sec - 0.5)
    diff = None
    for _ in range(OCR_MAX_ITER):
        observed = grab_overlay_time(video_path, offset)
        if observed is None:
            return None, "OCR失敗"
        diff = target_sec - observed
        if abs(diff) <= OCR_TOL_SEC:
            return offset, None
        offset += diff
        if offset < 0 or offset > duration_sec - 0.5:
            return None, f"探索範囲外に外れた(残差{diff}秒)"
    return None, f"収束せず(残差{diff}秒)"


def resolve_clip(clip, sources):
    """Find (file, offset_start, offset_end) for a clip's wall-clock window,
    via OCR search anchored on each candidate file's rough estimate."""
    cs = hms_to_sec(clip["cut_start"])
    ce = hms_to_sec(clip["cut_end"])
    for src in sources:
        rough_start = cs - src["start_sec"]
        if not (-DRIFT_MARGIN_SEC <= rough_start <= src["duration_sec"] + DRIFT_MARGIN_SEC):
            continue

        off_start, err = locate_in_file(src["file"], src["duration_sec"], cs, rough_start)
        if off_start is None:
            print(f"   ⤷ {os.path.basename(src['file'])} で開始時刻特定失敗: {err}")
            continue

        rough_end = off_start + (ce - cs)
        off_end, err = locate_in_file(src["file"], src["duration_sec"], ce, rough_end)
        if off_end is None:
            print(f"   ⤷ {os.path.basename(src['file'])} で終了時刻特定失敗: {err}")
            continue

        if off_end <= off_start:
            continue
        return src["file"], off_start, off_end
    return None


def esc(s):
    return s.replace("\\", "\\\\").replace("'", "’").replace(":", "\\:")


def wrap(text, width=18):
    text = text.strip()
    if not text:
        return [""]
    return [text[i:i + width] for i in range(0, len(text), width)]


def build_drawtext(clip):
    """Caption burned into the bottom-left: @author, then their comment
    (wrapped if long), then any replies the extraction step judged to be
    about the same moment (prefixed with an arrow, in a different color so
    they read as a follow-up rather than the original comment). Bottom-left
    is used deliberately so it never collides with the camera's own
    date/temperature overlay in the bottom-right."""
    rows = [(clip["caption_line1"], 30, "yellow")]
    rows += [(line, 26, "white") for line in wrap(clip["caption_line2"])]
    for reply in clip.get("replies", []):
        reply_text = f"{reply['author']}: {reply['text']}"
        for i, line in enumerate(wrap(reply_text)):
            prefix = "→ " if i == 0 else "   "
            rows.append((prefix + line, 22, "cyan"))

    row_h = 34
    total_h = len(rows) * row_h + 20
    filters = []
    for i, (text, size, color) in enumerate(rows):
        y_from_bottom = total_h - i * row_h
        filters.append(
            f"drawtext=fontfile='{FONT_PATH}':text='{esc(text)}':fontsize={size}:"
            f"fontcolor={color}:borderw=3:bordercolor=black:x=30:y=h-{y_from_bottom}"
        )
    return ",".join(filters)


def cut_clip(mmdd, clip, sources, out_dir):
    print(f"🔎 [{clip['id']}] {clip['cut_start']}〜{clip['cut_end']} を検索中...")
    resolved = resolve_clip(clip, sources)
    if resolved is None:
        print(f"⚠️  [{clip['id']}] どのソースファイルでも時刻を特定できませんでした。スキップ（要手動確認）。")
        return False

    src_file, off_start, off_end = resolved
    duration = off_end - off_start
    vf = build_drawtext(clip)
    out_path = os.path.join(out_dir, f"{mmdd}_{clip['id']}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(off_start), "-i", src_file, "-t", str(duration),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac",
        out_path,
    ]
    print(f"🎬 [{clip['id']}] {os.path.basename(src_file)} "
          f"{off_start:.1f}s〜{off_end:.1f}s ({duration:.1f}s) -> {out_path}")
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"❌ [{clip['id']}] ffmpeg失敗:\n{result.stderr.decode()[-800:]}")
        return False
    return True


def main():
    if len(sys.argv) != 2:
        print("Usage: python cut_clips.py MMDD")
        sys.exit(1)
    mmdd = sys.argv[1]

    clips_path = os.path.join("marugoto", f"{mmdd}_clips.json")
    if not os.path.exists(clips_path):
        print(f"❌ {clips_path} が見つかりません。先に抽出ステップを実行してください。")
        sys.exit(1)
    with open(clips_path, encoding="utf-8") as f:
        data = json.load(f)

    sources = get_sources(mmdd)
    if not sources:
        print(f"❌ {mmdd} のソース動画が input/{mmdd}/ に見つかりません。")
        sys.exit(1)
    print(f"📼 ソース: {[(os.path.basename(s['file']), s['start_sec'], round(s['duration_sec'])) for s in sources]}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ok, failed = 0, 0
    for clip in data["clips"]:
        if cut_clip(mmdd, clip, sources, OUTPUT_DIR):
            ok += 1
        else:
            failed += 1

    print(f"\n完了: 成功 {ok} / 失敗・スキップ {failed}")


if __name__ == "__main__":
    main()
