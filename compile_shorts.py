"""Combine the day's individual clips (marugoto/shorts/MMDD_MMDD-NN.mp4, made
by cut_clips.py) into one compilation with swipe-style transitions between
them, in the same order as marugoto/MMDD_clips.json. An opening (date/DAY +
title card, from sozai/opening_short.mp4) is prepended, and a subscribe/thanks
overlay is faded in over the last few seconds -- no extra runtime added.

Usage: python compile_shorts.py MMDD
Output: marugoto/MMDD_short_DAY<n>_<title>.mp4, or, when the result would run
        past MAX_SHORT_DUR, two files _part1_/_part2_ with a title each.
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

# A Short may not run longer than 3 minutes, so a day with too many clips is
# split into two videos (part1/part2), each with its own opening and title.
MAX_SHORT_DUR = 180.0

# Last N seconds of the finished video where the subscribe/thanks overlay
# fades in. Does not extend the video -- it's composited over existing tail.
ENDING_OVERLAY_DUR = 3.0
ENDING_FADE_DUR = 0.5

# Opening card text. These are the sizes we'd like; the real size is shrunk
# per-date by fit_fontsize() so the line always fits the frame.
OPENING_DATE_FONTSIZE = 64
OPENING_TITLE_FONTSIZE = 54
OPENING_BOX_BORDER = 14  # boxborderw; the drawn box is text_w + 2x this


def fit_fontsize(text, width, preferred, box_border=OPENING_BOX_BORDER):
    """Largest size <= preferred at which `text` still fits inside `width`.

    Both opening lines are full-width Japanese glyphs, whose advance width
    equals the font size exactly in NotoSansCJK-Bold (verified: 12 glyphs at
    64 measure 768px), so the drawn width is len(text) * size + 2 * box_border.

    This used to be a flat 64. Once the counter reached DAY100 the date line
    grew a glyph -- "８月１１日　ＤＡＹ１０６" is 12 glyphs = 768 + 28 = 796px --
    and ran off both edges of the 720px frame. Shrinking to fit rather than
    hard-coding a smaller size keeps short dates large and still survives the
    longest line date_list.txt can produce ("１０月３１日　ＤＡＹ１８７", 13 glyphs).
    """
    if not text:
        return preferred
    return max(1, min(preferred, (width - 2 * box_border) // len(text)))


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


def build_out_path(mmdd, day_text, title, part=None):
    """marugoto/MMDD_short[_partN]_DAY106_タイトル.mp4

    The DAY counter and title are in the name so the finished files can be
    told apart at a glance when several days are queued up for upload; the
    date_list.txt text is full-width for the opening card, so narrow it here.
    """
    narrow = str.maketrans("ＤＡＹ０１２３４５６７８９", "DAY0123456789")
    day = day_text.translate(narrow)
    name = f"{mmdd}_short"
    if part:
        name += f"_part{part}"
    for token in (day, title):
        token = "".join(c for c in token if c not in '/\\:*?"<>|. ')
        if token:
            name += f"_{token}"
    return os.path.join(OUT_DIR, name + ".mp4")


def estimate_total(opening_dur, durations):
    """Length of the finished video -- every join overlaps by TRANSITION_DUR,
    and there is one join per clip (the opening counts as the first piece)."""
    return opening_dur + sum(durations) - TRANSITION_DUR * len(durations)


def split_point(durations):
    """Index to cut the clip list at so the two parts come out as even as
    possible. Chronological order is preserved -- we only choose a boundary,
    never reorder, so part2 always picks up where part1 left off."""
    return min(range(1, len(durations)),
               key=lambda i: abs(sum(durations[:i]) - sum(durations[i:])))


def build_opening(mmdd, title, width, height, fps, tmp_dir, suffix=""):
    """sozai/opening_short.mp4 + date/DAY (white, same position as
    fast_bird_pipeline.py's motion-detection opening) + title (yellow, right
    below it) burned in, re-encoded to match the main clips' format so it can
    sit in the same xfade chain."""
    date_text, day_text = load_date_list(mmdd)
    line1 = f"{date_text}　{day_text}"
    size1 = fit_fontsize(line1, width, OPENING_DATE_FONTSIZE)
    size2 = fit_fontsize(title, width, OPENING_TITLE_FONTSIZE)

    vf = (f"drawtext=fontfile='{FONT_PATH}':text='{esc(line1)}':fontsize={size1}:"
          f"fontcolor=white:borderw=4:bordercolor=black:box=1:boxcolor=black@0.35:"
          f"boxborderw={OPENING_BOX_BORDER}:"
          f"x=(w-text_w)/2:y=(h/2)-80")
    if title:
        vf += (f",drawtext=fontfile='{FONT_PATH}':text='{esc(title)}':fontsize={size2}:"
               f"fontcolor=yellow:borderw=4:bordercolor=black:box=1:boxcolor=black@0.35:"
               f"boxborderw={OPENING_BOX_BORDER}:"
               f"x=(w-text_w)/2:y=(h/2)+10")

    out_path = os.path.join(tmp_dir, f"opening_with_text{suffix}.mp4")
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
    print(f"🎬 オープニング作成 ({line1}{' / ' + title if title else ''}) "
          f"fontsize={size1}/{size2}")
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


def render_short(mmdd, clip_files, title, out_path, width, height, fps, tmp_dir, part=None):
    """Opening + clips -> one finished file. Returns its length in seconds."""
    opening_path = build_opening(mmdd, title, width, height, fps, tmp_dir,
                                 suffix=f"_part{part}" if part else "")
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

    cmd = ["ffmpeg", "-y"] + inputs + [
        "-filter_complex", ";".join(filter_parts),
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
    return cum


def main():
    if len(sys.argv) != 2:
        print("Usage: python compile_shorts.py MMDD")
        sys.exit(1)
    mmdd = sys.argv[1]

    clips_path = os.path.join("marugoto", f"{mmdd}_clips.json")
    with open(clips_path, encoding="utf-8") as f:
        data = json.load(f)
    title = data.get("title", "")
    title_part2 = data.get("title_part2", "")

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

    _, day_text = load_date_list(mmdd)

    with tempfile.TemporaryDirectory() as tmp_dir:
        width, height, fps = get_video_info(clip_files[0])
        durations = [get_duration(p) for p in clip_files]
        total = estimate_total(get_duration(OPENING_VIDEO), durations)

        if total <= MAX_SHORT_DUR:
            groups = [(clip_files, title, None)]
        else:
            k = split_point(durations)
            if not title_part2:
                print("⚠️  clips.json に title_part2 が無いので part2 も同じタイトルにします")
                title_part2 = title
            groups = [(clip_files[:k], title, 1), (clip_files[k:], title_part2, 2)]
            print(f"✂️  推定 {total:.1f}秒 が上限 {MAX_SHORT_DUR:.0f}秒 を超えるので2本に分割します"
                  f"（part1: {k}本 / part2: {len(clip_files) - k}本）")

        for files, part_title, part in groups:
            out_path = build_out_path(mmdd, day_text, part_title, part)
            dur = render_short(mmdd, files, part_title, out_path,
                               width, height, fps, tmp_dir, part)
            print(f"✅ {out_path} ({dur:.1f}秒)")
            if dur > MAX_SHORT_DUR:
                # Only ever split in two (that is the rule), so a day with a
                # huge number of clips can still overflow. Say so rather than
                # hand back an over-length file that looks fine.
                print(f"   ⚠️  {MAX_SHORT_DUR:.0f}秒を超えています。"
                      f"clips.jsonのクリップを減らすか手動で分けてください")


if __name__ == "__main__":
    main()
