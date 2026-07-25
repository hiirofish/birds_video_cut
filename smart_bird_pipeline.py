"""
smart_bird_pipeline.py

Fire-and-forget pipeline: find unprocessed livestream dates, download every
stream of that day as an H.264 mp4, verify the files are actually complete,
then hand off to fast_bird_pipeline.py and merge the chat logs.

Design priority is COMPLETION, not elegance. Start it, walk away, come back to
a finished job. Anything retryable is retried; anything that cannot be finished
is reported at the end with a URL for manual download.

Key idea: never guess a format string. Ask yt-dlp what formats actually exist
for each player client, pick concrete format IDs out of that inventory, and
only then download. That removes "Requested format is not available" entirely.
"""

import os
import re
import sys
import json
import glob
import time
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ==================== Settings ====================
CHANNEL_HANDLE = "@Take1bit"
FAST_BIRD_PIPELINE = "fast_bird_pipeline.py"
LOOKBACK_DAYS = 7          # Only consider livestreams within the past 7 days (JST)

INPUT_ROOT = "input"
OUTPUT_ROOT = "marugoto"
LOG_ROOT = "logs"

DOWNLOAD_RATE_LIMIT = "5M"  # cap bandwidth; lower to "3M" if the Wi-Fi chip stalls
MAX_HEIGHT = 720            # resolution ceiling

# Player clients to rotate through. A session pinned to android_vr alone can
# hand back stream URLs that require a PO token and answer 403 Forbidden.
CLIENT_VARIANTS = [
    [],  # yt-dlp's own default selection -- historically the one that worked
    ["--extractor-args", "youtube:player_client=visionos,tv"],
    ["--extractor-args", "youtube:player_client=web_safari"],
    ["--extractor-args", "youtube:player_client=tv"],
    ["--extractor-args", "youtube:player_client=android_vr,web"],
]

MAX_CANDIDATES_PER_CLIENT = 3        # how many format combos to try per client
TRANSIENT_BACKOFF = [30, 60, 120, 300]  # seconds; last value repeats

# Accept a download as complete when its duration is within this ratio of the
# duration the YouTube API reported. Catches truncated / half-merged files.
DURATION_TOLERANCE = 0.02            # 2%

# Keep stdout readable: --quiet suppresses informational chatter,
# --progress keeps the single-line progress bar. They must be used together.
QUIET_PROGRESS = ["--quiet", "--no-warnings", "--progress"]

# Error text that means "retrying identically will never help"
FATAL_PATTERNS = (
    "requested format is not available",
    "video unavailable",
    "private video",
    "members-only",
    "this live event will begin",
    "is not a valid url",
    "removed by the uploader",
)
# ====================================================

# ---------------------------------------------------------------- utilities

def log_path_for(mmdd):
    os.makedirs(LOG_ROOT, exist_ok=True)
    return os.path.join(LOG_ROOT, f"{mmdd}_pipeline.log")

def write_log(mmdd, text):
    """Append a timestamped line to the per-date log file."""
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with open(log_path_for(mmdd), "a", encoding="utf-8") as f:
            f.write(f"[{stamp}] {text}\n")
    except OSError:
        pass

def say(mmdd, text):
    """Print to console and mirror into the log file."""
    print(text, flush=True)
    write_log(mmdd, text)

def client_label(extra_args):
    if not extra_args:
        return "default"
    return extra_args[-1].split("=", 1)[-1]

def human_hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"

def is_fatal_error(stderr_text):
    """True when the error is deterministic and retrying is pointless."""
    lowered = (stderr_text or "").lower()
    return any(pattern in lowered for pattern in FATAL_PATTERNS)

# ---------------------------------------------------------------- youtube api

def load_api_key(filename="config.txt"):
    if not os.path.exists(filename):
        print(f"❌ {filename} が見つかりません。")
        return None
    with open(filename, "r", encoding="utf-8") as f:
        return f.read().strip()

def get_channel_id(api_key, handle):
    url = "https://www.googleapis.com/youtube/v3/channels"
    params = {"part": "id", "forHandle": handle, "key": api_key}
    response = requests.get(url, params=params, timeout=30).json()
    items = response.get("items", [])
    return items[0]["id"] if items else None

def utc_to_jst(utc_str):
    utc_dt = datetime.strptime(utc_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(timezone(timedelta(hours=9)))

def parse_iso8601_duration(text):
    """Convert an ISO-8601 duration such as PT7H58M14S into seconds (int)."""
    if not text:
        return 0
    match = re.match(r"P(?:(\d+)D)?T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", text)
    if not match:
        return 0
    days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
    return days * 86400 + hours * 3600 + minutes * 60 + seconds

def find_unprocessed_dates(api_key, channel_id):
    """Return [(mmdd, video_list), ...] for unprocessed dates inside the
    lookback window, oldest first."""
    search_url = "https://www.googleapis.com/youtube/v3/search"
    search_params = {
        "part": "id", "channelId": channel_id, "order": "date",
        "type": "video", "maxResults": 30, "key": api_key,
    }
    search_res = requests.get(search_url, params=search_params, timeout=30).json()
    video_ids = [item["id"]["videoId"] for item in search_res.get("items", [])]
    if not video_ids:
        return []

    video_url = "https://www.googleapis.com/youtube/v3/videos"
    video_params = {
        "part": "snippet,liveStreamingDetails,contentDetails",
        "id": ",".join(video_ids), "key": api_key,
    }
    video_res = requests.get(video_url, params=video_params, timeout=30).json()

    today_jst = datetime.now(timezone(timedelta(hours=9))).date()
    cutoff = today_jst - timedelta(days=LOOKBACK_DAYS)
    grouped = {}

    for item in video_res.get("items", []):
        if "liveStreamingDetails" not in item:
            continue

        start_time_utc = (item["liveStreamingDetails"].get("actualStartTime")
                          or item["liveStreamingDetails"].get("scheduledStartTime"))
        if not start_time_utc:
            continue

        jst_dt = utc_to_jst(start_time_utc)
        if jst_dt.date() < cutoff or jst_dt.date() > today_jst:
            continue

        duration_sec = parse_iso8601_duration(
            item.get("contentDetails", {}).get("duration", "")
        )

        mmdd = jst_dt.strftime("%m%d")
        grouped.setdefault(mmdd, []).append({
            "id": item["id"],
            "title": item["snippet"]["title"],
            "status": item["snippet"]["liveBroadcastContent"],
            "start_time": jst_dt,
            "duration_sec": duration_sec,
            "url": f"https://www.youtube.com/watch?v={item['id']}",
        })

    unprocessed = []
    for mmdd in sorted(grouped.keys()):
        if os.path.exists(os.path.join(OUTPUT_ROOT, f"{mmdd}_output.mp4")):
            continue
        grouped[mmdd].sort(key=lambda x: x["start_time"])
        unprocessed.append((mmdd, grouped[mmdd]))

    return unprocessed

# ---------------------------------------------------------------- verification

def probe_media(path):
    """Return {'vcodec','height','duration'} for a media file, or None if the
    file is missing / unreadable / not decodable."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None

    cmd = ["ffprobe", "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError, OSError):
        return None

    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        return None

    try:
        duration = float(data.get("format", {}).get("duration", 0.0))
    except (TypeError, ValueError):
        duration = 0.0

    return {
        "vcodec": video_stream.get("codec_name", ""),
        "height": int(video_stream.get("height", 0) or 0),
        "duration": duration,
    }

def can_decode_near_end(path, seconds_before_end=5):
    """Actually decode a frame close to the file's real end, instead of
    trusting the container header's self-reported duration.

    An mp4 interrupted mid-write can end up with a moov atom that still
    claims the FULL intended duration while the actual media data stops far
    earlier. ffprobe's 'duration' field is read straight from that header, so
    a truncated-but-still-tagged-as-complete file passes a duration-only
    check. -sseof seeks relative to the real end of the demuxed stream, so
    this catches that case where the header-only check cannot.
    Returns True only if a frame was actually decoded near the true end."""
    cmd = [
        "ffmpeg", "-v", "error",
        "-sseof", f"-{seconds_before_end}",
        "-i", path,
        "-frames:v", "1",
        "-f", "null", "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and result.stderr.strip() == ""

def verify_download(path, expected_duration_sec):
    """Check a finished download is a usable, complete H.264 file.
    Returns (ok: bool, reason: str)."""
    info = probe_media(path)
    if info is None:
        return False, "ファイルが壊れている/読み取れない"

    if info["vcodec"] not in ("h264", "avc1"):
        return False, f"コーデックが H.264 ではない ({info['vcodec']})"

    if expected_duration_sec > 0:
        if info["duration"] < expected_duration_sec * (1 - DURATION_TOLERANCE):
            return False, (f"尺が短い（{human_hms(info['duration'])} / "
                           f"期待 {human_hms(expected_duration_sec)}）")

        # Header says the duration is fine -- but a killed/half-merged download
        # can leave a moov atom that still claims the FULL intended duration
        # while the actual media data stops early. Confirm by decoding a real
        # frame near the true end of the file rather than trusting the header.
        check_point = min(30, max(5, expected_duration_sec * DURATION_TOLERANCE))
        if not can_decode_near_end(path, seconds_before_end=check_point):
            return False, "末尾付近が実際にはデコードできない（尻切れの疑い）"

    return True, f"{info['vcodec']} {info['height']}p {human_hms(info['duration'])}"

def cleanup_stray_parts(expected_filepath, keep_parts=True):
    """Remove leftovers from a failed attempt.
    keep_parts=True preserves *.part so the next attempt can resume."""
    stem = os.path.splitext(expected_filepath)[0]
    patterns = [f"{stem}.f*.mp4", f"{stem}.f*.m4a", f"{stem}.f*.webm"]
    if not keep_parts:
        patterns += [f"{expected_filepath}.part", f"{stem}.f*.part",
                     f"{stem}.f*.part-Frag*"]

    for pattern in patterns:
        for leftover in glob.glob(pattern):
            try:
                os.remove(leftover)
            except OSError:
                pass

# ---------------------------------------------------------------- format inventory

def fetch_format_inventory(video_url, client_args, timeout=180):
    """Ask yt-dlp what formats this client can actually see.
    Returns (formats: list, error_text: str|None)."""
    cmd = [
        "yt-dlp", "--remote-components", "ejs:github",
        "--quiet", "--no-warnings",
        "--skip-download", "--dump-single-json",
        *client_args,
        video_url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                check=True, timeout=timeout)
        data = json.loads(result.stdout)
        return data.get("formats", []), None
    except subprocess.TimeoutExpired:
        return [], "timeout while listing formats"
    except subprocess.CalledProcessError as exc:
        return [], (exc.stderr or "").strip()
    except (json.JSONDecodeError, OSError) as exc:
        return [], str(exc)

def pick_format_candidates(formats):
    """Choose concrete format IDs from a real inventory.

    Returns an ordered list of (label, format_string). Preferences:
      1. H.264 video <= MAX_HEIGHT, progressive/DASH transport, highest quality
      2. same but HLS transport is acceptable
      3. any H.264 video regardless of height
      4. a pre-muxed H.264 mp4
    HLS vs DASH is only a TRANSPORT difference -- itag 136 is H.264 720p either
    way -- so DASH is preferred for a clean progress bar, never required.
    """
    def is_h264(fmt):
        return (fmt.get("vcodec") or "").startswith(("avc", "h264"))

    def is_video_only(fmt):
        return is_h264(fmt) and (fmt.get("acodec") or "none") == "none"

    def is_audio_only(fmt):
        return ((fmt.get("vcodec") or "none") == "none"
                and (fmt.get("acodec") or "none") != "none")

    def is_dash(fmt):
        return "m3u8" not in (fmt.get("protocol") or "")

    def quality(fmt):
        return (fmt.get("height") or 0, fmt.get("tbr") or 0)

    videos = [f for f in formats if is_video_only(f)]
    audios = [f for f in formats if is_audio_only(f)]

    # Prefer m4a/mp4 audio so the mp4 mux never needs a re-encode
    def audio_rank(fmt):
        ext_bonus = 1 if (fmt.get("ext") or "") == "m4a" else 0
        return (ext_bonus, 1 if is_dash(fmt) else 0, fmt.get("tbr") or 0)

    audios.sort(key=audio_rank, reverse=True)
    best_audio = audios[0]["format_id"] if audios else None

    candidates = []

    def add_video_tier(pool, label):
        if not pool or not best_audio:
            return
        pool = sorted(pool, key=quality, reverse=True)
        top = pool[0]
        combo = f"{top['format_id']}+{best_audio}"
        height = top.get("height") or "?"
        proto = "DASH" if is_dash(top) else "HLS"
        candidates.append((f"{label} {height}p {proto}", combo))

    capped = [f for f in videos if (f.get("height") or 0) <= MAX_HEIGHT]
    add_video_tier([f for f in capped if is_dash(f)], "H.264")
    add_video_tier([f for f in capped if not is_dash(f)], "H.264")
    add_video_tier(videos, "H.264(高解像度)")

    # Last resort: an already-muxed H.264 file
    muxed = [f for f in formats
             if is_h264(f) and (f.get("acodec") or "none") != "none"]
    if muxed:
        muxed.sort(key=quality, reverse=True)
        top = muxed[0]
        candidates.append((f"結合済み {top.get('height') or '?'}p",
                           top["format_id"]))

    # De-duplicate while preserving order
    seen, unique = set(), []
    for label, fmt in candidates:
        if fmt in seen:
            continue
        seen.add(fmt)
        unique.append((label, fmt))
    return unique

# ---------------------------------------------------------------- client discovery
#
# Cheap metadata-only probing, done ONCE per run (not once per video, and not
# once per attempt). All CLIENT_VARIANTS are queried IN PARALLEL since each
# call is dominated by network latency, not CPU -- this is the "test first,
# then act" step, done fast instead of sequentially guessing during download.

CLIENT_CACHE_PATH = os.path.join(LOG_ROOT, "client_cache.json")

def load_client_cache():
    """{channel_handle: {"client_label": ..., "updated": iso-timestamp}} or {}."""
    if not os.path.exists(CLIENT_CACHE_PATH):
        return {}
    try:
        with open(CLIENT_CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

def save_client_cache(cache):
    os.makedirs(LOG_ROOT, exist_ok=True)
    try:
        with open(CLIENT_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except OSError:
        pass

def variant_for_label(label):
    for variant in CLIENT_VARIANTS:
        if client_label(variant) == label:
            return variant
    return None

def probe_one_client(video_url, client_args):
    """Worker for the thread pool: (client_args, candidates, error)."""
    formats, error = fetch_format_inventory(video_url, client_args)
    if error:
        return client_args, [], error
    candidates = pick_format_candidates(formats)
    return client_args, candidates, None

def discover_working_client(mmdd, video_url):
    """Probe every CLIENT_VARIANTS entry in parallel against one video and
    return the first client that exposes at least one H.264 candidate,
    ordered by CLIENT_VARIANTS priority (not by which thread finished first).
    Returns (client_args, candidates) or (None, []) if nothing worked."""
    say(mmdd, f"  🔍 対応クライアントを並列診断中 "
              f"({len(CLIENT_VARIANTS)}クライアント同時)...")

    results = {}
    with ThreadPoolExecutor(max_workers=len(CLIENT_VARIANTS)) as pool:
        futures = {pool.submit(probe_one_client, video_url, c): c
                   for c in CLIENT_VARIANTS}
        for future in as_completed(futures):
            client_args, candidates, error = future.result()
            key = client_label(client_args)
            results[key] = (client_args, candidates, error)

    # Report in priority order, then pick the first usable one
    for client_args in CLIENT_VARIANTS:
        key = client_label(client_args)
        _, candidates, error = results.get(key, (client_args, [], "no result"))
        if error:
            say(mmdd, f"     ⚠️ {key}: 一覧取得失敗 ({error.splitlines()[0][:100]})")
        elif not candidates:
            say(mmdd, f"     ⚠️ {key}: H.264フォーマットなし")
        else:
            best_label = candidates[0][0]
            say(mmdd, f"     ✅ {key}: 利用可 ({best_label})")

    for client_args in CLIENT_VARIANTS:
        key = client_label(client_args)
        _, candidates, error = results.get(key, (client_args, [], "no result"))
        if not error and candidates:
            return client_args, candidates

    return None, []

# ---------------------------------------------------------------- download

def run_download(video_url, fmt, client_args, expected_filepath):
    """Run one yt-dlp download. Returns (ok: bool, stderr: str)."""
    command = [
        "yt-dlp", "--remote-components", "ejs:github",
        *QUIET_PROGRESS,
        "--continue",                 # resume from .part instead of restarting
        "--no-overwrites",
        "-r", DOWNLOAD_RATE_LIMIT,
        "--retries", "10",            # per-attempt network retries
        "--fragment-retries", "50",   # HLS fragments deserve more patience
        "--socket-timeout", "30",
        *client_args,
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", expected_filepath,
        video_url,
    ]
    try:
        subprocess.run(command, check=True)
        return True, ""
    except subprocess.CalledProcessError as exc:
        return False, (exc.stderr or "") if exc.stderr else "yt-dlp exited nonzero"

def try_candidates_for_client(mmdd, video, expected_filepath, label,
                              client_args, candidates, transient_failures):
    """Attempt each concrete format ID for one already-probed client.
    Returns "success", "exhausted" (move to next client), or an int giving the
    updated transient_failures count when neither of those apply."""
    name = client_label(client_args)
    preview = " / ".join(lbl for lbl, _ in candidates[:MAX_CANDIDATES_PER_CLIENT])
    say(mmdd, f"     📋 {name} の候補: {preview}")

    for cand_label, fmt in candidates[:MAX_CANDIDATES_PER_CLIENT]:
        say(mmdd, f"  📥 {label} DL開始 client={name} / {cand_label} ({fmt})")
        started = time.time()

        try:
            ok, stderr = run_download(video["url"], fmt, client_args, expected_filepath)
        except KeyboardInterrupt:
            say(mmdd, "  ⏹  中断されました。.part は残してあるので再実行で続行できます。")
            raise

        elapsed = time.time() - started

        if ok:
            good, reason = verify_download(expected_filepath, video["duration_sec"])
            if good:
                say(mmdd, f"  ✅ {label} 完了 ({reason}) 所要 {human_hms(elapsed)}")
                cleanup_stray_parts(expected_filepath, keep_parts=False)
                return "success", client_args
            say(mmdd, f"  ⚠️ {label} 検証で不合格: {reason}")
            try:
                os.remove(expected_filepath)
            except OSError:
                pass
            cleanup_stray_parts(expected_filepath, keep_parts=False)
            continue  # bad source: try the next format, no waiting

        write_log(mmdd, f"yt-dlp stderr: {stderr[:500]}")
        if is_fatal_error(stderr):
            say(mmdd, "     ⚠️ 恒久的なエラーのため待たずに次へ")
            cleanup_stray_parts(expected_filepath, keep_parts=False)
            continue

        # Transient (403, timeout, connection reset): keep .part and wait once
        say(mmdd, f"     ⚠️ 一時的なエラー（{human_hms(elapsed)} 経過）")
        cleanup_stray_parts(expected_filepath, keep_parts=True)
        wait = TRANSIENT_BACKOFF[min(transient_failures, len(TRANSIENT_BACKOFF) - 1)]
        transient_failures += 1
        say(mmdd, f"     ⏳ {wait} 秒待って再試行します...")
        try:
            time.sleep(wait)
        except KeyboardInterrupt:
            say(mmdd, "  ⏹  中断されました。")
            raise

        try:
            ok, stderr = run_download(video["url"], fmt, client_args, expected_filepath)
        except KeyboardInterrupt:
            say(mmdd, "  ⏹  中断されました。")
            raise
        if ok:
            good, reason = verify_download(expected_filepath, video["duration_sec"])
            if good:
                say(mmdd, f"  ✅ {label} 完了 ({reason})")
                cleanup_stray_parts(expected_filepath, keep_parts=False)
                return "success", client_args
        cleanup_stray_parts(expected_filepath, keep_parts=True)

    return "exhausted", transient_failures

def download_one_video(mmdd, video, expected_filepath, index, total, session_client=None):
    """Download a single video. Returns (success: bool, working_client or None).

    session_client, when given, is a client_args list already confirmed to
    work earlier in this run (e.g. for video 1 of the day) -- it is tried
    FIRST with no re-probing, so video 2+ of the same date skip diagnosis
    entirely. Only on failure does this fall back to a fresh parallel probe
    of every CLIENT_VARIANTS entry.
    """
    label = f"[{index}/{total}]"

    if os.path.exists(expected_filepath):
        ok, reason = verify_download(expected_filepath, video["duration_sec"])
        if ok:
            say(mmdd, f"  ✅ {label} 既存ファイルは正常です ({reason})")
            return True, session_client
        say(mmdd, f"  ♻️ {label} 既存ファイルが不完全 ({reason}) → 破棄して再取得します")
        try:
            os.remove(expected_filepath)
        except OSError:
            pass
        cleanup_stray_parts(expected_filepath, keep_parts=False)

    transient_failures = 0

    # Fast path: reuse the client that already worked earlier this session
    if session_client is not None:
        name = client_label(session_client)
        say(mmdd, f"  ⚡ {label} 前回成功した client={name} を再利用（診断スキップ）")
        formats, error = fetch_format_inventory(video["url"], session_client)
        candidates = pick_format_candidates(formats) if not error else []
        if candidates:
            result, extra = try_candidates_for_client(
                mmdd, video, expected_filepath, label,
                session_client, candidates, transient_failures)
            if result == "success":
                return True, extra
            transient_failures = extra if isinstance(extra, int) else transient_failures
            say(mmdd, "     ⚠️ 前回のクライアントが今回は通らず、再診断します")
        else:
            say(mmdd, "     ⚠️ 前回のクライアントが今回は使えず、再診断します")

    # Slow path: probe every client in parallel, then try candidates in order
    working_client, candidates = discover_working_client(mmdd, video["url"])
    tried_already = {client_label(session_client)} if session_client else set()

    if working_client is not None and client_label(working_client) not in tried_already:
        result, extra = try_candidates_for_client(
            mmdd, video, expected_filepath, label,
            working_client, candidates, transient_failures)
        if result == "success":
            return True, extra
        transient_failures = extra if isinstance(extra, int) else transient_failures

    # Discovered client failed at download time (rare: probing lies) -- walk
    # the remaining CLIENT_VARIANTS in priority order as a last resort.
    for client_args in CLIENT_VARIANTS:
        name = client_label(client_args)
        if name in tried_already or (working_client and name == client_label(working_client)):
            continue
        formats, error = fetch_format_inventory(video["url"], client_args)
        if error or not (candidates := pick_format_candidates(formats)):
            continue
        result, extra = try_candidates_for_client(
            mmdd, video, expected_filepath, label,
            client_args, candidates, transient_failures)
        if result == "success":
            return True, extra
        transient_failures = extra if isinstance(extra, int) else transient_failures

    say(mmdd, f"  ❌ {label} すべてのクライアント/フォーマットで失敗しました。")
    say(mmdd, f"     手動DL用URL: {video['url']}")
    say(mmdd, f"     手動で取得した場合は {expected_filepath} に置いて再実行してください。")
    return False, None

# ---------------------------------------------------------------- chat

def extract_chat_messages(chat_path, start_time):
    """Parse a live_chat JSON file into [(real_time, formatted_text), ...]."""
    messages_list = []
    if not os.path.exists(chat_path):
        return messages_list

    with open(chat_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                action_wrapper = data.get("replayChatItemAction", {})
                offset_msec = int(action_wrapper.get("videoOffsetTimeMsec", 0))

                # Add elapsed milliseconds to the stream start time
                real_time = start_time + timedelta(milliseconds=offset_msec)
                time_str = real_time.strftime("%H:%M:%S")

                actions = action_wrapper.get("actions", [])
                if not actions:
                    continue

                item = actions[0].get("addChatItemAction", {}).get("item", {})
                renderer = (item.get("liveChatTextMessageRenderer", {})
                            or item.get("liveChatPaidMessageRenderer", {}))
                if not renderer:
                    continue

                author = renderer.get("authorName", {}).get("simpleText", "Unknown")
                message_runs = renderer.get("message", {}).get("runs", [])
                message = "".join(run.get("text", "") for run in message_runs)

                messages_list.append((real_time, f"[{time_str}] {author}: {message}"))
            except Exception:
                continue
    return messages_list

def process_daily_chat(mmdd, video_list):
    """Download and merge every stream's chat log into one wall-clock timeline."""
    say(mmdd, f"\n💬 【{mmdd[:2]}月{mmdd[2:]}日】のチャットログを統合処理中...")

    all_messages = []
    for idx, video in enumerate(video_list, 1):
        say(mmdd, f"  📥 [{idx}/{len(video_list)}] チャットDL中: {video['title']}")
        temp_base = f"temp_chat_{mmdd}_{idx}"

        for client_args in CLIENT_VARIANTS:
            command = [
                "yt-dlp", "--remote-components", "ejs:github",
                *QUIET_PROGRESS,
                *client_args,
                "--skip-download",
                "--write-subs",
                "--sub-langs", "live_chat",
                "-o", temp_base,
                video["url"],
            ]
            try:
                subprocess.run(command, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except subprocess.CalledProcessError:
                continue
            if glob.glob(f"{temp_base}.*live_chat*.json"):
                break

        found_files = glob.glob(f"{temp_base}.*live_chat*.json")
        if not found_files:
            say(mmdd, "    ⚠️ チャットが取得できませんでした（動画処理は継続します）。")
            continue

        chat_file = found_files[0]
        all_messages.extend(extract_chat_messages(chat_file, video["start_time"]))
        try:
            os.remove(chat_file)
        except OSError:
            pass

    if not all_messages:
        say(mmdd, "  ⚠️ コメントが1件も取得できなかったため、テキストは作成しません。")
        return

    all_messages.sort(key=lambda x: x[0])

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    output_txt_path = os.path.join(OUTPUT_ROOT, f"{mmdd}_output.txt")
    with open(output_txt_path, "w", encoding="utf-8") as wf:
        for _, text in all_messages:
            wf.write(text + "\n")

    say(mmdd, f"  💾 チャットマージ完了 ➔ {output_txt_path} "
              f"（総コメント数: {len(all_messages)}件）")

# ---------------------------------------------------------------- per-date

def process_one_date(mmdd, video_list):
    """download -> verify -> fast_bird_pipeline -> chat.
    Returns True (done), False (failed), or None (skipped, retry later)."""
    say(mmdd, f"\n🎯 処理対象: 【{mmdd[:2]}月{mmdd[2:]}日】 (枠数: {len(video_list)}本)")

    total_expected = 0
    for idx, video in enumerate(video_list, 1):
        say(mmdd, f"  {idx}本目 ({video['start_time'].strftime('%H:%M')}開始): "
                  f"状況={video['status']} 長さ={human_hms(video['duration_sec'])}")
        total_expected += video["duration_sec"]
        if video["status"] != "none":
            say(mmdd, "  ⏭  まだ配信中/処理中の動画があるため、この日付はスキップします。")
            return None

    say(mmdd, f"  📊 合計 {human_hms(total_expected)} の映像を取得します。")

    target_dir = os.path.join(INPUT_ROOT, mmdd)
    os.makedirs(target_dir, exist_ok=True)

    # Reuse whatever client worked last time this channel was processed, so a
    # cold run doesn't re-diagnose from scratch every single day.
    cache = load_client_cache()
    cached_label = cache.get(CHANNEL_HANDLE, {}).get("client_label")
    session_client = variant_for_label(cached_label) if cached_label else None
    if session_client is not None:
        say(mmdd, f"  💾 前回のキャッシュから client={cached_label} を試します")

    for idx, video in enumerate(video_list, 1):
        expected_filepath = os.path.join(target_dir, f"{mmdd}-{idx}.mp4")
        ok, working_client = download_one_video(
            mmdd, video, expected_filepath, idx, len(video_list), session_client)
        if not ok:
            say(mmdd, f"  🛑 {mmdd} の取得を中止します"
                      f"（再実行すればDL済み分はスキップして続きから再開します）。")
            return False
        if working_client is not None:
            session_client = working_client  # reuse for the next video today
            cache[CHANNEL_HANDLE] = {
                "client_label": client_label(working_client),
                "updated": datetime.now().isoformat(timespec="seconds"),
            }
            save_client_cache(cache)

    # Guard the concat step: every file should share codec and resolution
    codecs = set()
    for idx in range(1, len(video_list) + 1):
        info = probe_media(os.path.join(target_dir, f"{mmdd}-{idx}.mp4"))
        if info:
            codecs.add((info["vcodec"], info["height"]))
    if len(codecs) > 1:
        say(mmdd, f"  ⚠️ 動画間でコーデック/解像度が不揃いです: {sorted(codecs)}")
        say(mmdd, "     結合時に再エンコードが発生する可能性があります。")

    say(mmdd, f"\n🚀 {FAST_BIRD_PIPELINE} {mmdd} を実行します...")
    print("=" * 75)
    try:
        subprocess.run([sys.executable, FAST_BIRD_PIPELINE, mmdd], check=True)
    except subprocess.CalledProcessError:
        print("=" * 75)
        say(mmdd, f"❌ {FAST_BIRD_PIPELINE} の実行中にエラーが発生しました。")
        say(mmdd, "   （動画は input/ に残っているので、再実行時はDLをやり直しません）")
        return False

    print("=" * 75)
    say(mmdd, f"🏁 【{mmdd[:2]}月{mmdd[2:]}日】完了! ➔ {OUTPUT_ROOT}/{mmdd}_output.mp4")

    # Chat failures must never invalidate a finished video
    try:
        process_daily_chat(mmdd, video_list)
    except Exception as exc:
        say(mmdd, f"  ⚠️ チャット処理で例外が出ましたが、動画は完成しています: {exc}")

    return True

# ---------------------------------------------------------------- main

def check_prerequisites():
    """Fail loudly and immediately if a required binary is missing, rather
    than after an hour of downloading."""
    missing = [tool for tool in ("yt-dlp", "ffmpeg", "ffprobe")
               if shutil.which(tool) is None]
    if missing:
        print(f"❌ 必要なコマンドが見つかりません: {', '.join(missing)}")
        return False
    return True

def main():
    if not check_prerequisites():
        return

    api_key = load_api_key()
    if not api_key:
        return

    channel_id = get_channel_id(api_key, CHANNEL_HANDLE)
    if not channel_id:
        print("❌ チャンネルIDが取得できませんでした。")
        return

    started = time.time()
    print("🦅 インテリジェンス・自動化パイプラインを起動しました。")
    print(f"※過去{LOOKBACK_DAYS}日以内の未処理日付をすべて処理します。")
    print(f"※DL前に{len(CLIENT_VARIANTS)}クライアントを並列診断し、通ったものだけ使います"
          f"（前回成功したクライアントはキャッシュから再利用）。")
    print("-" * 75)

    try:
        unprocessed = find_unprocessed_dates(api_key, channel_id)
    except requests.RequestException as exc:
        print(f"❌ YouTube API への接続に失敗しました: {exc}")
        return

    if not unprocessed:
        print(f"🟢 過去{LOOKBACK_DAYS}日分はすべて処理済みです。やることがありません。")
        return

    print(f"\n📋 未処理の日付: {[m for m, _ in unprocessed]} (古い順に処理)")

    succeeded, failed, skipped = [], [], []
    for mmdd, video_list in unprocessed:
        try:
            result = process_one_date(mmdd, video_list)
        except KeyboardInterrupt:
            print("\n⏹  中断されました。再実行すれば途中から再開します。")
            break
        except Exception as exc:
            say(mmdd, f"❌ 想定外のエラー: {exc}")
            result = False

        if result is True:
            succeeded.append(mmdd)
        elif result is None:
            skipped.append(mmdd)
        else:
            failed.append(mmdd)

    print("\n" + "=" * 75)
    print(f"🏁 全処理完了（総経過 {human_hms(time.time() - started)}）")
    if succeeded:
        print(f"  ✅ 成功: {succeeded}")
    if skipped:
        print(f"  ⏭  スキップ(配信未終了): {skipped}  ← 後で再実行してください")
    if failed:
        print(f"  ❌ 失敗: {failed}")
        print(f"     詳細は {LOG_ROOT}/<日付>_pipeline.log を確認してください。")
        print("     再実行すればDL済みファイルはスキップされ、続きから再開します。")

if __name__ == "__main__":
    main()