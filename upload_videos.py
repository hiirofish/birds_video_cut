"""Upload a day's finished videos to YouTube and add each to its playlist.

The channel gets two kinds of video per day, each filed into its own playlist:

  short  marugoto/MMDD_short[_partN]_DAY<n>_<title>.mp4  (compile_shorts.py, comment-based)
         -> the playlist whose title contains 「2年目」 (【2026年・2年目】帰ってきたコシアカツバメ…)
  full   marugoto/MMDD_output.mp4  (smart_bird_pipeline.py, motion detection)
         -> the playlist whose title contains 「動体検知」 (【全編記録】動体検知で見る…)

Whether YouTube treats a file as a Short or a regular video follows from its
length and shape alone, so nothing here marks it. Titles, descriptions and
tags reproduce what has been posted by hand until now; the description/tag
text lives in sozai/ so it can be edited without touching this file.

Usage: python upload_videos.py MMDD [--privacy private|unlisted|public] [--only short|full] [--dry-run]

Auth: uploader_credentials.json (OAuth desktop client) -> uploader_token.json.
The first run prints a URL to open in the browser.

Until the Cloud project passes YouTube's API compliance audit, YouTube locks
every video uploaded through the API to private, whatever --privacy says.
"""
import argparse
import json
import os
import random
import sys
import time
import unicodedata
from datetime import datetime

import httplib2
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from compile_shorts import OUT_DIR, SOZAI_DIR, build_out_path, load_date_list

CREDENTIALS_FILE = "uploader_credentials.json"
TOKEN_FILE = "uploader_token.json"
# "youtube" covers uploading as well as reading and adding to our playlists;
# youtube.upload alone can't touch playlists.
SCOPES = ["https://www.googleapis.com/auth/youtube"]
CHANNEL_HANDLE = "@take1bit"

# Update both when the next season's playlist starts -- otherwise this year's
# playlist goes on matching "2年目" and new videos land in the old one.
SEASON = "2年目"
PLAYLIST_KEYWORDS = {"short": "2年目", "full": "動体検知"}

DESCRIPTION_FILES = {
    "short": os.path.join(SOZAI_DIR, "description_short.txt"),
    "full": os.path.join(SOZAI_DIR, "description_full.txt"),
}
TAGS_FILE = os.path.join(SOZAI_DIR, "upload_tags.txt")
CATEGORY_ID = "15"  # Pets & Animals
MAX_TITLE_LEN = 100

CHUNK_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 8
RETRIABLE_STATUS = {500, 502, 503, 504}
# Dropped or timed-out connections (ConnectionError, TimeoutError and ssl
# errors are all OSError). The Wi-Fi here hangs now and then, so expect them.
RETRIABLE_EXCEPTIONS = (OSError, httplib2.HttpLib2Error)

HTTP_ERROR_HINTS = {
    "quotaExceeded": "今日のAPI利用量の上限です。太平洋時間0時（日本時間16〜17時）にリセットされます",
    "uploadLimitExceeded": "チャンネルの1日のアップロード本数の上限です",
    "youtubeSignupRequired": "認証したGoogleアカウントにYouTubeチャンネルがありません",
    "invalidTitle": "タイトルが不正です（空・100文字超・<>を含む）",
    "invalidDescription": "概要欄が不正です（5000バイト超・<>を含む）",
    "invalidTags": "タグが不正です（合計500文字超など）",
    "forbidden": "権限がありません。認証したチャンネルを確認してください",
    "insufficientPermissions": f"許可の範囲が足りません。{TOKEN_FILE} を消して認証し直してください",
    "playlistNotFound": "再生リストが見つかりません",
}


def short_title(day, topic):
    return (f"【{SEASON} {day} ハイライト版】「{topic}」｜コシアカツバメ（トックリツバメ）巣の定点観察"
            f" | Red-rumped Swallow Nest Cam (Highlights)")


def full_title(day):
    # The hand-typed full-version titles put a space after DAY; the shorts don't.
    return (f"【{SEASON} {day.replace('DAY', 'DAY ')} まるごと版】"
            f"コシアカツバメ（トックリツバメ）巣の定点観察#コシアカツバメ #トックリツバメ #野鳥観察")


def find_videos(mmdd):
    """[(kind, path, title)] for the day's finished files.

    The short's path is rebuilt from clips.json exactly the way
    compile_shorts.py named it, rather than globbed, so a leftover file from
    an earlier title can't be uploaded by mistake."""
    _, day_text = load_date_list(mmdd)
    if "?" in day_text:
        print(f"❌ {mmdd} が sozai/date_list.txt にありません（DAY数が分からない）")
        sys.exit(1)
    day = day_text.translate(str.maketrans("ＤＡＹ０１２３４５６７８９", "DAY0123456789"))
    videos = []

    full_path = os.path.join(OUT_DIR, f"{mmdd}_output.mp4")
    if os.path.exists(full_path):
        videos.append(("full", full_path, full_title(day)))
    else:
        print(f"⚠️  {full_path} がありません（まるごと版は飛ばします）")

    shorts = []
    clips_path = os.path.join(OUT_DIR, f"{mmdd}_clips.json")
    if os.path.exists(clips_path):
        with open(clips_path, encoding="utf-8") as f:
            data = json.load(f)
        title = data.get("title", "")
        title_part2 = data.get("title_part2", "") or title  # same fallback as compile_shorts.py
        single = build_out_path(mmdd, day_text, title)
        parts = [(build_out_path(mmdd, day_text, title, 1), title),
                 (build_out_path(mmdd, day_text, title_part2, 2), title_part2)]
        if os.path.exists(single):
            shorts = [(single, title)]
        elif all(os.path.exists(path) for path, _ in parts):
            shorts = parts
    if not shorts:
        print(f"⚠️  {mmdd} のショート版が見つかりません（compile_shorts.py 未実行？）")
    videos += [("short", path, short_title(day, topic)) for path, topic in shorts]
    return videos


def get_credentials():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except RefreshError as exc:
            # Revoked, or the 7-day expiry of an OAuth consent screen left in "Testing".
            print(f"⚠️  トークンを更新できませんでした（{exc}）。認証し直します")
            creds = None
    if not creds or not creds.valid:
        if not os.path.exists(CREDENTIALS_FILE):
            print(f"❌ {CREDENTIALS_FILE}（OAuthクライアント）がありません")
            sys.exit(1)
        flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
        # WSL can't open a browser itself: the URL is opened by hand in Windows,
        # and the redirect to localhost reaches this process via WSL's forwarding.
        creds = flow.run_local_server(
            port=0, open_browser=False,
            authorization_prompt_message="🔑 このURLをブラウザで開き、@Take1bit のアカウントで許可してください:\n{url}\n")
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    return creds


def check_channel(youtube):
    items = youtube.channels().list(part="snippet", mine=True).execute().get("items", [])
    handle = items[0]["snippet"].get("customUrl", "") if items else ""
    if handle.lower() != CHANNEL_HANDLE:
        print(f"❌ 認証したチャンネルが {CHANNEL_HANDLE} ではありません（{handle or 'チャンネル無し'}）。"
              f"{TOKEN_FILE} を消して、正しいチャンネルで認証し直してください")
        sys.exit(1)
    print(f"📺 チャンネル: {items[0]['snippet']['title']} ({handle})")


def find_playlist(youtube, keyword):
    """(id, title) of the one playlist of ours whose title contains keyword.

    Matches snippet.title, the default-language (Japanese) title. The playlists
    also carry English localizations, which never contain the keyword."""
    key = unicodedata.normalize("NFKC", keyword)
    matches = []
    request = youtube.playlists().list(part="snippet", mine=True, maxResults=50)
    while request is not None:
        response = request.execute()
        matches += [(p["id"], p["snippet"]["title"]) for p in response.get("items", [])
                    if key in unicodedata.normalize("NFKC", p["snippet"]["title"])]
        request = youtube.playlists().list_next(request, response)
    if len(matches) != 1:
        print(f"❌ 「{keyword}」を含む再生リストが {len(matches)} 件あります（1件のはず）: "
              f"{[title for _, title in matches]}")
        sys.exit(1)
    return matches[0]


def upload(youtube, path, body):
    """Resumable upload with retries; returns the new video ID. A retry picks
    the same upload session back up, so a dropped connection doesn't start a
    second copy of the video."""
    media = MediaFileUpload(path, mimetype="video/mp4", chunksize=CHUNK_SIZE, resumable=True)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)
    response, retries = None, 0
    while response is None:
        try:
            status, response = request.next_chunk()
        except HttpError as exc:
            if exc.resp.status not in RETRIABLE_STATUS:
                raise
            error = f"HTTP {exc.resp.status}"
        except RETRIABLE_EXCEPTIONS as exc:
            error = f"{type(exc).__name__}: {exc}"
        else:
            if status:
                print(f"   ⬆️  {status.progress() * 100:.0f}%")
            retries = 0
            continue
        retries += 1
        if retries > MAX_RETRIES:
            print(f"❌ {MAX_RETRIES}回再試行しても失敗しました（{error}）")
            sys.exit(1)
        wait = min(2 ** retries, 60) + random.random()
        print(f"   ⚠️  {error} … {wait:.0f}秒後に再試行 ({retries}/{MAX_RETRIES})")
        time.sleep(wait)
    return response["id"]


def add_to_playlist(youtube, playlist_id, video_id):
    """False when the video is already in the playlist -- YouTube would
    happily add it a second time, so check first.

    Newest goes first. A manually sorted playlist appends to the end unless
    told position 0; an automatically sorted one places the video itself and
    rejects any position with manualSortRequired, so fall back to no position."""
    existing = youtube.playlistItems().list(
        part="id", playlistId=playlist_id, videoId=video_id).execute()
    if existing.get("items"):
        return False
    snippet = {
        "playlistId": playlist_id,
        "resourceId": {"kind": "youtube#video", "videoId": video_id},
        "position": 0,
    }
    try:
        youtube.playlistItems().insert(part="snippet", body={"snippet": snippet}).execute()
    except HttpError as exc:
        if b"manualSortRequired" not in exc.content:
            raise
        del snippet["position"]
        youtube.playlistItems().insert(part="snippet", body={"snippet": snippet}).execute()
        print("   ℹ️  自動並べ替えの再生リストなので、並び順はYouTubeに任せました")
    return True


def describe_http_error(exc):
    try:
        error = json.loads(exc.content)["error"]
        reason = (error.get("errors") or [{}])[0].get("reason", "")
        message = error.get("message", "")
    except (ValueError, KeyError, TypeError):
        reason, message = "", str(exc)
    text = f"{exc.resp.status} {reason}: {message}"
    if reason in HTTP_ERROR_HINTS:
        text += f"\n   → {HTTP_ERROR_HINTS[reason]}"
    return text


def title_key(title):
    """Compare titles ignoring width and spacing, which vary in the
    hand-typed ones (e.g. "DAY 127  まるごと版")."""
    return " ".join(unicodedata.normalize("NFKC", title).split())


def playlist_videos_by_title(youtube, playlist_id):
    """{title_key: video_id} for everything already in the playlist, so a day
    that went up by hand -- which the ledger knows nothing about -- isn't
    posted twice. Titles carry the DAY number, so they are unique per day."""
    videos = {}
    request = youtube.playlistItems().list(part="snippet", playlistId=playlist_id, maxResults=50)
    while request is not None:
        response = request.execute()
        for item in response.get("items", []):
            snippet = item["snippet"]
            videos[title_key(snippet["title"])] = snippet["resourceId"]["videoId"]
        request = youtube.playlistItems().list_next(request, response)
    return videos


def ledger_path(mmdd):
    return os.path.join(OUT_DIR, f"{mmdd}_upload.json")


def load_ledger(mmdd):
    """What has already gone up for this date, keyed by file name. This is
    what stops a re-run (say, after the playlist step failed) from uploading
    the same video twice."""
    path = ledger_path(mmdd)
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_ledger(mmdd, ledger):
    with open(ledger_path(mmdd), "w", encoding="utf-8") as f:
        json.dump(ledger, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="その日の完成動画をYouTubeにアップして再生リストに追加する")
    parser.add_argument("mmdd")
    parser.add_argument("--privacy", choices=["private", "unlisted", "public"], default="private")
    parser.add_argument("--only", choices=["short", "full"], help="片方の種類だけアップする")
    parser.add_argument("--dry-run", action="store_true",
                        help="認証・チャンネル・再生リストを確認して、アップする内容を表示するだけ")
    args = parser.parse_args()

    videos = [v for v in find_videos(args.mmdd) if args.only in (None, v[0])]
    if not videos:
        print("❌ アップする動画がありません")
        sys.exit(1)
    for _, _, title in videos:
        if len(title) > MAX_TITLE_LEN or "<" in title or ">" in title:
            print(f"❌ タイトルが長すぎるか <> を含みます（{len(title)}文字）: {title}")
            sys.exit(1)

    descriptions = {}
    for kind, path in DESCRIPTION_FILES.items():
        with open(path, encoding="utf-8") as f:
            descriptions[kind] = f.read().rstrip()
    with open(TAGS_FILE, encoding="utf-8") as f:
        tags = [line.strip() for line in f if line.strip()]

    try:
        youtube = build("youtube", "v3", credentials=get_credentials())
        check_channel(youtube)
        playlists = {kind: find_playlist(youtube, PLAYLIST_KEYWORDS[kind])
                     for kind in sorted({v[0] for v in videos})}
        ledger = load_ledger(args.mmdd)
        already = {kind: playlist_videos_by_title(youtube, playlist_id)
                   for kind, (playlist_id, _) in playlists.items()}

        for kind, path, title in videos:
            name = os.path.basename(path)
            playlist_id, playlist_title = playlists[kind]
            print(f"\n🎬 {name} ({os.path.getsize(path) / 1e6:.1f}MB)")
            print(f"   タイトル: {title}（{len(title)}文字）")
            print(f"   概要欄: {DESCRIPTION_FILES[kind]}（{len(descriptions[kind])}文字）/ タグ {len(tags)}個")
            print(f"   再生リスト: {playlist_title}")
            print(f"   公開設定: {args.privacy}")

            entry = ledger.get(name)
            duplicate = already[kind].get(title_key(title))
            if duplicate and not entry:
                print(f"   ⏭  同じタイトルの動画がもう再生リストにあります（手動でアップ済み？）: "
                      f"https://youtu.be/{duplicate}")
                continue
            if args.dry_run:
                continue

            if entry and entry["privacy"] != args.privacy:
                print(f"❌ 既に {entry['privacy']} でアップ済みです（https://youtu.be/{entry['video_id']}）。"
                      f"上げ直すなら {ledger_path(args.mmdd)} から {name} を消してください")
                sys.exit(1)
            if entry:
                print(f"   ⏭  アップ済み: https://youtu.be/{entry['video_id']}")
            else:
                body = {
                    "snippet": {
                        "title": title, "description": descriptions[kind], "tags": tags,
                        "categoryId": CATEGORY_ID, "defaultLanguage": "ja", "defaultAudioLanguage": "ja",
                    },
                    "status": {
                        "privacyStatus": args.privacy,
                        "selfDeclaredMadeForKids": False,
                        "containsSyntheticMedia": False,  # real camera footage
                    },
                }
                video_id = upload(youtube, path, body)
                entry = ledger[name] = {"video_id": video_id, "privacy": args.privacy, "title": title,
                                        "uploaded_at": datetime.now().isoformat(timespec="seconds")}
                save_ledger(args.mmdd, ledger)
                print(f"   ✅ アップロード完了: https://youtu.be/{video_id}")

            if entry.get("playlist_id") != playlist_id:
                added = add_to_playlist(youtube, playlist_id, entry["video_id"])
                entry["playlist_id"] = playlist_id
                save_ledger(args.mmdd, ledger)
                print("   ✅ 再生リストに追加しました" if added else "   ⏭  再生リストに追加済み")
    except HttpError as exc:
        print(f"❌ YouTube APIエラー: {describe_http_error(exc)}")
        sys.exit(1)
    except RefreshError as exc:
        print(f"❌ 認証エラー（{exc}）。{TOKEN_FILE} を消して認証し直してください")
        sys.exit(1)


if __name__ == "__main__":
    main()
