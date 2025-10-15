import streamlit as st
st.set_page_config(page_title="🚫 잠금", layout="wide", initial_sidebar_state="collapsed")
st.title("🚫 이 앱은 현재 잠겨 있습니다.")
st.caption("문의: 미디어)디지털마케팅팀 데이터파트 김호범")
st.stop()


# -*- coding: utf-8 -*-
# 📊 유튜브 반응 리포트: AI 댓글요약 (Streamlit Cloud용 / GitHub 세션 아카이브)
# - 메모리 최적화: 댓글 수집을 CSV로 스트리밍 저장 (DataFrame 메모리 피크 방지)
# - 시각화: CSV 청크 집계(시간대/일자별, 키워드 버블, 작성자 Top10, 좋아요 Top10)
# - 탭 내비 및 세션 로드 안정화: safe_rerun() 래퍼로 Streamlit 버전 차 안전 대응
# - 고급 모드: AI 분석 완료 후에만 정량요약 표시(중첩 expander 방지)

import streamlit as st
import pandas as pd
import os, json, re, time, base64, requests, gc
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from uuid import uuid4

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import google.generativeai as genai

import plotly.express as px
from plotly import graph_objects as go
import circlify
import stopwordsiso as stopwords
from kiwipiepy import Kiwi
import numpy as np

try:
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
except Exception:
    ILLEGAL_CHARACTERS_RE = None

# --- Streamlit rerun 호환 래퍼 ---
def safe_rerun():
    fn = getattr(st, "rerun", None)
    if callable(fn):
        return fn()
    fn_old = getattr(st, "experimental_rerun", None)
    if callable(fn_old):
        return fn_old()
    raise RuntimeError("No rerun function available in this Streamlit build.")

# ===================== 기본 경로(Cloud) =====================
BASE_DIR = "/tmp"  # Streamlit Cloud는 /tmp만 쓰기 가능(휘발성)
SESS_DIR = os.path.join(BASE_DIR, "sessions")
os.makedirs(SESS_DIR, exist_ok=True)

# ===================== 비밀키 / 파라미터 =====================
_YT_FALLBACK = []
_GEM_FALLBACK = []

YT_API_KEYS = list(st.secrets.get("YT_API_KEYS", [])) or _YT_FALLBACK
GEMINI_API_KEYS = list(st.secrets.get("GEMINI_API_KEYS", [])) or _GEM_FALLBACK
GEMINI_MODEL = "gemini-2.0-flash-lite"
GEMINI_TIMEOUT = 120
GEMINI_MAX_TOKENS = 2048

# --- GitHub 저장소 설정 ---
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITHUB_REPO = st.secrets.get("GITHUB_REPO", "")
GITHUB_BRANCH = st.secrets.get("GITHUB_BRANCH", "main")

def _gh_headers(token: str):
    h = {"Accept": "application/vnd.github+json"}
    if token:
        h["Authorization"] = f"token {token}"
    return h

def github_upload_file(repo, branch, path_in_repo, local_path, token):
    """Contents API: PUT /repos/{owner}/{repo}/contents/{path}"""
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}"
    with open(local_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf-8")
    headers = _gh_headers(token)
    get_resp = requests.get(url + f"?ref={branch}", headers=headers)
    sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None
    data = {"message": f"upload {path_in_repo}", "content": content, "branch": branch}
    if sha: data["sha"] = sha
    resp = requests.put(url, headers=headers, json=data)
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"GitHub 업로드 실패: {resp.text}")
    return resp.json()

def github_list_dir(repo, branch, folder, token):
    url = f"https://api.github.com/repos/{repo}/contents/{folder}?ref={branch}"
    headers = _gh_headers(token)
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        return []
    return resp.json()

def github_download_file(repo, branch, path_in_repo, token, local_path):
    url = f"https://api.github.com/repos/{repo}/contents/{path_in_repo}?ref={branch}"
    headers = _gh_headers(token)
    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        content = base64.b64decode(data["content"])
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(content)
        return True
    return False

# 수집 상한(필요시 조정)
MAX_TOTAL_COMMENTS = 200_000
MAX_COMMENTS_PER_VIDEO = 5_000

# ===================== 동시 실행 1 슬롯(락 파일) =====================
LOCK_PATH = os.path.join(BASE_DIR, "ytccai.busy.lock")

def try_acquire_lock(ttl=7200):
    if os.path.exists(LOCK_PATH):
        try:
            if time.time() - os.path.getmtime(LOCK_PATH) > ttl:
                os.remove(LOCK_PATH)
        except:
            pass
    if os.path.exists(LOCK_PATH):
        return False
    open(LOCK_PATH, "w").close()
    return True

def refresh_lock():
    try: os.utime(LOCK_PATH, None)
    except: pass

def release_lock():
    try:
        if os.path.exists(LOCK_PATH):
            os.remove(LOCK_PATH)
    except:
        pass

def lock_guard_start_or_warn():
    if not try_acquire_lock():
        st.warning("다른 사용자가 작업 중입니다. 잠시 후 다시 시도하세요.")
        return False
    return True

# ===================== 기본 UI 세팅 =====================
st.set_page_config(page_title="📊 유튜브 반응 리포트: AI 댓글요약", layout="wide", initial_sidebar_state="collapsed")
st.title("📊 유튜브 반응 분석: AI 댓글요약")
st.caption("문의사항: 미디어)디지털마케팅팀 데이터파트")

_YT_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')
def _kst_tz(): return timezone(timedelta(hours=9))
def kst_to_rfc3339_utc(dt_kst: datetime) -> str:
    if dt_kst.tzinfo is None: dt_kst = dt_kst.replace(tzinfo=_kst_tz())
    return dt_kst.astimezone(timezone.utc).isoformat().replace("+00:00","Z")

def clean_illegal(val):
    if isinstance(val, str) and ILLEGAL_CHARACTERS_RE is not None:
        return ILLEGAL_CHARACTERS_RE.sub('', val)
    return val

def clean_df_strings(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if not obj_cols: return df
    df2 = df.copy()
    for c in obj_cols:
        df2[c] = df2[c].map(clean_illegal)
    return df2

# ===================== 형태소/불용어 =====================
kiwi = Kiwi()
korean_stopwords = stopwords.stopwords("ko")

# ===================== 로그 no-op =====================
def append_log(*args, **kwargs): return

# ===================== 키 로테이터 =====================
class RotatingKeys:
    def __init__(self, keys, state_key: str, on_rotate=None, treat_as_strings: bool = True):
        cleaned = []
        for k in (keys or []):
            if k is None: continue
            if treat_as_strings and isinstance(k, str):
                ks = k.strip()
                if ks: cleaned.append(ks)
            else:
                cleaned.append(k)
        self.keys = cleaned[:10]
        self.state_key = state_key
        self.on_rotate = on_rotate
        idx = st.session_state.get(state_key, 0)
        self.idx = 0 if not self.keys else (idx % len(self.keys))
        st.session_state[state_key] = self.idx
    def current(self):
        if not self.keys: return None
        return self.keys[self.idx % len(self.keys)]
    def rotate(self):
        if not self.keys: return
        self.idx = (self.idx + 1) % len(self.keys)
        st.session_state[self.state_key] = self.idx
        if callable(self.on_rotate): self.on_rotate(self.idx, self.current())

# ===================== API 호출 래퍼 =====================
def is_youtube_quota_error(e: HttpError) -> bool:
    try:
        data = json.loads(getattr(e, "content", b"{}").decode("utf-8", errors="ignore"))
        status = getattr(getattr(e, 'resp', None), 'status', None)
        if status in (403, 429):
            reasons = [(err.get("reason") or "").lower() for err in data.get("error", {}).get("errors", [])]
            msg = (data.get("error", {}).get("message", "") or "").lower()
            quota_flags = ("quotaexceeded", "dailylimitexceeded", "ratelimitexceeded")
            if any(r in quota_flags for r in reasons): return True
            if "rate" in msg and "limit" in msg: return True
            if "quota" in msg: return True
        return False
    except Exception:
        return False

def with_retry(fn, tries=2, backoff=1.4):
    for i in range(tries):
        try:
            return fn()
        except HttpError as e:
            status = getattr(getattr(e, 'resp', None), 'status', None)
            if status in (400, 401, 403) and not is_youtube_quota_error(e):
                raise
            if i == tries - 1: raise
            time.sleep((i + 1) * backoff)
        except Exception:
            if i == tries - 1: raise
            time.sleep((i + 1) * backoff)

class RotatingYouTube:
    def __init__(self, keys, state_key="yt_key_idx", log=None):
        self.rot = RotatingKeys(keys, state_key, on_rotate=lambda i, k: log and log(f"🔁 YouTube 키 전환 → #{i+1}"))
        self.log = log
        self.service = None
        self._build_service()
    def _build_service(self):
        key = self.rot.current()
        if not key:
            raise RuntimeError("YouTube API Key가 비어 있습니다.")
        self.service = build("youtube", "v3", developerKey=key)
    def _rotate_and_rebuild(self):
        self.rot.rotate(); self._build_service()
    def execute(self, request_factory, tries_per_key=2):
        attempts = 0
        max_attempts = len(self.rot.keys) if self.rot.keys else 1
        while attempts < max_attempts:
            try:
                req = request_factory(self.service)
                return with_retry(lambda: req.execute(), tries=tries_per_key, backoff=1.4)
            except HttpError as e:
                if is_youtube_quota_error(e) and len(self.rot.keys) > 1:
                    self._rotate_and_rebuild()
                    attempts += 1
                    continue
                raise

def is_gemini_quota_error(exc: Exception) -> bool:
    msg = (str(exc) or "").lower()
    return ("429" in msg) or ("too many requests" in msg) or ("rate limit" in msg) or ("resource exhausted" in msg) or ("quota" in msg)

def call_gemini_rotating(
    model_name: str,
    keys,
    system_instruction: str,
    user_payload: str,
    timeout_s: int = GEMINI_TIMEOUT,
    max_tokens: int = GEMINI_MAX_TOKENS,
    on_rotate=None
) -> str:
    rot = RotatingKeys(keys, state_key="gem_key_idx", on_rotate=lambda i, k: on_rotate and on_rotate(i, k))
    if not rot.current():
        raise RuntimeError("Gemini API Key가 비어 있습니다.")
    attempts = 0
    max_attempts = len(rot.keys) if rot.keys else 1
    while attempts < max_attempts:
        try:
            genai.configure(api_key=rot.current())
            model = genai.GenerativeModel(
                model_name,
                generation_config={"temperature": 0.2, "max_output_tokens": max_tokens, "top_p": 0.9}
            )
            resp = model.generate_content([system_instruction, user_payload],
                                          request_options={"timeout": timeout_s})
            out = getattr(resp, "text", None)
            if not out and hasattr(resp, "candidates") and resp.candidates:
                c0 = resp.candidates[0]
                if hasattr(c0, "content") and getattr(c0.content, "parts", None):
                    p0 = c0.content.parts[0]
                    if hasattr(p0, "text"):
                        out = p0.text
            return out or ""
        except Exception as e:
            if is_gemini_quota_error(e) and len(rot.keys) > 1:
                rot.rotate(); attempts += 1; continue
            raise

# ===================== 유틸: ID/URL =====================
def extract_video_id_one(s: str):
    s = (s or "").strip()
    if not s: return None
    if _YT_ID_RE.match(s): return s
    try:
        u = urlparse(s)
    except Exception:
        return None
    q = parse_qs(u.query or "")
    if u.path == "/watch" and "v" in q:
        v = q.get("v", [""])[0]
        return v if _YT_ID_RE.match(v) else None
    if u.netloc.endswith("youtu.be"):
        v = u.path.strip("/")
        return v if _YT_ID_RE.match(v) else None
    if "youtube.com" in u.netloc and "/embed/" in u.path:
        v = u.path.split("/embed/")[-1].split("/")[0]
        return v if _YT_ID_RE.match(v) else None
    if "youtube.com" in u.netloc and u.path.startswith("/shorts/"):
        v = u.path.split("/shorts/")[-1].split("/")[0]
        return v if _YT_ID_RE.match(v) else None
    return None

def extract_video_ids_from_text(text: str):
    ids = []
    for line in (text or "").splitlines():
        vid = extract_video_id_one(line)
        if vid and vid not in ids:
            ids.append(vid)
    return ids

# ===================== 직렬화(LLM) from CSV(청크) =====================
def serialize_comments_for_llm_from_file(csv_path: str, max_rows=1500, max_chars_per_comment=280, max_total_chars=450_000):
    if not csv_path or not os.path.exists(csv_path):
        return "", 0, 0
    lines, total = [], 0
    remaining = max_rows
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        if "likeCount" in chunk.columns:
            chunk = chunk.sort_values("likeCount", ascending=False)
        for _, r in chunk.iterrows():
            if remaining <= 0 or total >= max_total_chars:
                break
            is_reply = "R" if int(r.get("isReply", 0) or 0) == 1 else "T"
            author = str(r.get("author", "") or "").replace("\n", " ")
            likec = int(r.get("likeCount", 0) or 0)
            text = str(r.get("text", "") or "").replace("\n", " ")
            if len(text) > max_chars_per_comment:
                text = text[:max_chars_per_comment] + "…"
            line = f"[{is_reply}|♥{likec}] {author}: {text}"
            if total + len(line) + 1 > max_total_chars:
                break
            lines.append(line)
            total += len(line) + 1
            remaining -= 1
        if remaining <= 0 or total >= max_total_chars:
            break
    return "\n".join(lines), len(lines), total

# ===================== YouTube API 함수 =====================
def yt_search_videos(rt, keyword, max_results, order="relevance", published_after=None, published_before=None, log=None):
    video_ids, token = [], None
    while len(video_ids) < max_results:
        params = dict(q=keyword, part="id", type="video", order=order, maxResults=min(50, max_results - len(video_ids)))
        if published_after: params["publishedAfter"] = published_after
        if published_before: params["publishedBefore"] = published_before
        if token: params["pageToken"] = token
        resp = rt.execute(lambda s: s.search().list(**params))
        for it in resp.get("items", []):
            vid = it["id"]["videoId"]
            if vid not in video_ids: video_ids.append(vid)
        token = resp.get("nextPageToken")
        if not token: break
        if log: log(f"검색 진행: {len(video_ids)}개")
        time.sleep(0.35)
    return video_ids

def yt_video_statistics(rt, video_ids, log=None):
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        resp = rt.execute(lambda s: s.videos().list(part="statistics,snippet,contentDetails", id=",".join(batch)))
        for item in resp.get("items", []):
            stats = item.get("statistics", {})
            snip = item.get("snippet", {})
            cont = item.get("contentDetails", {})
            dur_iso = cont.get("duration", "")
            def _dsec(dur: str):
                if not dur or not dur.startswith("P"): return None
                h = re.search(r"(\d+)H", dur); m = re.search(r"(\d+)M", dur); s = re.search(r"(\d+)S", dur)
                return (int(h.group(1)) if h else 0) * 3600 + (int(m.group(1)) if m else 0) * 60 + (int(s.group(1)) if s else 0)
            dur_sec = _dsec(dur_iso)
            short_type = "Shorts" if (dur_sec is not None and dur_sec <= 60) else "Clip"
            vid_id = item.get("id")
            rows.append({
                "video_id": vid_id,
                "video_url": f"https://www.youtube.com/watch?v={vid_id}",
                "title": snip.get("title", ""),
                "channelTitle": snip.get("channelTitle", ""),
                "publishedAt": snip.get("publishedAt", ""),
                "duration": dur_iso,
                "shortType": short_type,
                "viewCount": int(stats.get("viewCount", 0) or 0),
                "likeCount": int(stats.get("likeCount", 0) or 0),
                "commentCount": int(stats.get("commentCount", 0) or 0),
            })
        if log: log(f"통계 배치 {i // 50 + 1} 완료")
        time.sleep(0.35)
    return rows

def yt_all_replies(rt, parent_id, video_id, title="", short_type="Clip", log=None, cap=None):
    replies, token = [], None
    while True:
        if cap is not None and len(replies) >= cap:
            return replies[:cap]
        params = dict(part="snippet", parentId=parent_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.comments().list(**params))
        except HttpError as e:
            if log: log(f"[오류] replies {video_id}/{parent_id}: {e}")
            break
        for c in resp.get("items", []):
            sn = c["snippet"]
            replies.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": c.get("id", ""), "parent_id": parent_id, "isReply": 1,
                "author": sn.get("authorDisplayName", ""),
                "text": sn.get("textDisplay", "") or "",
                "publishedAt": sn.get("publishedAt", ""),
                "likeCount": int(sn.get("likeCount", 0) or 0),
            })
            if cap is not None and len(replies) >= cap:
                return replies[:cap]
        token = resp.get("nextPageToken")
        if not token: break
        time.sleep(0.25)
    return replies

def yt_all_comments_sync(rt, video_id, title="", short_type="Clip", include_replies=True, log=None,
                    max_per_video: int | None = None):
    rows, token = [], None
    while True:
        if max_per_video is not None and len(rows) >= max_per_video:
            return rows[:max_per_video]
        params = dict(part="snippet,replies", videoId=video_id, maxResults=100, pageToken=token, textFormat="plainText")
        try:
            resp = rt.execute(lambda s: s.commentThreads().list(**params))
        except HttpError as e:
            if log: log(f"[오류] commentThreads {video_id}: {e}")
            break

        for it in resp.get("items", []):
            top = it["snippet"]["topLevelComment"]["snippet"]
            thread_id = it["snippet"]["topLevelComment"]["id"]
            total_replies = int(it["snippet"].get("totalReplyCount", 0) or 0)
            rows.append({
                "video_id": video_id, "video_title": title, "shortType": short_type,
                "comment_id": thread_id, "parent_id": "", "isReply": 0,
                "author": top.get("authorDisplayName", ""),
                "text": top.get("textDisplay", "") or "",
                "publishedAt": top.get("publishedAt", ""),
                "likeCount": int(top.get("likeCount", 0) or 0),
            })
            if include_replies and total_replies > 0:
                cap = None
                if max_per_video is not None:
                    cap = max(0, max_per_video - len(rows))
                if cap == 0:
                    return rows[:max_per_video]
                rows.extend(yt_all_replies(rt, thread_id, video_id, title, short_type, log, cap=cap))
                if max_per_video is not None and len(rows) >= max_per_video:
                    return rows[:max_per_video]

        token = resp.get("nextPageToken")
        if not token: break
        if log: log(f"  댓글 페이지 진행, 누계 {len(rows)}")
        time.sleep(0.25)
    return rows

# ===================== 댓글 수집: 스트리밍 저장 =====================
def parallel_collect_comments_streaming(
    video_list, rt_keys, include_replies, max_total_comments, max_per_video,
    log_callback=None, prog_callback=None
):
    out_csv = os.path.join(BASE_DIR, f"collect_{uuid4().hex}.csv")
    wrote_header = False
    total_written = 0
    total_videos = len(video_list)
    done_videos = 0
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(
                yt_all_comments_sync,
                RotatingYouTube(rt_keys),
                vid_info["video_id"],
                vid_info.get("title", ""),
                vid_info.get("shortType", "Clip"),
                include_replies,
                None,
                max_per_video
            ): vid_info for vid_info in video_list
        }
        for fut in as_completed(futures):
            vid_info = futures[fut]
            try:
                comments = fut.result()
                if comments:
                    df_chunk = pd.DataFrame(comments)
                    df_chunk = clean_df_strings(df_chunk)
                    df_chunk.to_csv(
                        out_csv, index=False,
                        mode=("a" if wrote_header else "w"),
                        header=(not wrote_header),
                        encoding="utf-8-sig"
                    )
                    wrote_header = True
                    total_written += len(df_chunk)
                done_videos += 1
                if log_callback: log_callback(f"✅ [{done_videos}/{total_videos}] {vid_info.get('title','')} - {len(comments):,}개 수집")
                if prog_callback: prog_callback(done_videos / total_videos)
            except Exception as e:
                done_videos += 1
                if log_callback: log_callback(f"❌ [{done_videos}/{total_videos}] {vid_info.get('title','')} - 실패: {e}")
                if prog_callback: prog_callback(done_videos / total_videos)
            if total_written >= max_total_comments:
                if log_callback: log_callback(f"최대 수집 한도({max_total_comments:,}개) 도달, 중단")
                break
    return out_csv, total_written

# ===================== 세션 상태 & 로드 =====================
def ensure_state():
    defaults = dict(
        focus_step=1,
        last_keyword="",
        # 심플
        s_query="", s_comments_path="", s_df_stats=None,
        s_serialized_sample="", s_result_text="",
        s_history=[], s_preset="최근 1년", s_total_count=0,
        # 고급
        mode="검색 모드",
        df_stats=None, selected_ids=[],
        adv_comments_path="", adv_serialized_sample="", adv_result_text="",
        adv_followups=[], adv_history=[], adv_total_count=0,
        # 입력값
        simple_follow_q="", adv_follow_q="",
        # 탭 표시값
        last_tab="심플",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

ensure_state()

def _apply_loaded_session(sess_name: str):
    base = os.path.join(SESS_DIR, sess_name)
    qa_file = os.path.join(base, "qa.json")

    # 다운로드
    github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/qa.json", GITHUB_TOKEN, qa_file)
    for fn in ["simple_comments_full.csv","simple_comments_full.csv.gz","simple_videos.csv",
               "adv_comments_full.csv","adv_comments_full.csv.gz","adv_videos.csv"]:
        try:
            github_download_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/{fn}", GITHUB_TOKEN, os.path.join(base, fn))
        except:
            pass

    # 세션값 주입
    if os.path.exists(qa_file):
        with open(qa_file, encoding="utf-8") as f:
            qa = json.load(f)
        st.session_state["s_history"] = qa.get("simple_history", [])
        st.session_state["adv_history"] = qa.get("adv_history", [])
        st.session_state["s_query"] = qa.get("simple_query","")
        st.session_state["last_keyword"] = qa.get("last_keyword","")
        st.session_state["s_preset"] = qa.get("preset","최근 1년")
        st.session_state["last_tab"] = qa.get("last_tab", "심플")
        if st.session_state["s_history"]:
            st.session_state["s_result_text"] = st.session_state["s_history"][-1][1]
        if st.session_state["adv_history"]:
            st.session_state["adv_result_text"] = st.session_state["adv_history"][-1][1]

    def _first_existing(*names):
        for n in names:
            p = os.path.join(base, n)
            if os.path.exists(p): return p
        return ""

    st.session_state["s_comments_path"] = _first_existing("simple_comments_full.csv.gz","simple_comments_full.csv")
    s_videos_path = os.path.join(base,"simple_videos.csv")
    st.session_state["s_df_stats"] = pd.read_csv(s_videos_path) if os.path.exists(s_videos_path) else None

    st.session_state["adv_comments_path"] = _first_existing("adv_comments_full.csv.gz","adv_comments_full.csv")
    a_videos_path = os.path.join(base,"adv_videos.csv")
    st.session_state["df_stats"] = pd.read_csv(a_videos_path) if os.path.exists(a_videos_path) else None

# --- 탭 내비 유틸 ---
def consume_next_tab(default_val: str = "심플") -> str:
    if "__next_tab" in st.session_state:
        val = st.session_state["__next_tab"]
        del st.session_state["__next_tab"]
        return val
    return st.session_state.get("last_tab", default_val)

# --- (라디오 생성 전) pending 로드 처리 ---
if "__pending_session_load" in st.session_state:
    target = st.session_state["__pending_session_load"]
    del st.session_state["__pending_session_load"]
    _apply_loaded_session(target)
    st.session_state["__next_tab"] = st.session_state.get("last_tab", "심플")
    st.success("세션을 불러왔습니다.")
    safe_rerun()

# 상단 탭(라디오)
default_tab = consume_next_tab(default_val=st.session_state.get("last_tab","심플"))
tab = st.radio(
    "화면",
    ["심플", "고급", "세션"],
    index=["심플","고급","세션"].index(default_tab),
    horizontal=True,
    key="tab"
)
if st.session_state.get("last_tab") != tab:
    st.session_state["last_tab"] = tab

# ===================== 히스토리 → 컨텍스트 =====================
def build_history_context(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return ""
    lines = []
    for i, (q, a) in enumerate(pairs, 1):
        lines.append(f"[이전 Q{i}]: {q}")
        lines.append(f"[이전 A{i}]: {a}")
    return "\n".join(lines)

# ===================== 키워드 Counter (CSV 청크) =====================
@st.cache_data(ttl=600, show_spinner=False)
def compute_keyword_counter_from_file(csv_path: str, stopset_list: list[str], per_comment_cap: int = 200) -> list[tuple[str,int]]:
    if not csv_path or not os.path.exists(csv_path):
        return []
    stopset = set(stopset_list)
    counter = Counter()
    for chunk in pd.read_csv(csv_path, usecols=["text"], chunksize=100_000):
        texts = (chunk["text"].astype(str).str.slice(0, per_comment_cap)).tolist()
        if not texts:
            continue
        tokens = kiwi.tokenize(" ".join(texts), normalize_coda=True)
        words = [t.form for t in tokens if t.tag in ("NNG","NNP") and len(t.form) > 1 and t.form not in stopset]
        counter.update(words)
    return counter.most_common(300)

def keyword_bubble_figure_from_counter(counter_items: list[tuple[str,int]]) -> go.Figure | None:
    if not counter_items:
        return None
    df_kw = pd.DataFrame(counter_items[:30], columns=["word", "count"])
    df_kw["label"] = df_kw["word"] + "<br>" + df_kw["count"].astype(str)
    df_kw["scaled"] = np.sqrt(df_kw["count"])
    circles = circlify.circlify(
        [{"id": w, "datum": s} for w, s in zip(df_kw["word"], df_kw["scaled"])],
        show_enclosure=False,
        target_enclosure=circlify.Circle(x=0, y=0, r=1)
    )
    pos = {c.ex["id"]: (c.x, c.y, c.r) for c in circles if "id" in c.ex}
    df_kw["x"] = df_kw["word"].map(lambda w: pos[w][0])
    df_kw["y"] = df_kw["word"].map(lambda w: pos[w][1])
    df_kw["r"] = df_kw["word"].map(lambda w: pos[w][2])
    s_min, s_max = df_kw["scaled"].min(), df_kw["scaled"].max()
    df_kw["font_size"] = df_kw["scaled"].apply(lambda s: int(10 + (s - s_min) / max(s_max - s_min, 1) * 12))
    fig_kw = go.Figure()
    palette = px.colors.sequential.Blues
    df_kw["color_idx"] = df_kw["scaled"].apply(lambda s: int((s - s_min) / max(s_max - s_min, 1) * (len(palette) - 1)))
    for _, row in df_kw.iterrows():
        color = palette[int(row["color_idx"])]
        fig_kw.add_shape(type="circle", xref="x", yref="y",
                         x0=row["x"] - row["r"], y0=row["y"] - row["r"],
                         x1=row["x"] + row["r"], y1=row["y"] + row["r"],
                         line=dict(width=0), fillcolor=color, opacity=0.88, layer="below")
    fig_kw.add_trace(go.Scatter(
        x=df_kw["x"], y=df_kw["y"], mode="text",
        text=df_kw["label"], textposition="middle center",
        textfont=dict(color="white", size=df_kw["font_size"].tolist()),
        hovertext=df_kw["word"] + " (" + df_kw["count"].astype(str) + ")",
        hovertemplate="%{hovertext}<extra></extra>",
    ))
    fig_kw.update_xaxes(visible=False, range=[-1.05, 1.05])
    fig_kw.update_yaxes(visible=False, range=[-1.05, 1.05], scaleanchor="x", scaleratio=1)
    fig_kw.update_layout(title="Top30 키워드 버블", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0, r=0, t=40, b=0))
    return fig_kw

# ===================== 정량 시각화: CSV 청크 집계 =====================
def timeseries_from_file(csv_path: str):
    if not csv_path or not os.path.exists(csv_path): return None, None
    tmin = None; tmax = None
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True)
        if dt.notna().any():
            lo, hi = dt.min(), dt.max()
            tmin = lo if (tmin is None or (lo < tmin)) else tmin
            tmax = hi if (tmax is None or (hi > tmax)) else tmax
    if tmin is None or tmax is None:
        return None, None
    span_hours = (tmax - tmin).total_seconds()/3600.0
    use_hour = (span_hours <= 48)

    agg = {}
    for chunk in pd.read_csv(csv_path, usecols=["publishedAt"], chunksize=200_000):
        dt = pd.to_datetime(chunk["publishedAt"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
        dt = dt.dropna()
        if dt.empty: continue
        bucket = (dt.dt.floor("H") if use_hour else dt.dt.floor("D"))
        vc = bucket.value_counts()
        for t, c in vc.items():
            agg[t] = agg.get(t, 0) + int(c)
    # pandas 구버전 호환 (names= 미지원)
    ts = pd.Series(agg).sort_index().rename("count").reset_index().rename(columns={"index":"bucket"})
    return ts, ("시간별" if use_hour else "일자별")

def top_authors_from_file(csv_path: str, topn=10):
    if not csv_path or not os.path.exists(csv_path): return None
    counts = {}
    for chunk in pd.read_csv(csv_path, usecols=["author"], chunksize=200_000):
        vc = chunk["author"].astype(str).value_counts()
        for k, v in vc.items():
            counts[k] = counts.get(k, 0) + int(v)
    if not counts: return None
    s = pd.Series(counts).sort_values(ascending=False).head(topn)
    return s.reset_index().rename(columns={"index": "author", 0: "count"}).rename(columns={"count": "count"})

def render_quant_viz_from_paths(comments_csv_path: str, df_stats: pd.DataFrame, scope_label="(KST 기준)", wrap_in_expander: bool = True):
    if not comments_csv_path or not os.path.exists(comments_csv_path): return
    wrapper = (st.expander("📊 정량 요약", expanded=True) if wrap_in_expander else st.container(border=True))
    with wrapper:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.subheader("① 키워드 버블")
                try:
                    custom_stopwords = {
                        "아","휴","아이구","아이쿠","아이고","어","나","우리","저희","따라","의해","을","를",
                        "에","의","가","으로","로","에게","뿐이다","의거하여","근거하여","입각하여","기준으로",
                        "그냥","댓글","영상","오늘","이제","뭐","진짜","정말","부분","요즘","제발","완전",
                        "그게","일단","모든","위해","대한","있지","이유","계속","실제","유튜브","이번","가장","드라마",
                    }
                    stopset = set(korean_stopwords); stopset.update(custom_stopwords)
                    query_kw = (st.session_state.get("s_query") or st.session_state.get("last_keyword") or st.session_state.get("adv_analysis_keyword") or "").strip()
                    if query_kw:
                        tokens_q = kiwi.tokenize(query_kw, normalize_coda=True)
                        query_words = [t.form for t in tokens_q if t.tag in ("NNG","NNP") and len(t.form) > 1]
                        stopset.update(query_words)
                    with st.spinner("키워드 계산 중…"):
                        items = compute_keyword_counter_from_file(comments_csv_path, list(stopset), per_comment_cap=200)
                    fig = keyword_bubble_figure_from_counter(items)
                    if fig is None:
                        st.info("표시할 키워드가 없습니다(불용어 제거 후 남은 단어 없음).")
                    else:
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"키워드 분석 불가: {e}")

        with col2:
            with st.container(border=True):
                st.subheader("② 시점별 댓글량 변동 추이")
                ts, label = timeseries_from_file(comments_csv_path)
                if ts is not None:
                    fig_ts = px.line(ts, x="bucket", y="count", markers=True, title=f"{label} 댓글량 추이 {scope_label}")
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("댓글 타임스탬프가 비어 있습니다.")

        if df_stats is not None and not df_stats.empty:
            col3, col4 = st.columns(2)
            with col3:
                with st.container(border=True):
                    st.subheader("③ Top10 영상 댓글수")
                    top_vids = df_stats.sort_values(by="commentCount", ascending=False).head(10).copy()
                    top_vids["title_short"] = top_vids["title"].apply(lambda t: t[:20] + "…" if isinstance(t, str) and len(t) > 20 else t)
                    fig_vids = px.bar(top_vids, x="commentCount", y="title_short",
                                      orientation="h", text="commentCount", title="Top10 영상 댓글수")
                    st.plotly_chart(fig_vids, use_container_width=True)
            with col4:
                with st.container(border=True):
                    st.subheader("④ 댓글 작성자 활동량 Top10")
                    ta = top_authors_from_file(comments_csv_path, topn=10)
                    if ta is not None and not ta.empty:
                        fig_auth = px.bar(ta, x="count", y="author", orientation="h", text="count", title="Top10 댓글 작성자 활동량")
                        st.plotly_chart(fig_auth, use_container_width=True)
                    else:
                        st.info("작성자 데이터 없음")

        with st.container(border=True):
            st.subheader("⑤ 댓글 좋아요 Top10")
            best = []
            for chunk in pd.read_csv(comments_csv_path, usecols=["video_id","video_title","author","text","likeCount"], chunksize=200_000):
                chunk["likeCount"] = pd.to_numeric(chunk["likeCount"], errors="coerce").fillna(0).astype(int)
                best.append(chunk.sort_values("likeCount", ascending=False).head(10))
            if best:
                df_top = pd.concat(best).sort_values("likeCount", ascending=False).head(10)
                for _, row in df_top.iterrows():
                    url = f"https://www.youtube.com/watch?v={row['video_id']}"
                    st.markdown(
                        f"<div style='margin-bottom:15px;'>"
                        f"<b>{int(row['likeCount'])} 👍</b> — {row.get('author','')}<br>"
                        f"<span style='font-size:14px;'>▶️ <a href='{url}' target='_blank' style='color:black; text-decoration:none;'>"
                        f"{str(row.get('video_title','(제목없음)'))[:60]}</a></span><br>"
                        f"> {str(row.get('text',''))[:150]}{'…' if len(str(row.get('text','')))>150 else ''}"
                        f"</div>", unsafe_allow_html=True
                    )

# ===================== 저장/아카이브 =====================
def _slugify_filename(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^\w\-]+", "", s)
    if not s: s = "no_kw"
    return s[:60]

def _build_session_name() -> str:
    kw = (st.session_state.get("s_query") or st.session_state.get("last_keyword") or "").strip() or "no_kw"
    preset = (st.session_state.get("s_preset") or "최근 1년").replace(" ", "")
    now_kst = datetime.now(_kst_tz()).strftime("%Y-%m-%d-%H:%M")
    return f"{_slugify_filename(kw)}_{now_kst}_{preset}"

def save_current_session():
    sess_name = _build_session_name()
    outdir = os.path.join(SESS_DIR, sess_name)
    os.makedirs(outdir, exist_ok=True)

    qa_data = {
        "simple_history": st.session_state.get("s_history", []),
        "adv_history": st.session_state.get("adv_history", []),
        "simple_query": st.session_state.get("s_query",""),
        "last_keyword": st.session_state.get("last_keyword",""),
        "preset": st.session_state.get("s_preset",""),
        "last_tab": st.session_state.get("last_tab","심플"),
        "saved_at_kst": datetime.now(_kst_tz()).strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(outdir, "qa.json"), "w", encoding="utf-8") as f:
        json.dump(qa_data, f, ensure_ascii=False, indent=2)

    def _copy_if_exists(src_path, dst_name):
        if src_path and os.path.exists(src_path):
            dst_path = os.path.join(outdir, dst_name)
            with open(src_path, "rb") as rf, open(dst_path, "wb") as wf:
                wf.write(rf.read())
            return dst_path
        return None

    # 심플
    _copy_if_exists(st.session_state.get("s_comments_path",""), "simple_comments_full.csv")
    s_df_stats = st.session_state.get("s_df_stats")
    if s_df_stats is not None and not s_df_stats.empty:
        s_df_stats.to_csv(os.path.join(outdir,"simple_videos.csv"), index=False, encoding="utf-8-sig")

    # 고급
    _copy_if_exists(st.session_state.get("adv_comments_path",""), "adv_comments_full.csv")
    df_stats = st.session_state.get("df_stats")
    if df_stats is not None and not df_stats.empty:
        df_stats.to_csv(os.path.join(outdir,"adv_videos.csv"), index=False, encoding="utf-8-sig")

    uploaded = []
    if GITHUB_TOKEN and GITHUB_REPO:
        info = github_upload_file(GITHUB_REPO, GITHUB_BRANCH, f"sessions/{sess_name}/qa.json", os.path.join(outdir,"qa.json"), GITHUB_TOKEN)
        uploaded.append(info)
        for fn in sorted(os.listdir(outdir)):
            if fn == "qa.json": continue
            p = os.path.join(outdir, fn)
            if os.path.isfile(p):
                path_in_repo = f"sessions/{sess_name}/{fn}"
                info = github_upload_file(GITHUB_REPO, GITHUB_BRANCH, path_in_repo, p, GITHUB_TOKEN)
                uploaded.append(info)
    return sess_name, uploaded

def list_sessions_github():
    if not (GITHUB_TOKEN and GITHUB_REPO):
        return []
    items = github_list_dir(GITHUB_REPO, GITHUB_BRANCH, "sessions", GITHUB_TOKEN)
    return [x["name"] for x in items if isinstance(x, dict) and x.get("type") == "dir"]

# ===================== 다운로드 UI =====================
def render_downloads_from_paths(comments_csv_path: str, df_stats: pd.DataFrame, prefix="simple"):
    if comments_csv_path and os.path.exists(comments_csv_path):
        st.markdown("---")
        st.subheader("⬇️ 결과 다운로드")
        with open(comments_csv_path, "rb") as f:
            st.download_button(
                "전체 댓글 (CSV)", data=f.read(),
                file_name=f"{prefix}_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv", key=f"{prefix}_dl_full_csv"
            )
    if df_stats is not None and not df_stats.empty:
        df_videos = df_stats.copy()
        if "viewCount" in df_videos.columns:
            df_videos = df_videos.sort_values(by="viewCount", ascending=False).reset_index(drop=True)
        csv_videos = df_videos.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            "전체 영상 (CSV)", data=csv_videos,
            file_name=f"{prefix}_videolist_{len(df_videos)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv", key=f"{prefix}_dl_videos_csv"
        )

# ===================== 추가질문 핸들러 =====================
def handle_followup_simple():
    follow_q = (st.session_state.get("simple_follow_q") or "").strip()
    if not follow_q: return
    if not GEMINI_API_KEYS:
        st.error("Gemini API Key가 없습니다."); return
    if not st.session_state.get("s_comments_path"):
        st.error("댓글 파일이 없습니다. 먼저 수집/분석 실행."); return
    append_log("심플-추가", st.session_state.get("s_query",""), follow_q)
    context_str = build_history_context(st.session_state.get("s_history", []))
    system_instruction = (
        "너는 유튜브 댓글을 분석하는 어시스턴트다. "
        "아래는 이미 추출/가공된 댓글 샘플과 이전 질의응답 히스토리다. "
        "이전 맥락을 모두 고려하여 한국어로 간결하고 구조화된 답을 하라. "
        "반드시 모든 댓글을 읽고 답변하라."
    )
    s_text, _, _ = serialize_comments_for_llm_from_file(st.session_state["s_comments_path"])
    payload = ((context_str + "\n\n") if context_str else "") + (
        f"[현재 질문]: {follow_q}\n"
        f"[기간]: {st.session_state.get('s_preset','최근 1년')}\n\n"
        f"[댓글 샘플]:\n{s_text}\n"
    )
    out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, payload)
    st.session_state["s_history"].append((follow_q, out))
    st.session_state["s_result_text"] = out
    st.session_state["simple_follow_q"] = ""
    st.success("추가 분석 완료")

def handle_followup_advanced():
    adv_follow_q = (st.session_state.get("adv_follow_q") or "").strip()
    if not adv_follow_q: return
    if not GEMINI_API_KEYS:
        st.error("Gemini API Key가 없습니다."); return
    if not st.session_state.get("adv_comments_path"):
        st.error("댓글 파일이 없습니다. 먼저 수집/분석 실행."); return
    append_log("고급-추가", st.session_state.get("last_keyword",""), adv_follow_q)
    a_text, _, _ = serialize_comments_for_llm_from_file(st.session_state["adv_comments_path"])
    context_str = build_history_context(st.session_state.get("adv_history", []))
    system_instruction = (
        "너는 유튜브 댓글을 분석하는 어시스턴트다. "
        "아래는 이미 추출/가공된 댓글 샘플과 이전 질의응답 히스토리다. "
        "이전 맥락을 모두 고려하여 한국어로 간결하고 구조화된 답을 하라. "
        "반드시 모든 댓글을 읽고 답변하라."
    )
    payload = ((context_str + "\n\n") if context_str else "") + (
        f"[현재 질문]: {adv_follow_q}\n\n"
        f"[댓글 샘플]:\n{a_text}\n"
    )
    out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, payload)
    st.session_state["adv_followups"].append((adv_follow_q, out))
    st.session_state["adv_history"].append((adv_follow_q, out))
    st.session_state["adv_result_text"] = out
    st.session_state["adv_follow_q"] = ""
    st.success("추가 분석 완료(고급)")

# ===================== 1) 심플 =====================
if tab == "심플":
    st.subheader("최근 기간 댓글 반응 — 드라마/배우명으로 바로 분석")
    s_query = st.text_input("드라마 or 배우명", value=st.session_state.get("s_query", ""),
                            placeholder="키워드 입력", key="simple_query")
    preset_simple = st.radio(
        "업로드 기간 (KST)",
        ["최근 12시간","최근 24시간","최근 48시간","최근 1주일","최근 1개월","최근 6개월",
         "최근 1년","최근 2년","최근 3년","최근 4년","최근 5년","최근 10년"],
        horizontal=True, key="simple_preset"
    )
    user_question = st.text_area("추가 질문/요청(선택, 비우면 기본 질문)", height=80,
                                 placeholder="예: 연기력/호불호 중심으로 분석해줘", key="simple_question")

    SIMPLE_TOP_N = 50
    SIMPLE_ORDER = "viewCount"
    now_kst = datetime.now(_kst_tz())

    if preset_simple == "최근 12시간":      start_dt = now_kst - timedelta(hours=12)
    elif preset_simple == "최근 24시간":     start_dt = now_kst - timedelta(hours=24)
    elif preset_simple == "최근 48시간":     start_dt = now_kst - timedelta(hours=48)
    elif preset_simple == "최근 1주일":     start_dt = now_kst - timedelta(days=7)
    elif preset_simple == "최근 1개월":     start_dt = now_kst - timedelta(days=30)
    elif preset_simple == "최근 6개월":     start_dt = now_kst - timedelta(days=182)
    elif preset_simple == "최근 1년":       start_dt = now_kst - timedelta(days=365)
    elif preset_simple == "최근 2년":       start_dt = now_kst - timedelta(days=365*2)
    elif preset_simple == "최근 3년":       start_dt = now_kst - timedelta(days=365*3)
    elif preset_simple == "최근 4년":       start_dt = now_kst - timedelta(days=365*4)
    elif preset_simple == "최근 5년":       start_dt = now_kst - timedelta(days=365*5)
    elif preset_simple == "최근 10년":      start_dt = now_kst - timedelta(days=365*10)
    else:                                   start_dt = now_kst - timedelta(days=365)

    published_after = kst_to_rfc3339_utc(start_dt)
    published_before = kst_to_rfc3339_utc(now_kst)

    if st.button("🚀 분석하기", type="primary", key="simple_run"):
        if not YT_API_KEYS:
            st.error("YouTube API Key가 없습니다.")
        elif not GEMINI_API_KEYS:
            st.error("Gemini API Key가 없습니다.")
        elif not st.session_state["simple_query"].strip():
            st.warning("드라마 or 배우명을 입력하세요.")
        else:
            if not lock_guard_start_or_warn():
                st.stop()
            try:
                st.session_state["s_query"] = st.session_state["simple_query"].strip()
                st.session_state["s_preset"] = preset_simple
                st.session_state["s_history"] = []
                append_log("심플", st.session_state["s_query"], st.session_state.get("simple_question", ""))

                status_ph = st.empty()
                with status_ph.status("심플 모드 실행 중…", expanded=True) as status:
                    rt = RotatingYouTube(YT_API_KEYS, log=lambda m: status.write(m))
                    status.write(f"🔍 영상 검색 중… ({preset_simple}, 정렬: {SIMPLE_ORDER})")
                    ids = yt_search_videos(rt, st.session_state["s_query"], SIMPLE_TOP_N,
                                           SIMPLE_ORDER, published_after, published_before, log=status.write)

                    status.write(f"🎞️ 대상 영상: {len(ids)} — 메타 조회…")
                    stats = yt_video_statistics(rt, ids, log=status.write)
                    df_stats = pd.DataFrame(stats)
                    st.session_state["s_df_stats"] = df_stats

                    status.write("💬 댓글 수집 중…")
                    video_list = df_stats.to_dict('records')
                    prog = st.progress(0, text="수집 진행 중")
                    log_ph = st.empty()
                    csv_path, total_cnt = parallel_collect_comments_streaming(
                        video_list=video_list,
                        rt_keys=YT_API_KEYS,
                        include_replies=False,
                        max_total_comments=MAX_TOTAL_COMMENTS,
                        max_per_video=MAX_COMMENTS_PER_VIDEO,
                        log_callback=log_ph.write,
                        prog_callback=prog.progress
                    )
                    st.session_state["s_comments_path"] = csv_path
                    st.session_state["s_total_count"] = int(total_cnt)

                    if total_cnt == 0:
                        status.update(label="⚠️ 댓글을 수집하지 못했습니다.", state="error")
                        st.session_state["s_result_text"] = ""
                    else:
                        status.write("🧠 AI 분석 중…")
                        system_instruction = (
                            "너는 유튜브 댓글을 분석하는 어시스턴트다. "
                            "아래 키워드와 지정된 기간 내 댓글 샘플을 바탕으로, 전반적 반응을 한국어로 간결하게 요약하라. "
                            "핵심 포인트를 항목화하고, 긍/부정/중립의 대략적 비율과 대표 코멘트(10개미만)를 예시로 제시하라. "
                            "키워드가 인물명이면 인물 중심, 드라마명이면 드라마 중심으로 분석하라. "
                            "반드시 모든 댓글을 읽고 답변하라."
                        )
                        default_q = f"{preset_simple} 기준으로 '{st.session_state['s_query']}'에 대한 유튜브 댓글 반응을 요약해줘."
                        prompt_q = (st.session_state.get("simple_question", "").strip() or default_q)
                        s_text, _, _ = serialize_comments_for_llm_from_file(csv_path)
                        payload = (
                            f"[키워드]: {st.session_state['s_query']}\n"
                            f"[질문]: {prompt_q}\n"
                            f"[기간]: {preset_simple}\n\n"
                            f"[댓글 샘플]:\n{s_text}\n"
                        )
                        out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, payload,
                                                   timeout_s=GEMINI_TIMEOUT, max_tokens=GEMINI_MAX_TOKENS,
                                                   on_rotate=lambda i, k: status.write(f"🔁 Gemini 키 전환 → #{i+1}"))
                        st.session_state["s_result_text"] = out
                        st.session_state["s_history"].append((prompt_q, out))
                        status.update(label="🎉 분석 완료", state="complete")
                status_ph.empty()
            finally:
                release_lock()
                gc.collect()

    s_comments_path = st.session_state.get("s_comments_path","")
    s_df_stats = st.session_state.get("s_df_stats")

    if s_comments_path and os.path.exists(s_comments_path):
        st.success(f"수집 완료 — 전체 {st.session_state.get('s_total_count',0):,}개")

    if st.session_state.get("s_history"):
        with st.expander("🧠 AI 분석 결과 (최신)", expanded=True):
            last_q, last_a = st.session_state["s_history"][-1]
            st.markdown(f"**Q. {last_q}**")
            st.markdown(last_a)
        st.markdown("### 📝 전체 질문 히스토리")
        for i, (q, a) in enumerate(st.session_state["s_history"], start=1):
            with st.expander(f"{i}. {q}", expanded=False):
                st.markdown(a or "_응답 없음_")
        st.markdown("#### ➕ 추가 질문하기")
        st.text_input("추가 질문", placeholder="예: 주연배우들에 대한 반응은 어때?",
                      key="simple_follow_q", on_change=handle_followup_simple)
        st.button("질문 보내기", key="simple_follow_btn", on_click=handle_followup_simple)

    render_quant_viz_from_paths(s_comments_path, s_df_stats, scope_label="(KST 기준)", wrap_in_expander=True)
    render_downloads_from_paths(s_comments_path, s_df_stats, prefix="simple")

    if st.button("💾 세션 저장하기", key="simple_save_session"):
        name, _ = save_current_session()
        st.success(f"세션 저장 완료: {name}")

# ===================== 2) 고급 =====================
if tab == "고급":
    st.subheader("고급 모드")

    mode = st.radio("모드", ["검색 모드", "URL 직접 입력 모드"],
                    index=(0 if st.session_state.get("mode", "검색 모드") == "검색 모드" else 1),
                    horizontal=True, key="adv_mode_radio")
    if mode != st.session_state["mode"]:
        st.session_state["mode"] = mode
        st.session_state["focus_step"] = 1

    include_replies = st.checkbox("대댓글 포함", value=False, key="adv_include_replies")

    # ① 영상목록추출
    expanded1 = (st.session_state["focus_step"] == 1)
    with st.expander("① 영상목록추출", expanded=expanded1):
        published_after = published_before = None
        if st.session_state["mode"] == "검색 모드":
            st.markdown("**업로드 기간 (KST)**")
            preset = st.radio("프리셋", ["최근 12시간", "최근 30일", "최근 1년", "직접 입력"],
                              horizontal=True, key="adv_preset")
            now_kst = datetime.now(_kst_tz())
            if preset == "최근 12시간":
                start_dt = now_kst - timedelta(hours=12); end_dt = now_kst
            elif preset == "최근 30일":
                start_dt = now_kst - timedelta(days=30); end_dt = now_kst
            elif preset == "최근 1년":
                start_dt = now_kst - timedelta(days=365); end_dt = now_kst
            else:
                c1, c2 = st.columns(2)
                sd = c1.date_input("시작일", now_kst.date()-timedelta(days=30), key="adv_sd")
                stime = c1.time_input("시작 시:분", value=datetime.min.time().replace(hour=0, minute=0), key="adv_stime")
                ed = c2.date_input("종료일", now_kst.date(), key="adv_ed")
                etime = c2.time_input("종료 시:분", value=datetime.min.time().replace(hour=23, minute=59), key="adv_etime")
                start_dt = datetime.combine(sd, stime, tzinfo=_kst_tz())
                end_dt = datetime.combine(ed, etime, tzinfo=_kst_tz())
            published_after = kst_to_rfc3339_utc(start_dt)
            published_before = kst_to_rfc3339_utc(end_dt)

            c1, c2, c3 = st.columns([3, 1, 1])
            keyword = c1.text_input("검색 키워드", st.session_state.get("last_keyword", "") or "", key="adv_keyword")
            top_n = c2.number_input("TOP N", min_value=1, value=50, step=1, key="adv_topn")
            order = c3.selectbox("정렬", ["relevance", "viewCount"], key="adv_order")
        else:
            keyword = None; top_n = None; order = None
            urls_main = st.text_area("URL/ID 목록 (줄바꿈 구분)", height=160,
                                     placeholder="https://youtu.be/XXXXXXXXXXX\nXXXXXXXXXXX\n...", key="adv_urls")

        if st.button("목록 가져오기", use_container_width=True, key="adv_fetch_list"):
            if not YT_API_KEYS:
                st.error("YouTube API Key가 없습니다.")
            else:
                rt = RotatingYouTube(YT_API_KEYS, log=lambda m: st.write(m))
                log_box = st.empty()
                def log(msg): log_box.write(msg)
                if st.session_state["mode"] == "검색 모드":
                    st.session_state["last_keyword"] = keyword or ""
                    log("🔍 검색 실행 중…")
                    ids = yt_search_videos(rt, keyword, int(top_n), order, published_after, published_before, log)
                else:
                    ids = extract_video_ids_from_text(urls_main or "")
                    if not ids:
                        st.warning("URL/ID가 비어 있습니다."); st.stop()
                log(f"🎞️ 대상 영상 수: {len(ids)} — 메타/통계 조회…")
                stats = yt_video_statistics(rt, ids, log)
                df = pd.DataFrame(stats)
                if not df.empty and "publishedAt" in df.columns:
                    df["publishedAt"] = (
                        pd.to_datetime(df["publishedAt"], errors="coerce", utc=True)
                        .dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
                    )
                if "viewCount" in df.columns:
                    df = df.sort_values(by="viewCount", ascending=False).reset_index(drop=True)
                st.session_state["df_stats"] = df
                st.session_state["selected_ids"] = df["video_id"].tolist() if not df.empty else []
                st.session_state["focus_step"] = 2

    # ② 영상선택 및 URL추가
    expanded2 = (st.session_state["focus_step"] == 2)
    with st.expander("② 영상선택 및 URL추가", expanded=expanded2):
        df_stats = st.session_state["df_stats"]
        if df_stats is None or df_stats.empty:
            st.info("①에서 먼저 목록을 가져오세요.")
        else:
            st.dataframe(
                df_stats[["video_id","title","channelTitle","shortType","viewCount","commentCount","publishedAt"]],
                use_container_width=True
            )
            st.caption("선택: video_id 콤마(,)로 입력, 비우면 전체 선택")
            manual_select = st.text_input("선택 video_id 목록(옵션)", key="adv_manual_select")
            if st.button("선택 적용", key="adv_apply_select"):
                if manual_select.strip():
                    ids = [s.strip() for s in manual_select.split(",") if s.strip()]
                    valid = [i for i in ids if i in df_stats["video_id"].tolist()]
                    st.session_state["selected_ids"] = valid
                    st.success(f"적용됨: {len(valid)}개 선택")
                else:
                    st.session_state["selected_ids"] = df_stats["video_id"].tolist()
                    st.success(f"전체 {len(st.session_state['selected_ids'])}개 선택")

            if st.session_state["mode"] == "검색 모드":
                st.markdown("---")
                st.subheader("➕ 추가 URL/ID 병합")
                add_text = st.text_area("추가할 URL/ID (줄바꿈 구분)", height=100,
                                        placeholder="https://youtu.be/XXXXXXXXXXX\nXXXXXXXXXXX\n...", key="adv_add_text")
                if st.button("추가 병합 실행", key="adv_merge_btn"):
                    add_ids = extract_video_ids_from_text(add_text or "")
                    already = set(df_stats["video_id"].tolist())
                    dup = [v for v in add_ids if v in already]
                    add_ids = [v for v in add_ids if v not in already]
                    if dup: st.info(f"⚠️ 기존 목록과 중복 {len(dup)}개 제외")
                    if add_ids:
                        rt = RotatingYouTube(YT_API_KEYS, log=lambda m: st.write(m))
                        add_stats = yt_video_statistics(rt, add_ids)
                        add_df = pd.DataFrame(add_stats)
                        if not add_df.empty and "publishedAt" in add_df.columns:
                            add_df["publishedAt"] = (
                                pd.to_datetime(add_df["publishedAt"], errors="coerce", utc=True)
                                .dt.tz_convert("Asia/Seoul").dt.strftime("%Y-%m-%d %H:%M:%S")
                            )
                        st.session_state["df_stats"] = (
                            pd.concat([st.session_state["df_stats"], add_df], ignore_index=True)
                            .drop_duplicates(subset=["video_id"])
                            .sort_values(by="viewCount", ascending=False)
                            .reset_index(drop=True)
                        )
                        st.session_state["selected_ids"] = list(dict.fromkeys(st.session_state["selected_ids"] + add_ids))
                        st.success(f"추가 {len(add_ids)}개 병합 완료")
                    else:
                        st.info("추가할 신규 URL/ID가 없습니다.")
            st.markdown("---")
            if st.button("다음: 댓글추출로 이동", type="primary", key="adv_next_to_comments"):
                if not st.session_state["selected_ids"]:
                    st.warning("선택된 영상이 없습니다.")
                else:
                    st.session_state["focus_step"] = 3

    # ③ 댓글추출
    expanded3 = (st.session_state["focus_step"] == 3)
    with st.expander("③ 댓글추출", expanded=expanded3):
        if not st.session_state["selected_ids"]:
            st.info("②에서 먼저 영상을 선택하세요.")
        else:
            st.write(f"대상 영상 수: **{len(st.session_state['selected_ids'])}**")
            include_replies_local = st.checkbox("대댓글 포함(이 단계에서만 적용)",
                                                value=include_replies, key="adv_include_replies_collect")
            if st.button("댓글 수집 시작", type="primary", use_container_width=True, key="adv_collect_btn"):
                if not YT_API_KEYS:
                    st.error("YouTube API Key가 없습니다.")
                else:
                    if not lock_guard_start_or_warn():
                        st.stop()
                    try:
                        df_stats = st.session_state["df_stats"]
                        target_ids = st.session_state["selected_ids"]
                        if df_stats is not None and not df_stats.empty and "viewCount" in df_stats.columns:
                            video_list = df_stats[df_stats["video_id"].isin(target_ids)].sort_values("viewCount", ascending=False).to_dict('records')
                        else:
                            video_list = [{"video_id": vid, "title": "", "shortType": "Clip"} for vid in target_ids]
                        prog = st.progress(0, text="수집 진행")
                        log_ph = st.empty()
                        csv_path, total_cnt = parallel_collect_comments_streaming(
                            video_list=video_list,
                            rt_keys=YT_API_KEYS,
                            include_replies=st.session_state.get("adv_include_replies_collect", False),
                            max_total_comments=MAX_TOTAL_COMMENTS,
                            max_per_video=MAX_COMMENTS_PER_VIDEO,
                            log_callback=log_ph.write,
                            prog_callback=prog.progress
                        )
                        if total_cnt > 0:
                            st.session_state["adv_comments_path"] = csv_path
                            st.session_state["adv_total_count"] = int(total_cnt)
                            st.success(f"댓글 수집 완료! 총 {total_cnt:,}개")
                        else:
                            st.warning("댓글이 수집되지 않았습니다.")
                    finally:
                        release_lock()
                        gc.collect()
            if st.button("다음: AI분석으로 이동", type="primary", key="adv_go_to_step4"):
                st.session_state["focus_step"] = 4

    # ④ AI분석
    expanded4 = (st.session_state["focus_step"] == 4)
    with st.expander("④ AI분석", expanded=expanded4):
        adv_comments_path = st.session_state.get("adv_comments_path","")
        if not adv_comments_path or not os.path.exists(adv_comments_path):
            st.info("③에서 댓글 수집을 완료하면 여기에 분석 기능이 활성화됩니다.")
        else:
            st.write(f"분석 대상 전체 댓글 수: **{st.session_state.get('adv_total_count',0):,}**")
            analysis_keyword = st.text_input("관련 키워드(분석 컨텍스트)",
                                             value=st.session_state.get("last_keyword", ""),
                                             placeholder="예: 윤두준", key="adv_analysis_keyword")
            user_question_adv = st.text_area("사용자 질문", height=80, placeholder="예: 최근 반응은?", key="adv_user_question")

            if st.button("✨ AI 분석 실행", type="primary", key="adv_run_gem"):
                if not GEMINI_API_KEYS:
                    st.error("Gemini API Key가 없습니다.")
                else:
                    if not lock_guard_start_or_warn():
                        st.stop()
                    try:
                        append_log("고급", analysis_keyword, user_question_adv)
                        st.session_state["adv_history"] = []
                        st.session_state["adv_followups"] = []
                        a_text, _, _ = serialize_comments_for_llm_from_file(adv_comments_path)
                        system_instruction = (
                            "너는 유튜브 댓글을 분석하는 어시스턴트다. "
                            "아래 키워드와 댓글 샘플을 바탕으로 사용자 질문에 답하라. "
                            "핵심 포인트를 항목화하고, 대략적 비율과 대표 코멘트(10개미만)도 제시하라. 출력은 한국어. "
                            "키워드가 배우명(사람이름)이면 배우 중심으로 분석하라. "
                            "반드시 모든 댓글을 읽고 답변하라."
                        )
                        user_payload = f"[키워드]: {analysis_keyword}\n[질문]: {user_question_adv}\n\n[댓글 샘플]:\n{a_text}\n"
                        out = call_gemini_rotating(GEMINI_MODEL, GEMINI_API_KEYS, system_instruction, user_payload)
                        st.session_state["adv_result_text"] = out
                        st.session_state["adv_history"].append((user_question_adv or "최근 반응 요약", out))
                        st.success("AI 분석 완료")
                    finally:
                        release_lock()
                        gc.collect()

            if st.session_state.get("adv_result_text"):
                st.markdown("#### 📄 분석 결과 (최신)")
                last_q, last_a = st.session_state["adv_history"][-1]
                st.markdown(f"**Q. {last_q}**")
                st.markdown(st.session_state["adv_result_text"])

                # 정량요약: ④ 내부에서만 표시 (expander 중첩 회피)
                df_stats_cur = st.session_state.get("df_stats")
                render_quant_viz_from_paths(adv_comments_path, df_stats_cur, scope_label="(KST 기준)", wrap_in_expander=False)

                if st.session_state["adv_followups"]:
                    st.markdown("### 📝 추가 질문 히스토리")
                    for i, (q, a) in enumerate(st.session_state["adv_followups"], start=1):
                        with st.expander(f"{i}. {q}", expanded=False):
                            st.markdown(a or "_응답 없음_")
                st.markdown("#### ➕ 추가 질문하기")
                st.text_input("추가 질문", placeholder="예: 긍/부정 키워드 Top5는?",
                              key="adv_follow_q", on_change=handle_followup_advanced)
                st.button("질문 보내기(고급)", key="adv_follow_btn", on_click=handle_followup_advanced)

                if st.button("💾 세션 저장하기", key="adv_save_session_analysis"):
                    name, _ = save_current_session()
                    st.success(f"세션 저장 완료: {name}")

    # 하단: 다운로드 (전체댓글/영상)
    render_downloads_from_paths(st.session_state.get("adv_comments_path",""), st.session_state.get("df_stats"), prefix=f"adv_{len(st.session_state.get('selected_ids', []))}vids")

    if st.button("💾 세션 저장하기", key="adv_save_session_comments"):
        name, _ = save_current_session()
        st.success(f"세션 저장 완료: {name}")

# ===================== 3) 세션 아카이브 (GitHub) =====================
if tab == "세션":
    st.subheader("저장된 세션 아카이브 ")

    if not (GITHUB_TOKEN and GITHUB_REPO):
        st.warning("⚠️ GitHub 설정이 비어 있습니다. st.secrets에 GITHUB_TOKEN / GITHUB_REPO / GITHUB_BRANCH를 넣어주세요.")
    else:
        session_dirs = list_sessions_github()
        if not session_dirs:
            st.info("저장된 세션이 없습니다.")
        else:
            selected_name = st.selectbox("세션 선택", session_dirs, key="sess_select_github")

            # 선택 즉시: 두 개 파일만 표시 (전체댓글 / 전체영상)
            if selected_name:
                base_local = os.path.join(SESS_DIR, selected_name)
                os.makedirs(base_local, exist_ok=True)

                candidates_comments = [
                    "adv_comments_full.csv.gz", "adv_comments_full.csv",
                    "simple_comments_full.csv.gz", "simple_comments_full.csv"
                ]
                candidates_videos = [
                    "adv_videos.csv", "simple_videos.csv"
                ]

                def _ensure_local_and_button(label, filename_key, btn_key):
                    remote_path = f"sessions/{selected_name}/{filename_key}"
                    local_path = os.path.join(base_local, filename_key)
                    ok = github_download_file(GITHUB_REPO, GITHUB_BRANCH, remote_path, GITHUB_TOKEN, local_path)
                    if ok and os.path.exists(local_path):
                        with open(local_path, "rb") as f:
                            st.download_button(f"{label}", data=f.read(), file_name=filename_key, key=btn_key)
                        return True
                    return False

                st.markdown("### ⬇️ 빠른 다운로드")

                # 전체댓글
                got_comments = False
                for fn in candidates_comments:
                    if _ensure_local_and_button("전체댓글 CSV", fn, f"dl_{selected_name}_{fn}"):
                        got_comments = True
                        break
                if not got_comments:
                    st.caption("전체댓글 파일 없음")

                # 전체영상
                got_videos = False
                for fn in candidates_videos:
                    if _ensure_local_and_button("전체 영상 CSV", fn, f"dl_{selected_name}_{fn}"):
                        got_videos = True
                        break
                if not got_videos:
                    st.caption("전체영상 파일 없음")

                st.markdown("---")
                if st.button("📂 이 세션 불러오기", key="btn_load_session"):
                    st.session_state["__pending_session_load"] = selected_name
                    safe_rerun()

# ===================== 초기화 / 캐시 정리 =====================
st.markdown("---")
cols = st.columns(2)
with cols[0]:
    if st.button("🔄 초기화 하기", type="secondary"):
        st.session_state.clear()
        safe_rerun()
with cols[1]:
    if st.button("🧹 캐시/메모리 정리"):
        st.cache_data.clear()
        gc.collect()
        st.success("캐시와 메모리 정리 완료")



