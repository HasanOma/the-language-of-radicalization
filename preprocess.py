#!/usr/bin/env python
"""
preprocess.py
pipeline:
    • forum + Telegram + Discord + Twitter ingestion
    • rhetorical features
    • emotion scores (pysentimiento)
    • per-year balanced sampling helper
"""

from __future__ import annotations
import json, math, os, re, string, warnings, logging, sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Final

import numpy as np
import pandas as pd
import psycopg2
import emoji, spacy
from tqdm import tqdm
import nltk
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─────────────────────────  logging  ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("preprocess")

# ───────────────────────  files  ─────────────────────
DATA_DIR:       Final[Path] = Path("data/revised-preprocess/data80k")
RAW_FILE:       Final[Path] = DATA_DIR / "01_raw.feather"
FEAT_FILE:      Final[Path] = DATA_DIR / "02_features.feather"
CONFIG:         Final[Path] = Path("config_ideology.json")  
TWEET_PATH:     Final[Path] = Path("tweets.csv")           

LIMIT_PER_FORUM:  Final[int | None] = 80000   
LIMIT_PER_SOCIAL: Final[int | None] = 80000

DATA_DIR.mkdir(parents=True, exist_ok=True)
pd.options.mode.chained_assignment = None   

# ─────────────────────  spaCy / NLTK boot-strap  ──────────────
for pkg in ("punkt", "stopwords", "wordnet"):
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
LEMM       = WordNetLemmatizer()
NLP        = spacy.load("en_core_web_sm", disable=["ner", "parser"])

INCL = {"we", "our", "us", "together"}
EXCL = {"they", "them", "those", "enemy"}

# ────────────────────────── config ────────────────────────────
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.warn(f"JSON parse error in {path}: {exc}")
        return default

conf = _read_json(CONFIG, {})


IDEOLOGY_MAP      : dict[str,str] = conf.get("ideology_map", {})
TG_FROM_JSON      : dict[str,str] = conf.get("telegram_channels", {})
DC_FROM_JSON      : dict[str,str] = conf.get("discord_servers", {})

def canon(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())

FULL_MAP: dict[str, str] = {
    **{canon(k): v for k, v in IDEOLOGY_MAP.items()},
    **{canon(k): v for k, v in TG_FROM_JSON.items()},
    **{canon(k): v for k, v in DC_FROM_JSON.items()},
}

TELEGRAM_CHANNELS = TG_FROM_JSON
DISCORD_SERVERS   = DC_FROM_JSON

if not TELEGRAM_CHANNELS:
    log.warning("No Telegram channels in JSON – using LAST-RESORT placeholders")
    TELEGRAM_CHANNELS = {"White Phoenix Action": "far_right",
                         "Proud Boys": "far_right"}
if not DISCORD_SERVERS:
    log.warning("No Discord servers in JSON – using LAST-RESORT placeholders")
    DISCORD_SERVERS   = {"Nazi Weeb Gamers": "far_right_accelerationist",
                         "Traditional Catholicism": "religious_extremist"}

log.info("Config loaded: %d ideology map entries, %d TG, %d Discord",
         len(IDEOLOGY_MAP), len(TELEGRAM_CHANNELS), len(DISCORD_SERVERS))

# ──────────────────────────  Emotion analyser  ─────────────────────────
from pysentimiento import create_analyzer
_EMO_MODEL = create_analyzer(task="emotion", lang="en")
EMO_COLS   = list(_EMO_MODEL.model.config.id2label.values())
log.info("🔮 using pysentimiento (%s)", ", ".join(EMO_COLS))

def _emotion_scores(text: str) -> dict[str, float]:
    if not text:
        return dict.fromkeys(EMO_COLS, 0.0)
    return dict(_EMO_MODEL.predict(text).probas)

# ────────────────────────  helpers  ────────────────────────
_PSQL = dict(
    user=os.getenv("PSQL_USER", "postgres"),
    password=os.getenv("PSQL_PWD", ""),
    host=os.getenv("PSQL_HOST", "localhost"),
    port=os.getenv("PSQL_PORT", 5432),
)


def _conn(db: str):
    return psycopg2.connect(dbname=db, **_PSQL)


def first_schema_with(cx, table: str) -> str | None:
    with cx.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema
            FROM   information_schema.tables
            WHERE  table_name = %s
            ORDER  BY (table_schema = 'public') DESC
            LIMIT  1
            """,
            (table,),
        )
        row = cur.fetchone()
    return row[0] if row else None


def cols_in(cx, schema: str, table: str) -> set[str]:
    with cx.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM   information_schema.columns
            WHERE  table_schema = %s AND table_name = %s
            """,
            (schema, table),
        )
        return {r[0] for r in cur.fetchall()}

# ───────────────────────  DB  ───────────────────────
def list_extreme_dbs() -> list[str]:
    """Return DBs whose name starts with 'extreme'."""
    with psycopg2.connect(dbname="postgres", **_PSQL) as cx, cx.cursor() as cur:
        cx.autocommit = True
        cur.execute("SELECT datname FROM pg_database WHERE datname LIKE 'extreme%';")
        return [r[0] for r in cur.fetchall()]


def db_has_posts(db: str) -> bool:
    try:
        with _conn(db) as cx:
            return first_schema_with(cx, "posts") is not None
    except Exception:
        return False

# ───────────────────────  forum extractor  ────────────────────
def _safe_ident(name: str) -> str:
    """Very conservative quoting to defend against SQL injection."""
    return "\"" + name.replace("\"", "\"\"") + "\""


def fetch_forum(db: str, forum_name: str, limit: int | None) -> pd.DataFrame:
    """Fetch posts from one forum inside an ExtremeBB/CrimeBB database."""
    with _conn(db) as cx:
        schema = first_schema_with(cx, "posts")
        if not schema:
            log.warning("%s: no posts table → skipped", db)
            return pd.DataFrame()

        cols = cols_in(cx, schema, "posts")
        sel = [
            "p.id",
            "p.creator"    if "creator"     in cols else "NULL AS creator",
            "p.creator_id" if "creator_id"  in cols else "NULL::int AS creator_id",
            "p.content",
            "p.created_on" if "created_on"  in cols else "p.timestamp AS created_on",
            "p.is_a_reply" if "is_a_reply"  in cols else "FALSE AS is_a_reply",
        ]
        if "thread_id" in cols:
            sel.append("t.name AS thread_name")
        if "board_id" in cols:
            sel.append("b.name AS board_name")

        base = f"""
            SELECT {', '.join(sel)}
              FROM {_safe_ident(schema)}.posts p
            {f"LEFT JOIN {_safe_ident(schema)}.threads t ON p.thread_id = t.id" if 'thread_id' in cols else ''}
            {f"LEFT JOIN {_safe_ident(schema)}.boards  b ON p.board_id  = b.id" if 'board_id'  in cols else ''}
        """

        def try_query(where_clause: str | None, params: tuple = ()) -> pd.DataFrame:
            q = base + (f" WHERE {where_clause}" if where_clause else "")
            if limit:
                q += f" LIMIT {limit}"
            with cx.cursor() as cur:
                cur.execute(q, params)
                rows = cur.fetchall()
                if not rows:
                    return pd.DataFrame()
                names = [c[0] for c in cur.description]
                df = pd.DataFrame(rows, columns=names)
            df["source"]   = db
            df["forum"]    = forum_name
            df["ideology"] = FULL_MAP.get(canon(forum_name), forum_name)
            return df

        with cx.cursor() as cur:
            cur.execute(f"SELECT id FROM {_safe_ident(schema)}.sites WHERE name = %s", (forum_name,))
            row = cur.fetchone()
        if row:
            df = try_query("p.site_id = %s", (row[0],))
            if not df.empty:
                return df

        with cx.cursor() as cur:
            cur.execute(f"SELECT id FROM {_safe_ident(schema)}.sites WHERE lower(name) = lower(%s)", (forum_name,))
            row = cur.fetchone()
        if row:
            df = try_query("p.site_id = %s", (row[0],))
            if not df.empty:
                return df

        return try_query(None)

# ───────── Telegram / Discord helpers (extremecc_db) ──────────
def _fetch_social(db: str,
                  table_prefix: str,          
                  name_field  : str,          
                  name_value  : str,          
                  limit       : int | None) -> pd.DataFrame:
    """
    Generic helper for social-media exports that live in `extremecc_db`.

    Telegram:  one channel  → telegram_messages.channel_id
    Discord :  one *server* → many channels  → discord_messages.channel_id
    """
    with _conn(db) as cx:
        sch = first_schema_with(
            cx,
            f"{table_prefix}_channels" if table_prefix == "telegram"
                                       else f"{table_prefix}_servers"
        )
        if sch is None:
            return pd.DataFrame()

        if table_prefix == "telegram":
            id_col = "name" if name_field == "name" else name_field
            with cx.cursor() as cur:
                cur.execute(
                    f'''SELECT id
                        FROM   {_safe_ident(sch)}.telegram_channels
                        WHERE  {id_col} = %s''',
                    (name_value,)
                )
                row = cur.fetchone()
            if not row:
                return pd.DataFrame()
            obj_ids = [row[0]]                       
            fk_col  = "channel_id"

        else:  
            with cx.cursor() as cur:
                cur.execute(
                    f'''SELECT id
                        FROM   {_safe_ident(sch)}.discord_channels
                        WHERE  server_id = %s''',
                    (name_value,)
                )
                obj_ids = [r[0] for r in cur.fetchall()]
            if not obj_ids:
                log.warning("Discord server '%s' not found → skipped", name_value)
                return pd.DataFrame()
            fk_col = "channel_id"

        msg_tbl  = f"{table_prefix}_messages"
        msg_cols = cols_in(cx, sch, msg_tbl)

        creator_expr = (
            "name"      if "name"      in msg_cols else
            "username"  if "username"  in msg_cols else
            "user_id::text"
        )

        if "timestamp" in msg_cols:
            ts_expr = "timestamp"
        elif "created_on" in msg_cols:
            ts_expr = "created_on"
        elif "time" in msg_cols:
            ts_expr = "\"time\""
        else:
            ts_expr = "NULL::text"

        ids_sql = ", ".join(["%s"] * len(obj_ids))
        q = f"""
            SELECT id, user_id, {creator_expr} AS creator,
                   content, {ts_expr} AS created_on
              FROM {_safe_ident(sch)}.{msg_tbl}
             WHERE {fk_col} IN ({ids_sql})
        """
        if limit:
            q += f" LIMIT {limit}"

        with cx.cursor() as cur:
            cur.execute(q, tuple(obj_ids))
            rows = cur.fetchall()
            cols = [c[0] for c in cur.description]

    df = pd.DataFrame(rows, columns=cols)
    df["created_on"] = pd.to_datetime(df["created_on"], errors="coerce", utc=True)
    df["source"]     = table_prefix
    df["forum"]      = name_value
    df["ideology"]   = FULL_MAP.get(canon(name_value), name_value)

    log.info("Fetched %s messages from %s '%s'",
             f"{len(df):,}", table_prefix, name_value)
    return df

def fetch_telegram(channel: str, limit: int | None = LIMIT_PER_SOCIAL):
    print(f"Fetching Telegram channel '{channel}'")
    return _fetch_social("extremecc_db", "telegram", "name", channel, limit)


def fetch_discord(server: str, limit: int | None = LIMIT_PER_SOCIAL):
    print(f"Fetching Discord server '{server}'")
    return _fetch_social("extremecc_db", "discord", "name", server, limit)


def fetch_twitter(path: Path | str = TWEET_PATH) -> pd.DataFrame:
    """
    The Kaggle dump we use is 100 % pro-ISIS, so every row → jihadist.
    """
    path = Path(path)
    if not path.exists():
        alt = DATA_DIR / path.name
        if alt.exists():
            path = alt
        else:
            log.warning("⚠️  tweets.csv not found – Twitter dataset skipped")
            return pd.DataFrame()

    df = (pd.read_csv(path)
            .rename(columns={"tweets": "content",
                             "username": "creator",
                             "time": "created_on"}))

    df["ideology"]  = "jihadist"              
    df["creator_id"] = df["creator"].fillna("anon").astype(str)
    df["id"]         = df.index.map(lambda i: f"tw_{i}")
    df["forum"]      = "twitter"
    df["source"]     = "twitter"
    df["created_on"] = pd.to_datetime(df["created_on"],
                                      errors="coerce", utc=True)

    return df[["id","creator","creator_id","content",
               "created_on","source","forum","ideology"]]

# ───────────────────────────── sampler ──────────────────────────────

def sample_fixed_n_by_year(df: pd.DataFrame, total=80000, oversample=True, seed=42) -> pd.DataFrame:
    if total is None or df.empty:
        return df.copy()
    df["year"] = pd.to_datetime(df["created_on"], errors="coerce").dt.year
    groups = {y: g for y, g in df.groupby("year") if not pd.isna(y)}
    if not groups:
        return df.head(total)
    quota  = math.ceil(total / len(groups))
    take   = {y: min(quota, len(g)) for y, g in groups.items()}
    left   = total - sum(take.values())
    for y in groups:
        if left == 0:
            break
        spare = len(groups[y]) - take[y]
        extra = min(spare, left)
        take[y] += extra; left -= extra
    rng = np.random.RandomState(seed)
    parts = [groups[y].sample(take[y], replace=oversample and len(groups[y]) < take[y], random_state=rng) for y in groups]
    out = pd.concat(parts, ignore_index=True).drop(columns="year")
    if oversample and len(out) < total:
        out = pd.concat([out, out.sample(total-len(out), replace=True, random_state=rng)], ignore_index=True)
    return out

# ─────────────────────── RAW compilation ────────────────────────

def list_forums(db: str) -> list[str]:
    with _conn(db) as cx:
        sch = first_schema_with(cx, "sites")
        if not sch:
            return []
        with cx.cursor() as cur:
            cur.execute(f'SELECT name FROM {_safe_ident(sch)}.sites;')
            return [r[0] for r in cur.fetchall()]


def build_raw() -> pd.DataFrame:
    dbs = list_extreme_dbs()
    if not dbs:
        raise RuntimeError("No databases starting with 'extreme%' found")

    dfs: list[pd.DataFrame] = []

    log.info("📚  scanning %d databases for forum dumps", len(dbs))
    for db in tqdm(dbs, desc="databases", unit="db"):
        if not db_has_posts(db):
            continue
        for forum in list_forums(db):
            if db != "extremebb_survivalist_boards_2024_09_04":
                print(f"Fetching {forum} from {db}")
                df_forum = fetch_forum(db, forum, LIMIT_PER_FORUM)
                if not df_forum.empty:
                    dfs.append(sample_fixed_n_by_year(df_forum, LIMIT_PER_FORUM))

    log.info("💬  loading Telegram channels")
    for chan in tqdm(TELEGRAM_CHANNELS, desc="telegram"):
        dfs.append(fetch_telegram(chan, LIMIT_PER_SOCIAL))

    log.info("💬  loading Discord servers")
    for srv in tqdm(DISCORD_SERVERS, desc="discord"):
        dfs.append(fetch_discord(srv, LIMIT_PER_SOCIAL))

    dfs.append(fetch_twitter())

    raw = pd.concat([d for d in dfs if not d.empty], ignore_index=True)

    drop_forums: Iterable[str] = conf.get("drop_forums", [])
    if drop_forums:
        raw = raw[~raw["forum"].isin(drop_forums)]

    for col in ("id", "creator_id"):
        if col in raw.columns:
            raw[col] = raw[col].astype(str)
    if "created_on" in raw.columns:
        raw["created_on"] = pd.to_datetime(raw["created_on"], errors="coerce", utc=True)

    raw.to_feather(RAW_FILE)

    log.info(
        "✅  RAW dataset ready (%s rows)\n%s",
        f"{len(raw):,}",
        "\n" + "\n".join(f"   {k:<9s}: {v:>8,}" for k, v in Counter(raw["source"]).items()),
    )
    return raw

# ────────────────────── feature engineering ─────────────────────
URL_RE      = re.compile(r"https?://\S+|www\.\S+")
TAG_RE      = re.compile(r"\*\*\*[A-Z]+?\*\*\*.*?(?=\*\*\*|\s|$)", re.S)
HTML_RE     = re.compile(r"<.*?>")
PUNCT_TABLE = str.maketrans({p: " " for p in string.punctuation})


def normalise(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = URL_RE.sub(" ", text)
    text = TAG_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    text = emoji.demojize(text)
    text = text.translate(PUNCT_TABLE)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    return [LEMM.lemmatize(t) for t in wordpunct_tokenize(text.lower()) if t.isalpha() and t not in STOP_WORDS]


def rhet_vec(text: str) -> dict[str, float]:
    if not text:
        return {"incl": 0.0, "excl": 0.0}
    MAXLEN = NLP.max_length
    chunks = (text[i : i + MAXLEN] for i in range(0, len(text), MAXLEN))
    incl = excl = tok_total = 0
    for chunk in chunks:
        doc = NLP(chunk)
        tok_total += len(doc)
        for t in doc:
            low = t.lower_
            if low in INCL:
                incl += 1
            elif low in EXCL:
                excl += 1
    tok_total = tok_total or 1
    return {"incl": incl / tok_total, "excl": excl / tok_total}


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in tqdm(raw.itertuples(index=False), total=len(raw),
                  desc="🛠️  features", unit="row"):
        clean = normalise(r.content)
        emo   = _emotion_scores(clean)
        tot   = sum(emo.values()) or 1.0          
        emo   = {k: v/tot for k, v in emo.items()}
        feats = {
            "cleaned_content": clean,
            "token_count"    : len(tokens(clean)),
            **emo,
            **rhet_vec(clean),
        }
        rows.append({**r._asdict(), **feats})
    feat_df = pd.DataFrame(rows)
    feat_df.to_feather(FEAT_FILE)
    log.info("✅  FEATURES saved → %s  (%s rows)", FEAT_FILE, f"{len(feat_df):,}")
    return feat_df

# ───────────────────────────── MAIN ──────────────────────────────

def main(argv: list[str] | None = None):
    argv = argv or sys.argv[1:]
    if argv:
        global TWEET_PATH
        TWEET_PATH = Path(argv[0])
    raw = build_raw()
    build_features(raw)

if __name__ == "__main__":
    main()