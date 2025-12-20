import os
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, DefaultDict

import numpy as np
import psycopg
from psycopg.rows import dict_row
from scipy.optimize import minimize
from collections import defaultdict

from datetime import datetime, timedelta, timezone

# =============================================================================
# このスクリプトの目的
# -----------------------------------------------------------------------------
# 目的:
#   対戦結果（AがBに勝った/負けた/引き分け）から、
#   各アーキタイプの「強さパラメータ theta」を推定し、
#   さらに「A vs B の推定勝率」とその不確かさ（信頼区間）を保存する。
#
# 背景:
#   単純な勝率は、対戦相手の強さ・当たり運・サンプル数に強く左右される。
#   Bradley-Terry (BT) モデルでは、
#     「強いデッキほど勝ちやすい」
#   を確率モデルとして表現し、観測された勝敗から一貫した“強さ”を推定する。
#
# ここでのモデル（BT）:
#   A が B に勝つ確率を
#     P(A wins vs B) = sigmoid(theta_A - theta_B)
#   とする（sigmoid(x)=1/(1+exp(-x))）。
#
# 推定の不確かさ:
#   theta は推定値なので誤差がある。
#   ヘッセ行列（2階微分）から共分散 cov を近似し、
#   そこから A-B の差分の標準誤差 → 信頼区間 → 勝率区間(例 51%-70%) を出す。
# =============================================================================


# -----------------------------
# 設定（環境変数で上書き）
# -----------------------------
DATABASE_URL = os.environ["DATABASE_URL"]

SCOPE = os.environ.get("BT_SCOPE", "all")

def env_uuid(name: str) -> Optional[str]:
    """
    環境変数から uuid の文字列を取る（空文字は None 扱い）。
    """
    v = os.environ.get(name)
    v = v.strip() if v else None
    return v or None

FILTER_EVENT_ID = env_uuid("BT_EVENT_ID")
FILTER_GAME_TITLE_ID = env_uuid("BT_GAME_TITLE_ID")
FILTER_FORMAT_ID = env_uuid("BT_FORMAT_ID")

# DRAWの扱い： "half"=0.5勝扱い / "ignore"=捨てる
# - ignore: 引き分けは情報として使わない（観測から除外）
# - half:   引き分けを「両者が0.5勝」としてモデルに入れる（簡易な扱い）
DRAW_MODE = os.environ.get("BT_DRAW_MODE", "ignore")

# -----------------------------------------------------------------------------
# ★追加：期間フィルタの運用方針（matched_datetime はUTCとして保存されている前提）
#
# matches.matched_datetime が '2025-12-14 00:00:00' のような形式（tz無し）で入っており、
# それを「UTC時刻として解釈」して絞り込みたい、という要件に合わせる。
#
# BT_TIME_FILTER_MODE:
#   - "auto_weekly_utc": UTC週区切り（月曜0:00 UTC）で直近1週間 [from,to) を自動生成
#   - "env": BT_TIME_FROM/BT_TIME_TO をそのまま使う（手動指定）
#   - "off": 期間絞り込みを無効（全範囲。デバッグ用途）
#
# BT_TIME_FROM / BT_TIME_TO:
#   DBの形式に揃えるのが安全: 'YYYY-MM-DD HH:MM:SS'
#
# BT_WEEKLY_ANCHOR_ISO:
#   自動生成の基準時刻（再現性のための固定値）
#   例: '2025-12-20T12:00:00Z'
#
# BT_WEEKS_BACK:
#   何週間分遡るか（通常 1）
# -----------------------------------------------------------------------------

# CIと数値安定化
# 区間の信頼係数（95%CIなら 1.96）
# 例:
#   90% CI ≈ 1.645
#   95% CI ≈ 1.96
#   99% CI ≈ 2.576
CI_Z = float(os.environ.get("BT_CI_Z", "1.96"))
# Hessianの数値安定化（微小な正則化）
# データが疎だったり、デッキ同士がほとんど当たっていない場合に
# ヘッセ行列の逆が不安定になることがあるため、微小な対角成分を足す。
HESSIAN_RIDGE = float(os.environ.get("BT_HESSIAN_RIDGE", "1e-9"))

# 手動指定（envモードで使用）
BT_TIME_FROM = os.environ.get("BT_TIME_FROM")  # inclusive
BT_TIME_TO = os.environ.get("BT_TIME_TO")      # exclusive


# =============================================================================
# 期間ウィンドウ（matched_datetime は tz無しで UTC を意味する前提）
# =============================================================================

def compute_week_window_utc(
    *,
    anchor_utc: Optional[datetime] = None,
    weeks_back: int = 1,
) -> Tuple[str, str]:
    """
    UTC週区切り（月曜0:00 UTC）で [from,to) を作り、
    DBの matched_datetime（tz無し）に合わせて 'YYYY-MM-DD HH:MM:SS' で返す。
    """
    if anchor_utc is None:
        anchor_utc = datetime.now(timezone.utc)
    if anchor_utc.tzinfo is None:
        anchor_utc = anchor_utc.replace(tzinfo=timezone.utc)

    # 直近の「月曜 00:00:00 UTC」を求める
    # weekday(): Mon=0 ... Sun=6
    days_since_monday = anchor_utc.weekday()  # Mon=0
    this_monday_00 = (anchor_utc - timedelta(days=days_since_monday)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    to_utc = this_monday_00
    from_utc = to_utc - timedelta(days=7 * weeks_back)

    def fmt(dt: datetime) -> str:
        return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    return fmt(from_utc), fmt(to_utc)


def resolve_time_filter_window_utc() -> Tuple[Optional[str], Optional[str]]:
    """
    期間フィルタに使う (time_from, time_to) を決定する。

    - mode=off のときは (None, None) を返し、SQLに条件を入れない（全範囲）
    - mode=env のときは BT_TIME_FROM/TO をそのまま使う（手動）
    - mode=auto_weekly_utc のときは UTC週区切りで自動生成する（通常運用）
    """
    mode = os.environ.get("BT_TIME_FILTER_MODE", "auto_weekly_utc").strip().lower()

    if mode == "off":
        return None, None

    if mode == "env":
        # 手動指定。DBに合わせて 'YYYY-MM-DD HH:MM:SS' を推奨。
        return BT_TIME_FROM, BT_TIME_TO

    if mode == "auto_weekly_utc":
        anchor_iso = os.environ.get("BT_WEEKLY_ANCHOR_ISO")  # 例: 2025-12-20T12:00:00Z
        anchor_utc: Optional[datetime] = None
        if anchor_iso:
            s = anchor_iso.strip().replace("Z", "+00:00")
            anchor_utc = datetime.fromisoformat(s)
            if anchor_utc.tzinfo is None:
                anchor_utc = anchor_utc.replace(tzinfo=timezone.utc)

        weeks_back = int(os.environ.get("BT_WEEKS_BACK", "1"))
        return compute_week_window_utc(anchor_utc=anchor_utc, weeks_back=weeks_back)

    raise ValueError(f"Unknown BT_TIME_FILTER_MODE={mode!r}")


# =============================================================================
# DB読み取り
# =============================================================================

@dataclass
class MatchRow:
    """
    DBから取った1試合ぶんの最小情報。
    my_archtype_id: 自分のデッキ（アーキタイプ）ID
    opp_archtype_id: 相手のデッキID
    result: WIN/LOSE/DRAW（"my"視点の結果）
    """
    my_archtype_id: str
    opp_archtype_id: str
    result: str  # WIN/LOSE/DRAW


def fetch_matches(
    conn: psycopg.Connection,
    time_from: Optional[str],
    time_to: Optional[str],
) -> Tuple[List[MatchRow], Optional[str], Optional[str]]:
    where = ["m.is_delete = false", "ud.is_delete = false", "m.result in ('WIN','LOSE','DRAW')"]
    params: Dict[str, object] = {}

    if FILTER_EVENT_ID:
        where.append("m.event_id = %(event_id)s")
        params["event_id"] = FILTER_EVENT_ID

    if FILTER_GAME_TITLE_ID:
        where.append("ud.game_title_id = %(game_title_id)s")
        params["game_title_id"] = FILTER_GAME_TITLE_ID

    if FILTER_FORMAT_ID:
        where.append("a_my.format_id = %(format_id)s")
        params["format_id"] = FILTER_FORMAT_ID

    # 期間フィルタ（matched_datetime を UTC として扱う前提）
    if time_from:
        where.append("m.matched_datetime >= %(time_from)s")
        params["time_from"] = time_from
    if time_to:
        where.append("m.matched_datetime < %(time_to)s")
        params["time_to"] = time_to

    sql = f"""
    select
      ud.deck_id as my_archtype_id,
      m.opponent_deck_id as opp_archtype_id,
      m.result as result
    from matches m
    join user_decks ud
      on ud.user_deck_id = m.deck_id
    join archtypes a_my
      on a_my.archtype_id = ud.deck_id
    where {" and ".join(where)}
    """

    rows: List[MatchRow] = []
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        for r in cur.fetchall():
            rows.append(MatchRow(
                my_archtype_id=str(r["my_archtype_id"]),
                opp_archtype_id=str(r["opp_archtype_id"]),
                result=str(r["result"]),
            ))

    return rows, FILTER_GAME_TITLE_ID, FILTER_FORMAT_ID


def build_obs_and_pair_stats(rows: List[MatchRow]):
    """
    DBの生データ（MatchRow）を
      1) Bradley-Terry 推定に使う観測データ obs
      2) “実際の対戦数/勝数” の集計 pair_stats
    に変換する。

    -------------------------
    1) obs の形式
      obs: List[(winner_idx, loser_idx, weight)]
    - winner_idx: 勝ったデッキの index（0..N-1）
    - loser_idx:  負けたデッキの index
    - weight:     観測の重み（通常は1、DRAWをhalf扱いにすると0.5など）

    BTモデルでは「勝った/負けた」のみで尤度が書けるため、
    まず観測を勝敗のペアに落とす。

    -------------------------
    2) pair_stats（順序なし）
      (min_idx, max_idx) -> {
        n: 対戦数,
        low_wins: low側が勝った回数,
        high_wins: high側が勝った回数,
        draws: 引き分け数
      }

    ここで (low,high) に正規化しておくと、
    A vs B の集計が二重に数えられない。
    ただし DB 保存は A→B と B→A で2行作る（後段の upsert_matchup_probs）。
    """
    ids = set()
    for r in rows:
        ids.add(r.my_archtype_id)
        ids.add(r.opp_archtype_id)
    id_list = sorted(ids)
    id_to_idx = {a: i for i, a in enumerate(id_list)}

    obs: List[Tuple[int, int, float]] = []
    # 対戦数（アーキ別）: そのアーキが関与した試合数（双方+1）
    n_games: Dict[int, int] = {i: 0 for i in range(len(id_list))}
    # ペア統計（順序なし）
    pair_stats: DefaultDict[Tuple[int, int], Dict[str, int]] = defaultdict(
        lambda: {"n": 0, "low_wins": 0, "high_wins": 0, "draws": 0}
    )

    for r in rows:
        i = id_to_idx[r.my_archtype_id]
        j = id_to_idx[r.opp_archtype_id]
        low, high = (i, j) if i < j else (j, i)

        n_games[i] += 1
        n_games[j] += 1

        st = pair_stats[(low, high)]
        st["n"] += 1

        if r.result == "WIN":
            # i が勝者（my視点でWIN）
            obs.append((i, j, 1.0))
            if i == low:
                st["low_wins"] += 1
            else:
                st["high_wins"] += 1
        elif r.result == "LOSE":
            # my視点でLOSE → 相手(j)が勝者
            obs.append((j, i, 1.0))
            if j == low:
                st["low_wins"] += 1
            else:
                st["high_wins"] += 1
        elif r.result == "DRAW":
            st["draws"] += 1
            if DRAW_MODE == "ignore":
                # 引き分けをモデルに入れない（情報を使わない）
                continue
            # half: 引き分けを「双方0.5勝」として扱う簡易版
            # （厳密な引き分けモデルを入れたい場合は別モデルになる）
            obs.append((i, j, 0.5))
            obs.append((j, i, 0.5))

    return obs, id_to_idx, id_list, n_games, pair_stats


# =============================================================================
# BT推定（theta + cov）
# =============================================================================

def _sigmoid_stable(x: float) -> float:
    """
    sigmoid(x) = 1/(1+exp(-x)) をオーバーフローしにくい形で計算する。
    theta差が大きいと exp が極端になりやすいので、安全にする。
    """
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def fit_bradley_terry_with_cov(
    n_items: int,
    obs: List[Tuple[int, int, float]],
    ridge: float = HESSIAN_RIDGE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bradley-Terry モデルで theta を最尤推定し、推定誤差の近似共分散 cov を返す。

    ------------------------------------
    モデル:
      P(w beats l) = sigmoid(theta[w] - theta[l])

    観測（勝ち/負け）に対する負の対数尤度:
      nll(theta) = Σ weight * log(1 + exp(-(theta[w]-theta[l])))

    これを最小化（=尤度最大化）して theta を求める。

    ------------------------------------
    識別制約（平均0）:
      theta は全員に同じ定数を足しても確率が変わらない（差しか効かない）。
      そのため "thetaの平均=0" に正規化して、解を一意にする。

    実装上は
      - 評価時は theta を中心化して t を使う
      - 最後も theta を中心化して返す
    としている。

    ------------------------------------
    不確かさ（cov）:
      最尤推定の周りで二次近似すると、
      cov ≈ (ヘッセ行列 H)^{-1} になる（Fisher情報の逆、Laplace近似）。

      ただし平均0制約により1次元だけ特異になるので、
      平均成分を落とす射影 P を使って制約空間で擬似逆を取る。

    戻り値:
      theta: (n,)
      cov:   (n,n)
    """
    if n_items <= 1:
        theta = np.zeros((n_items,), dtype=float)
        cov = np.zeros((n_items, n_items), dtype=float)
        return theta, cov

    x0 = np.zeros((n_items,), dtype=float)

    def nll(theta: np.ndarray) -> float:
        t = theta - theta.mean()
        total = 0.0
        for w, l, weight in obs:
            d = t[w] - t[l]
            total += weight * np.log1p(np.exp(-d))
        return float(total)

    def grad(theta: np.ndarray) -> np.ndarray:
        # nll の勾配（1階微分）
        t = theta - theta.mean()
        g = np.zeros_like(t)
        for w, l, weight in obs:
            d = t[w] - t[l]
            s = 1.0 / (1.0 + np.exp(d))  # sigmoid(-d)
            g[w] += -weight * s
            g[l] += +weight * s
        # 平均0制約（中心化）と整合させるため、勾配も平均を0に寄せる
        g -= g.mean()
        return g

    res = minimize(nll, x0, jac=grad, method="L-BFGS-B")
    theta = res.x
    theta = theta - theta.mean()

    # 解析ヘッセ（2階微分）
    # BTのヘッセは「グラフのラプラシアン」の形になる（対角が正、非対角が負）
    t = theta - theta.mean()
    H = np.zeros((n_items, n_items), dtype=float)

    for w, l, weight in obs:
        d = float(t[w] - t[l])
        p = _sigmoid_stable(d)              # sigmoid(d)
        c = float(weight) * p * (1.0 - p)   # 二階微分係数（常に >= 0）
        H[w, w] += c
        H[l, l] += c
        H[w, l] -= c
        H[l, w] -= c

    # 数値安定化（微小な対角成分を足す）
    if ridge > 0.0:
        H = H + ridge * np.eye(n_items)

    # 平均0制約の部分空間へ射影（1ベクトル方向を落とす）
    one = np.ones((n_items, 1), dtype=float)
    P = np.eye(n_items) - (one @ one.T) / n_items
    Hc = P @ H @ P
    # 制約空間で擬似逆を取って共分散近似にする
    cov = np.linalg.pinv(Hc)
    return theta, cov


def _pair_logit_se(cov: np.ndarray, a: int, b: int) -> float:
    """
    logit差 m = theta[a]-theta[b] の標準誤差を計算する。

    共分散 cov があると、線形結合の分散は
      Var(theta[a]-theta[b]) = Var(theta[a]) + Var(theta[b]) - 2Cov(theta[a],theta[b])
    になる（分散の公式）。
    """
    var_m = float(cov[a, a] + cov[b, b] - 2.0 * cov[a, b])
    if var_m < 0.0:
        # 数値誤差でマイナスになり得るので0に丸める
        var_m = 0.0
    return math.sqrt(var_m)


def _matchup_prob_and_ci(theta: np.ndarray, cov: np.ndarray, a: int, b: int, z: float) -> Tuple[float, float, float, float]:
    """
    a vs b の
      - 推定勝率 p = sigmoid(theta[a]-theta[b])
      - その信頼区間 [p_lo, p_hi]
      - logit差の標準誤差 se_m
    を返す。

    区間の作り方:
      まず logit差 m の近似正規性（最尤推定の漸近性）を使い、
        m ± z * se_m
      を作る。
      それを sigmoid に戻して確率の区間にする。
    """
    m = float(theta[a] - theta[b])
    se_m = _pair_logit_se(cov, a, b)
    p = _sigmoid_stable(m)
    p_lo = _sigmoid_stable(m - z * se_m)
    p_hi = _sigmoid_stable(m + z * se_m)
    return p, p_lo, p_hi, se_m


# =============================================================================
# ★追加：run を作る
# =============================================================================

def _make_config_hash(
    *,
    scope: str,
    event_id: Optional[str],
    game_title_id: str,
    format_id: str,
    draw_mode: str,
    ci_z: float,
    ridge: float,
    window_from: Optional[str],
    window_to: Optional[str],
) -> str:
    """
    実行条件を1つの文字列にまとめてハッシュ化（比較・追跡用）。
    """
    payload = "|".join([
        f"scope={scope}",
        f"event_id={event_id or ''}",
        f"game_title_id={game_title_id}",
        f"format_id={format_id}",
        f"draw_mode={draw_mode}",
        f"ci_z={ci_z}",
        f"ridge={ridge}",
        f"window_from={window_from or ''}",
        f"window_to={window_to or ''}",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def create_run(
    conn: psycopg.Connection,
    *,
    window_from: Optional[str],
    window_to: Optional[str],
    n_rows: int,
    n_items: int,
    n_obs: int,
) -> str:
    """
    bt_matchup_runs に1行insertして run_id を返す。
    """
    if not FILTER_GAME_TITLE_ID or not FILTER_FORMAT_ID:
        raise RuntimeError(
            "BT_GAME_TITLE_ID / BT_FORMAT_ID must be set because bt_matchup_runs requires NOT NULL."
        )

    config_hash = _make_config_hash(
        scope=SCOPE,
        event_id=FILTER_EVENT_ID,
        game_title_id=FILTER_GAME_TITLE_ID,
        format_id=FILTER_FORMAT_ID,
        draw_mode=DRAW_MODE,
        ci_z=CI_Z,
        ridge=HESSIAN_RIDGE,
        window_from=window_from,
        window_to=window_to,
    )

    sql = """
    insert into bt_matchup_runs
      (window_from, window_to, scope, event_id, game_title_id, format_id,
       draw_mode, ci_z, hessian_ridge,
       n_rows, n_items, n_obs, config_hash)
    values
      (%(window_from)s, %(window_to)s, %(scope)s, %(event_id)s, %(game_title_id)s, %(format_id)s,
       %(draw_mode)s, %(ci_z)s, %(hessian_ridge)s,
       %(n_rows)s, %(n_items)s, %(n_obs)s, %(config_hash)s)
    returning run_id
    """

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, {
            "window_from": window_from,  # 'YYYY-MM-DD HH:MM:SS' or None
            "window_to": window_to,
            "scope": SCOPE,
            "event_id": FILTER_EVENT_ID,
            "game_title_id": FILTER_GAME_TITLE_ID,
            "format_id": FILTER_FORMAT_ID,
            "draw_mode": DRAW_MODE,
            "ci_z": CI_Z,
            "hessian_ridge": HESSIAN_RIDGE,
            "n_rows": int(n_rows),
            "n_items": int(n_items),
            "n_obs": int(n_obs),
            "config_hash": config_hash,
        })
        r = cur.fetchone()
        return str(r["run_id"])


def upsert_matchup_probs(
    conn: psycopg.Connection,
    *,
    run_id: str,
    archtype_ids: List[str],
    theta: np.ndarray,
    cov: np.ndarray,
    pair_stats,
    z: float = CI_Z,
) -> None:
    """
    ★変更：bt_matchup_probs は run_id 配下に保存する。
    """
    sql = """
    insert into bt_matchup_probs
      (run_id,
       archtype_a_id, archtype_b_id,
       p_a_win, p_a_win_lo, p_a_win_hi, se_logit,
       n_ab, a_wins, b_wins, draws, updated_datetime)
    values
      (%(run_id)s,
       %(a)s, %(b)s,
       %(p)s, %(p_lo)s, %(p_hi)s, %(se_logit)s,
       %(n)s, %(a_wins)s, %(b_wins)s, %(draws)s, now())
    on conflict (run_id, archtype_a_id, archtype_b_id)
    do update set
      p_a_win = excluded.p_a_win,
      p_a_win_lo = excluded.p_a_win_lo,
      p_a_win_hi = excluded.p_a_win_hi,
      se_logit = excluded.se_logit,
      n_ab = excluded.n_ab,
      a_wins = excluded.a_wins,
      b_wins = excluded.b_wins,
      draws = excluded.draws,
      updated_datetime = now()
    """

    with conn.cursor() as cur:
        for (low, high), st in pair_stats.items():
            a_id = archtype_ids[low]
            b_id = archtype_ids[high]

            p, p_lo, p_hi, se_m = _matchup_prob_and_ci(theta, cov, low, high, z)
            cur.execute(sql, {
                "run_id": run_id,
                "a": a_id,
                "b": b_id,
                "p": float(p),
                "p_lo": float(p_lo),
                "p_hi": float(p_hi),
                "se_logit": float(se_m),
                "n": int(st["n"]),
                "a_wins": int(st["low_wins"]),
                "b_wins": int(st["high_wins"]),
                "draws": int(st["draws"]),
            })

            p2, p2_lo, p2_hi, se_m2 = _matchup_prob_and_ci(theta, cov, high, low, z)
            cur.execute(sql, {
                "run_id": run_id,
                "a": b_id,
                "b": a_id,
                "p": float(p2),
                "p_lo": float(p2_lo),
                "p_hi": float(p2_hi),
                "se_logit": float(se_m2),
                "n": int(st["n"]),
                "a_wins": int(st["high_wins"]),
                "b_wins": int(st["low_wins"]),
                "draws": int(st["draws"]),
            })


def main() -> None:
    # runの実行時刻（ログ用途。DBの executed_at は now()）
    as_of_iso = os.environ.get("BT_AS_OF")
    if not as_of_iso:
        as_of_iso = datetime.now(timezone.utc).isoformat()

    # ★期間をここで確定（fetchとrun保存で同じ値を使う）
    window_from, window_to = resolve_time_filter_window_utc()

    with psycopg.connect(DATABASE_URL) as conn:
        rows, game_title_id, format_id = fetch_matches(conn, window_from, window_to)
        if len(rows) == 0:
            print("No matches found. Nothing to estimate.")
            return

        obs, id_to_idx, id_list, n_games, pair_stats = build_obs_and_pair_stats(rows)
        if len(obs) == 0:
            print("No usable observations (maybe all DRAW ignored).")
            return

        theta, cov = fit_bradley_terry_with_cov(len(id_list), obs)

        # ★run作成（この run_id に結果が紐づく）
        run_id = create_run(
            conn,
            window_from=window_from,
            window_to=window_to,
            n_rows=len(rows),
            n_items=len(id_list),
            n_obs=len(obs),
        )

        # ★run_id配下に保存
        upsert_matchup_probs(
            conn,
            run_id=run_id,
            archtype_ids=id_list,
            theta=theta,
            cov=cov,
            pair_stats=pair_stats,
            z=CI_Z,
        )

        conn.commit()

        # 参考ログ
        print(f"run_id={run_id} as_of={as_of_iso} window=[{window_from},{window_to}) rows={len(rows)} obs={len(obs)} items={len(id_list)}")
        print(f"Matchup pairs saved (undirected): {len(pair_stats)}")


if __name__ == "__main__":
    main()