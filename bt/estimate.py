import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, DefaultDict

import numpy as np
import psycopg
from psycopg.rows import dict_row
from scipy.optimize import minimize
from collections import defaultdict

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

# 出力スコープ（all / event:<uuid> / user:<uuid> など）
SCOPE = os.environ.get("BT_SCOPE", "all")

# 任意フィルタ
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


def fetch_matches(conn: psycopg.Connection) -> Tuple[List[MatchRow], Optional[str], Optional[str]]:
    """
    DBから対戦結果を取り出す。

    NOTE:
      - ここでは user_decks.deck_id をアーキタイプIDとして扱っている（スキーマ仕様）。
      - format_id は user_decks にないため archtypes.format_id を使って絞っている。
      - game_title_id / format_id は「フィルタ指定された値」をそのまま出力に入れる。
        => bt_matchup_probs の NOT NULL 制約があるので、環境変数で指定必須になる。
    """
    where = ["m.is_delete = false", "ud.is_delete = false", "m.result in ('WIN','LOSE','DRAW')"]
    params = {}

    if FILTER_EVENT_ID:
        where.append("m.event_id = %(event_id)s")
        params["event_id"] = FILTER_EVENT_ID

    if FILTER_GAME_TITLE_ID:
        where.append("ud.game_title_id = %(game_title_id)s")
        params["game_title_id"] = FILTER_GAME_TITLE_ID

    if FILTER_FORMAT_ID:
        where.append("a_my.format_id = %(format_id)s")
        params["format_id"] = FILTER_FORMAT_ID

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

    game_title_id = FILTER_GAME_TITLE_ID
    format_id = FILTER_FORMAT_ID
    return rows, game_title_id, format_id


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
# BT推定（theta と共分散 cov）
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
            # sigmoid(-d) = 1/(1+exp(d))
            s = 1.0 / (1.0 + np.exp(d))
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


def _matchup_prob_and_ci(
    theta: np.ndarray,
    cov: np.ndarray,
    a: int,
    b: int,
    z: float,
) -> Tuple[float, float, float, float]:
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
    m_lo = m - z * se_m
    m_hi = m + z * se_m
    p_lo = _sigmoid_stable(m_lo)
    p_hi = _sigmoid_stable(m_hi)
    return p, p_lo, p_hi, se_m


def upsert_matchup_probs(
    conn: psycopg.Connection,
    as_of_iso: str,
    scope: str,
    game_title_id: str,
    format_id: str,
    archtype_ids: List[str],
    theta: np.ndarray,
    cov: np.ndarray,
    pair_stats,
    z: float = CI_Z,
) -> None:
    """
    bt_matchup_probs へ、実対戦があるペアだけ保存する。

    保存内容（A→Bの行）:
      - n_ab: 実対戦数（観測数）
      - a_wins / b_wins / draws: 実測の勝敗回数（参考）
      - p_a_win:     BTモデルから推定した AがBに勝つ確率
      - p_a_win_lo:  推定勝率の下限（信頼区間）
      - p_a_win_hi:  推定勝率の上限（信頼区間）
      - se_logit:    logit差の標準誤差（診断用・UIで非表示でもOK）

    ここでは (low,high) の順序なし統計を
      low->high と high->low の2行に展開して保存する。
    """
    sql = """
    insert into bt_matchup_probs
      (as_of, scope, game_title_id, format_id,
       archtype_a_id, archtype_b_id,
       p_a_win, p_a_win_lo, p_a_win_hi, se_logit,
       n_ab, a_wins, b_wins, draws, updated_datetime)
    values
      (%(as_of)s, %(scope)s, %(game_title_id)s, %(format_id)s,
       %(a)s, %(b)s,
       %(p)s, %(p_lo)s, %(p_hi)s, %(se_logit)s,
       %(n)s, %(a_wins)s, %(b_wins)s, %(draws)s, now())
    on conflict (scope, game_title_id, format_id, archtype_a_id, archtype_b_id)
    do update set
      as_of = excluded.as_of,
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

            # low -> high
            p, p_lo, p_hi, se_m = _matchup_prob_and_ci(theta, cov, low, high, z)
            cur.execute(sql, {
                "as_of": as_of_iso,
                "scope": scope,
                "game_title_id": game_title_id,
                "format_id": format_id,
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

            # high -> low（勝数も入れ替える）
            p2, p2_lo, p2_hi, se_m2 = _matchup_prob_and_ci(theta, cov, high, low, z)
            cur.execute(sql, {
                "as_of": as_of_iso,
                "scope": scope,
                "game_title_id": game_title_id,
                "format_id": format_id,
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
    """
    エントリポイント。
    1) DBから試合を取る
    2) 観測(obs)を作る
    3) BT推定で theta + cov を作る
    4) マッチアップ推定勝率と信頼区間を保存する
    """
    as_of_iso = os.environ.get("BT_AS_OF")  # 例: 2025-12-18T00:00:00Z
    if not as_of_iso:
        from datetime import datetime, timezone
        as_of_iso = datetime.now(timezone.utc).isoformat()

    with psycopg.connect(DATABASE_URL) as conn:
        rows, game_title_id, format_id = fetch_matches(conn)
        if len(rows) == 0:
            print("No matches found. Nothing to estimate.")
            return

        obs, id_to_idx, id_list, n_games, pair_stats = build_obs_and_pair_stats(rows)

        if len(obs) == 0:
            print("No usable observations (maybe all DRAW ignored).")
            return

        theta, cov = fit_bradley_terry_with_cov(len(id_list), obs)

        if not game_title_id or not format_id:
            raise RuntimeError(
                "game_title_id / format_id must be set for bt_matchup_probs (NOT NULL). "
                "Pass BT_GAME_TITLE_ID and BT_FORMAT_ID, or adjust schema."
            )

        upsert_matchup_probs(
            conn=conn,
            as_of_iso=as_of_iso,
            scope=SCOPE,
            game_title_id=game_title_id,
            format_id=format_id,
            archtype_ids=id_list,
            theta=theta,
            cov=cov,
            pair_stats=pair_stats,
            z=CI_Z,
        )

        conn.commit()

        # ログ: 上位のtheta（参考）
        order = np.argsort(-np.exp(theta))
        print("Top 10 ratings:")
        for k in order[:10]:
            # cov[k,k] は theta[k] の分散（近似）。sqrtで標準誤差。
            se_k = math.sqrt(max(float(cov[k, k]), 0.0))
            print(
                f"{id_list[k]} rating={math.exp(theta[k]):.4f} "
                f"theta={theta[k]:+.4f} se={se_k:.4f} n_games={n_games.get(int(k), 0)}"
            )

        print(f"Matchup pairs saved (undirected): {len(pair_stats)}")


if __name__ == "__main__":
    main()
