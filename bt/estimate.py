import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import psycopg
from psycopg.rows import dict_row
from scipy.optimize import minimize


# -----------------------------
# 設定（環境変数で上書き）
# -----------------------------
DATABASE_URL = os.environ["DATABASE_URL"]

# 出力スコープ（all / event:<uuid> / user:<uuid> など）
SCOPE = os.environ.get("BT_SCOPE", "all")

# 任意フィルタ
FILTER_EVENT_ID = os.environ.get("BT_EVENT_ID")  # uuid文字列
FILTER_GAME_TITLE_ID = os.environ.get("BT_GAME_TITLE_ID")  # uuid文字列
FILTER_FORMAT_ID = os.environ.get("BT_FORMAT_ID")  # uuid文字列

# DRAWの扱い： "half"=0.5勝扱い / "ignore"=捨てる
DRAW_MODE = os.environ.get("BT_DRAW_MODE", "ignore")


@dataclass
class MatchRow:
    my_archtype_id: str
    opp_archtype_id: str
    result: str  # WIN/LOSE/DRAW


def fetch_matches(conn: psycopg.Connection) -> Tuple[List[MatchRow], Optional[str], Optional[str]]:
    """
    matches + user_decks を JOIN して
    (my_archtype_id, opp_archtype_id, result) を取得する。
    ついでに game_title_id / format_id を（指定されていれば）そのまま返す。
    """
    where = ["m.is_delete = false", "ud.is_delete = false", "m.result in ('WIN','LOSE','DRAW')"]
    params = {}

    if FILTER_EVENT_ID:
        where.append("m.event_id = %(event_id)s")
        params["event_id"] = FILTER_EVENT_ID

    # game_title_id / format_id は "user_decks" or "archtypes" / "formats" 経由でも絞れるが、
    # まずは最小：user_decks.game_title_id で絞る（あなたの定義にある）
    if FILTER_GAME_TITLE_ID:
        where.append("ud.game_title_id = %(game_title_id)s")
        params["game_title_id"] = FILTER_GAME_TITLE_ID

    # format_id は user_decks には無いので、my archtype の format_id を使う（archtypes.format_id）
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

    # 出力に入れる game_title_id / format_id は「フィルタに使った値」をそのまま採用（指定されていなければNULL）
    game_title_id = FILTER_GAME_TITLE_ID
    format_id = FILTER_FORMAT_ID
    return rows, game_title_id, format_id


def build_pairs(rows: List[MatchRow]) -> Tuple[List[Tuple[int, int, float]], Dict[str, int], List[str], Dict[int, int]]:
    """
    BTに渡すために
    - archtype_id -> index
    - 観測: (winner_idx, loser_idx, weight)
    に変換する。
    DRAWは half なら 0.5勝/0.5敗として2本に分ける。
    """
    ids = set()
    for r in rows:
        ids.add(r.my_archtype_id)
        ids.add(r.opp_archtype_id)
    id_list = sorted(ids)
    id_to_idx = {a: i for i, a in enumerate(id_list)}

    obs: List[Tuple[int, int, float]] = []
    n_games: Dict[int, int] = {i: 0 for i in range(len(id_list))}

    for r in rows:
        i = id_to_idx[r.my_archtype_id]
        j = id_to_idx[r.opp_archtype_id]

        if r.result == "WIN":
            obs.append((i, j, 1.0))
            n_games[i] += 1
            n_games[j] += 1
        elif r.result == "LOSE":
            obs.append((j, i, 1.0))
            n_games[i] += 1
            n_games[j] += 1
        elif r.result == "DRAW":
            if DRAW_MODE == "ignore":
                continue
            # half: iがjに0.5勝、jがiに0.5勝
            obs.append((i, j, 0.5))
            obs.append((j, i, 0.5))
            n_games[i] += 1
            n_games[j] += 1

    return obs, id_to_idx, id_list, n_games


def fit_bradley_terry(n_items: int, obs: List[Tuple[int, int, float]]) -> np.ndarray:
    """
    theta を最尤推定する。
    識別制約：thetaの平均=0 に正規化（最適化後に中心化）
    """
    if n_items <= 1:
        return np.zeros((n_items,), dtype=float)

    # 初期値
    x0 = np.zeros((n_items,), dtype=float)

    def nll(theta: np.ndarray) -> float:
        # 安定化のため中心化して使う
        t = theta - theta.mean()

        # 負の対数尤度
        total = 0.0
        for w, l, weight in obs:
            # P(w beats l) = sigmoid(t[w]-t[l])
            d = t[w] - t[l]
            # log(sigmoid(d)) を安定計算
            # -log(sigmoid(d)) = log(1+exp(-d))
            total += weight * np.log1p(np.exp(-d))
        return float(total)

    def grad(theta: np.ndarray) -> np.ndarray:
        t = theta - theta.mean()
        g = np.zeros_like(t)

        for w, l, weight in obs:
            d = t[w] - t[l]
            # sigmoid(-d) = 1/(1+exp(d))
            s = 1.0 / (1.0 + np.exp(d))
            # nll += weight * log(1+exp(-d))
            # d/dt[w] = -weight * sigmoid(-d)
            # d/dt[l] = +weight * sigmoid(-d)
            g[w] += -weight * s
            g[l] += +weight * s

        # 中心化の影響を打ち消す（平均0制約に相当）
        g -= g.mean()
        return g

    res = minimize(nll, x0, jac=grad, method="L-BFGS-B")
    theta = res.x
    theta = theta - theta.mean()
    return theta


def upsert_ratings(
    conn: psycopg.Connection,
    as_of_iso: str,
    scope: str,
    game_title_id: Optional[str],
    format_id: Optional[str],
    archtype_ids: List[str],
    theta: np.ndarray,
    n_games: Dict[int, int],
) -> None:
    """
    bt_archtype_ratings に upsert（最新値で上書き）
    """
    # rating = exp(theta)
    rating = np.exp(theta)

    sql = """
    insert into bt_archtype_ratings
      (as_of, game_title_id, format_id, scope, archtype_id, theta, rating, n_games, updated_datetime)
    values
      (%(as_of)s, %(game_title_id)s, %(format_id)s, %(scope)s, %(archtype_id)s, %(theta)s, %(rating)s, %(n_games)s, now())
    on conflict (scope, game_title_id, format_id, archtype_id)
    do update set
      as_of = excluded.as_of,
      theta = excluded.theta,
      rating = excluded.rating,
      n_games = excluded.n_games,
      updated_datetime = now()
    """

    with conn.cursor() as cur:
        for idx, arch_id in enumerate(archtype_ids):
            cur.execute(sql, {
                "as_of": as_of_iso,
                "game_title_id": game_title_id,
                "format_id": format_id,
                "scope": scope,
                "archtype_id": arch_id,
                "theta": float(theta[idx]),
                "rating": float(rating[idx]),
                "n_games": int(n_games.get(idx, 0)),
            })


def main() -> None:
    as_of_iso = os.environ.get("BT_AS_OF")  # 例: 2025-12-18T00:00:00Z
    if not as_of_iso:
        # Actions ならUTC nowで十分
        from datetime import datetime, timezone
        as_of_iso = datetime.now(timezone.utc).isoformat()

    with psycopg.connect(DATABASE_URL) as conn:
        rows, game_title_id, format_id = fetch_matches(conn)
        if len(rows) == 0:
            print("No matches found. Nothing to estimate.")
            return

        obs, id_to_idx, id_list, n_games = build_pairs(rows)
        if len(obs) == 0:
            print("No usable observations (maybe all DRAW ignored).")
            return

        theta = fit_bradley_terry(len(id_list), obs)

        # 書き込み
        upsert_ratings(
            conn=conn,
            as_of_iso=as_of_iso,
            scope=SCOPE,
            game_title_id=game_title_id,
            format_id=format_id,
            archtype_ids=id_list,
            theta=theta,
            n_games=n_games,
        )
        conn.commit()

        # ざっくり上位表示（ログ）
        order = np.argsort(-np.exp(theta))
        print("Top 10 ratings:")
        for k in order[:10]:
            print(f"{id_list[k]} rating={math.exp(theta[k]):.4f} theta={theta[k]:+.4f} n_games={n_games.get(int(k),0)}")


if __name__ == "__main__":
    main()
