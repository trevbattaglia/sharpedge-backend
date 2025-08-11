from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, Body, Query
from fastapi.responses import JSONResponse
import datetime as dt
import httpx
from functools import lru_cache
import time
import os
from collections import defaultdict

# ------------------------------
# Config / constants
# ------------------------------
app = FastAPI(title="SharpEdge Actions", version="1.0.0")

STATSAPI_BASE = "https://statsapi.mlb.com/api"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Bookmaker display mapping (Odds API key -> our short code)
BOOK_NAME_MAP = {
    "draftkings": "dk",
    "fanduel": "fd",
    "betmgm": "mgm",
    "caesars": "czu",
    "espnbet": "espn",
    # add more as needed
}

# For chunking bookmaker filters (keeps URLs sane)
BOOK_CHUNK_SIZE = 8

# Global thresholds / caps
EDGE_THRESHOLD_DEFAULT = 0.015  # 1.5%
MAX_KELLY = 0.05                # cap ½‑Kelly at 5%

# Lineup TTL
_BOX_TTL_SEC = 60
_BOX_CACHE: Dict[int, Tuple[float, dict]] = {}

# ------------------------------
# Odds math helpers
# ------------------------------
def american_to_implied(odds: int) -> float:
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return (-odds) / ((-odds) + 100.0)

def decimal_from_american(odds: int) -> float:
    return 1.0 + (odds / 100.0) if odds > 0 else 1.0 + (100.0 / (-odds))

def devig_two_way(p1: float, p2: float) -> Tuple[float, float]:
    s = p1 + p2
    return (p1 / s, p2 / s) if s > 0 else (0.0, 0.0)

def ev_per_dollar(q: float, odds: int) -> float:
    b = decimal_from_american(odds) - 1.0
    return q * b - (1.0 - q)

def kelly_half(q: float, odds: int) -> float:
    b = decimal_from_american(odds) - 1.0
    if b <= 0:
        return 0.0
    k = (b * q - (1.0 - q)) / b
    return max(0.0, min(k / 2.0, MAX_KELLY))

# ------------------------------
# Health / version
# ------------------------------
@app.get("/health")
def health():
    return {"ok": True, "service": "sharpedge-backend"}

@app.get("/version")
def version():
    return {"version": "1.0.0", "endpoints": [
        "/odds", "/mlb/stats", "/mlb/savant",
        "/fangraphs/projections", "/model/probability",
        "/rank", "/news/consensus", "/build_batch", "/picks"
    ]}

# ------------------------------
# MLB Stats (probables + lineups w/ TTL)
# ------------------------------
def _iso_date(d: Optional[str]) -> str:
    return d or dt.datetime.utcnow().date().isoformat()

def _mk_game_id(away_abbr: str, home_abbr: str, date_iso: str) -> str:
    return f"mlb-{away_abbr.lower()}-{home_abbr.lower()}-{date_iso}"

def _lineup_from_boxscore(box: dict) -> Tuple[bool, List[dict], List[dict]]:
    """
    Returns: (confirmed, away_lineup, home_lineup)
    Each lineup row: {id, name, pos, order}
    """
    try:
        away_players = box["teams"]["away"]["players"]
        home_players = box["teams"]["home"]["players"]

        def starters(players: dict) -> List[dict]:
            rows = []
            for p in players.values():
                order = p.get("battingOrder")
                if not order:
                    continue
                try:
                    order_i = int(order)
                except Exception:
                    continue
                person = p.get("person", {})
                pos = (p.get("position") or {}).get("abbreviation") or ""
                rows.append({
                    "id": person.get("id"),
                    "name": person.get("fullName"),
                    "pos": pos,
                    "order": order_i
                })
            rows.sort(key=lambda r: r["order"])
            return rows[:9]

        away9 = starters(away_players)
        home9 = starters(home_players)
        confirmed = len(away9) >= 9 and len(home9) >= 9
        return confirmed, away9, home9
    except Exception:
        return False, [], []

@lru_cache(maxsize=16)
def _fetch_schedule(date_iso: str) -> dict:
    url = f"{STATSAPI_BASE}/v1/schedule"
    params = {
        "sportId": 1,
        "date": date_iso,
        "hydrate": "probablePitcher,venue"
    }
    with httpx.Client(timeout=15) as client:
        r = client.get(url, params=params)
        r.raise_for_status()
        return r.json()

def _fetch_boxscore(game_pk: int) -> dict:
    now = time.time()
    cached = _BOX_CACHE.get(game_pk)
    if cached and (now - cached[0]) < _BOX_TTL_SEC:
        return cached[1]
    url = f"{STATSAPI_BASE}/v1/game/{game_pk}/boxscore"
    with httpx.Client(timeout=15) as client:
        r = client.get(url)
        r.raise_for_status()
        data = r.json()
    _BOX_CACHE[game_pk] = (now, data)
    return data

@app.get("/mlb/stats")
def mlb_stats(date: Optional[str] = None):
    """
    schedule?sportId=1&date=YYYY-MM-DD&hydrate=probablePitcher,venue
    + per-game boxscore (60s TTL) to infer lineup + extract player IDs
    """
    date_iso = _iso_date(date)
    try:
        sched = _fetch_schedule(date_iso)
    except Exception as e:
        return JSONResponse({"date": date_iso, "games": [], "error": f"schedule fetch failed: {e}"}, status_code=200)

    out_games: List[Dict[str, Any]] = []
    for d in sched.get("dates", []):
        for g in d.get("games", []):
            try:
                game_pk = g["gamePk"]
                away_team = g["teams"]["away"]["team"]
                home_team = g["teams"]["home"]["team"]
                away_abbr = away_team.get("abbreviation") or away_team.get("teamName", "")[:3].upper()
                home_abbr = home_team.get("abbreviation") or home_team.get("teamName", "")[:3].upper()

                away_prob = g["teams"]["away"].get("probablePitcher") or {}
                home_prob = g["teams"]["home"].get("probablePitcher") or {}
                away_pid = away_prob.get("id")
                home_pid = home_prob.get("id")
                away_pname = away_prob.get("fullName") or away_prob.get("boxscoreName") or away_prob.get("lastFirstName")
                home_pname = home_prob.get("fullName") or home_prob.get("boxscoreName") or home_prob.get("lastFirstName")

                venue = (g.get("venue") or {}).get("name") or ""

                confirmed = False
                lineup_away: List[dict] = []
                lineup_home: List[dict] = []
                try:
                    box = _fetch_boxscore(game_pk)
                    confirmed, lineup_away, lineup_home = _lineup_from_boxscore(box)
                except Exception:
                    confirmed = False

                out_games.append({
                    "game_id": _mk_game_id(away_abbr, home_abbr, date_iso),
                    "game_pk": game_pk,
                    "away": away_abbr,
                    "home": home_abbr,
                    "probables": {
                        away_abbr: {"id": away_pid, "name": away_pname},
                        home_abbr: {"id": home_pid, "name": home_pname},
                    },
                    "park": venue,
                    "lineup_confirmed": confirmed,
                    "lineup": {
                        away_abbr: lineup_away,
                        home_abbr: lineup_home
                    }
                })
            except Exception:
                continue

    return JSONResponse({"date": date_iso, "games": out_games})

# ------------------------------
# Fangraphs / Savant stubs
# ------------------------------
@app.get("/mlb/savant")
def mlb_savant(player_id: str):
    return JSONResponse({
        "player_id": player_id,
        "pitch_mix": {"sinker": 42.0, "curve": 27.0, "change": 12.0, "cutter": 11.0, "four_seam": 8.0},
        "splits": {"vs_L": {"wOBA": 0.305}, "vs_R": {"wOBA": 0.320}}
    })

@app.get("/fangraphs/projections")
def fangraphs_projections(
    entity_id: Optional[str] = Query(None, alias="entity_id"),
    entityId: Optional[str] = Query(None, alias="entityId"),
    player_id: Optional[str] = Query(None, alias="player_id"),
    id_: Optional[str] = Query(None, alias="id"),
):
    eid = entity_id or entityId or player_id or id_ or "UNKNOWN"
    return JSONResponse({
        "entity_id": eid,
        "projections": {"K%": 24.5, "BB%": 7.1, "ERA": 3.65, "wOBA": 0.334, "TB_per_game": 1.75},
        "meta": {"note": "stub projections"}
    })

@app.post("/model/probability")
def model_probability(payload: Dict[str, Any] = Body(...)):
    q = {"HOU": 0.47, "NYY": 0.53}
    return JSONResponse({"q": q, "meta": {"note": "stubbed probs", "echo": payload}})

# ------------------------------
# The Odds API integration
# ------------------------------
def _oddsapi_sport_code(sport: str) -> str:
    s = sport.lower()
    if s == "mlb":
        return "baseball_mlb"
    raise ValueError("Only mlb supported in this demo")

def _normalize_team(name: str) -> str:
    tok = name.split()[-1].upper()
    return tok if 2 <= len(tok) <= 4 else name[:3].upper()

def _chunk(lst: List[str], size: int) -> List[List[str]]:
    return [lst[i:i+size] for i in range(0, len(lst), size)]

def _fetch_odds_oddsapi(sport: str, markets: List[str], books: List[str]) -> dict:
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY missing")

    sport_code = _oddsapi_sport_code(sport)
    want_ml = "ml" in (m.lower() for m in markets)
    want_total = "total" in (m.lower() for m in markets)

    # If books provided, map to provider names; otherwise None (provider returns all)
    bookmaker_param_list: Optional[List[str]] = None
    if books:
        rev = {v: k for k, v in BOOK_NAME_MAP.items()}
        bookmaker_param_list = [rev.get(b.lower(), b) for b in books]

    params_base = {
        "apiKey": api_key,
        "regions": "us",
        "oddsFormat": "american",
        "dateFormat": "iso",
        # "eventIds": "", # optional
    }

    out_games: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"markets": {}})

    def _ingest_games(gjson: List[dict], mode: str):
        for g in gjson:
            gid = f"mlb-{_normalize_team(g['away_team']).lower()}-{_normalize_team(g['home_team']).lower()}-{dt.datetime.utcnow().date().isoformat()}"
            away = _normalize_team(g["away_team"])
            home = _normalize_team(g["home_team"])
            node = out_games[gid]
            node["game_id"] = gid
            node["away"] = away
            node["home"] = home

            for bk in g.get("bookmakers", []):
                bname = BOOK_NAME_MAP.get(bk["key"], bk["key"])
                for m in bk.get("markets", []):
                    key = m.get("key")
                    if mode == "h2h" and key != "h2h":
                        continue
                    if mode == "totals" and key != "totals":
                        continue

                    if key == "h2h":
                        ml = node["markets"].setdefault("ml", defaultdict(dict))
                        for o in m.get("outcomes", []):
                            side = _normalize_team(o["name"])
                            price = int(o["price"])
                            ml[side][bname] = price

                    elif key == "totals":
                        tot = node["markets"].setdefault("total", {})
                        pts = None
                        over_price, under_price = None, None
                        for o in m.get("outcomes", []):
                            pts = o.get("point", pts)
                            nm = o["name"].lower()
                            if nm == "over":
                                over_price = int(o["price"])
                            elif nm == "under":
                                under_price = int(o["price"])
                        if pts is not None:
                            over_key = f"{pts}_over"
                            under_key = f"{pts}_under"
                            if over_price is not None:
                                tot.setdefault(over_key, {})[bname] = over_price
                            if under_price is not None:
                                tot.setdefault(under_key, {})[bname] = under_price

    with httpx.Client(timeout=20) as client:
        # h2h (moneyline)
        if want_ml:
            if bookmaker_param_list:
                for chunk in _chunk(bookmaker_param_list, BOOK_CHUNK_SIZE):
                    p = params_base | {"markets": "h2h", "bookmakers": ",".join(chunk)}
                    r = client.get(f"{ODDS_API_BASE}/sports/{sport_code}/odds", params=p)
                    try:
                        r.raise_for_status()
                        _ingest_games(r.json(), "h2h")
                    except Exception:
                        # keep going; totals might still work
                        pass
            else:
                p = params_base | {"markets": "h2h"}
                r = client.get(f"{ODDS_API_BASE}/sports/{sport_code}/odds", params=p)
                r.raise_for_status()
                _ingest_games(r.json(), "h2h")

        # totals
        if want_total:
            if bookmaker_param_list:
                for chunk in _chunk(bookmaker_param_list, BOOK_CHUNK_SIZE):
                    p = params_base | {"markets": "totals", "bookmakers": ",".join(chunk)}
                    r = client.get(f"{ODDS_API_BASE}/sports/{sport_code}/odds", params=p)
                    try:
                        r.raise_for_status()
                        _ingest_games(r.json(), "totals")
                    except Exception:
                        pass
            else:
                p = params_base | {"markets": "totals"}
                r = client.get(f"{ODDS_API_BASE}/sports/{sport_code}/odds", params=p)
                r.raise_for_status()
                _ingest_games(r.json(), "totals")

    games = []
    for gid, v in out_games.items():
        games.append({
            "game_id": v["game_id"],
            "away": v.get("away"),
            "home": v.get("home"),
            "markets": v["markets"],
        })
    return {"sport": sport, "markets": markets, "books": books or [], "games": games}

# Public /odds endpoint with graceful fallback
@app.get("/odds")
def get_odds(sport: str, markets: Optional[List[str]] = None, books: Optional[List[str]] = None):
    markets = markets or ["ml", "total"]
    books = books or []
    try:
        data = _fetch_odds_oddsapi(sport, markets, books)
        # If provider returned nothing at all, fall back once
        if not data.get("games"):
            raise RuntimeError("provider returned 0 games")
        return JSONResponse(data)
    except Exception as e:
        # Fallback stub (keeps UI/actions usable)
        data = {
            "sport": sport,
            "markets": markets,
            "books": books or ["dk", "mgm", "czu"],
            "fallback_reason": str(e),
            "games": [
                {
                    "game_id": f"mlb-nyy-hou-{dt.datetime.utcnow().date().isoformat()}",
                    "away": "HOU",
                    "home": "NYY",
                    "markets": {
                        "ml": {
                            "HOU": {"dk": 110, "mgm": 112, "czu": 115},
                            "NYY": {"dk": -130, "mgm": -128, "czu": -125},
                        },
                        "total": {
                            "8.5_over": {"dk": -105, "mgm": -110},
                            "8.5_under": {"dk": -115, "mgm": -110},
                        },
                    },
                }
            ],
        }
        return JSONResponse(data)

# ------------------------------
# Batch builder (from odds)
# ------------------------------
def best_price_line(bookmap: Dict[str, int]) -> Tuple[str, int]:
    best_book, best_odds = None, None
    for book, price in bookmap.items():
        cur = int(price)
        if best_odds is None:
            best_book, best_odds = book, cur
            continue
        if (cur > 0 and cur > best_odds) or (cur < 0 and best_odds < 0 and cur > best_odds) or (cur < 0 and best_odds > 0):
            best_book, best_odds = book, cur
    return best_book or "consensus", best_odds or 0

@app.get("/build_batch")
def build_batch(sport: str = "mlb"):
    odds_resp = get_odds(sport=sport, markets=["ml", "total"]).body
    import json
    odds = json.loads(odds_resp.decode("utf-8"))

    items: List[Dict[str, Any]] = []
    for g in odds.get("games", []):
        gid = g["game_id"]
        mkts = g.get("markets", {})
        ml = mkts.get("ml")
        if isinstance(ml, dict):
            away, home = g.get("away"), g.get("home")
            if away in ml and isinstance(ml[away], dict) and ml[away]:
                _, best = best_price_line(ml[away])
                items.append({
                    "market": "ml",
                    "ref_id": gid,
                    "sides": {away: best, home: -125},
                    "model_q": {away: 0.53, home: 0.47}
                })
            if home in ml and isinstance(ml[home], dict) and ml[home]:
                _, best = best_price_line(ml[home])
                items.append({
                    "market": "ml",
                    "ref_id": gid,
                    "sides": {home: best, away: -125},
                    "model_q": {home: 0.53, away: 0.47}
                })

        tot = mkts.get("total")
        if isinstance(tot, dict) and tot:
            pts_key = next((k.split("_")[0] for k in tot if k.endswith("_over")), None)
            if pts_key and f"{pts_key}_under" in tot:
                _, over_best = best_price_line(tot[f"{pts_key}_over"])
                _, under_best = best_price_line(tot[f"{pts_key}_under"])
                ref = f"{gid}:{pts_key}"
                items.append({
                    "market": "total",
                    "ref_id": ref,
                    "sides": {"Over": over_best, "Under": under_best},
                    "model_q": {"Over": 0.49, "Under": 0.51}
                })

    return JSONResponse({"items": items, "limits": {"ml": 5, "total": 5, "prop": 8}})

# ------------------------------
# Ranking (model or batch)
# ------------------------------
def _two_way_card(market: str, ref_id: str, sides: Dict[str, int], model_q: Dict[str, float], note: str = "") -> List[Dict[str, Any]]:
    keys = list(sides.keys())
    if len(keys) != 2:
        return []
    a, b = keys[0], keys[1]
    p_a = american_to_implied(int(sides[a]))
    p_b = american_to_implied(int(sides[b]))
    fair_a, fair_b = devig_two_way(p_a, p_b)

    rows = []
    for team, fair in [(a, fair_a), (b, fair_b)]:
        q = float(model_q.get(team, fair))
        odds = int(sides[team])
        edge = q - fair
        ev = ev_per_dollar(q, odds)
        k = kelly_half(q, odds)
        rows.append({
            "Market": market.upper(),
            "Ref": ref_id,
            "Side": team,
            "Line": odds,
            "Fair%": round(fair * 100.0, 1),
            "Model%": round(q * 100.0, 1),
            "Edge": round(edge * 100.0, 1),
            "EV_per_$": round(ev, 4),
            "Kelly_half": round(k, 3),
            "Notes": note or "calc"
        })
    return rows

@app.post("/rank")
def rank(payload: Dict[str, Any] = Body(...)):
    """
    Accepts EITHER a single item OR a batch:
      { "market":"ml","game_id":"...","sides":{...},"model_q":{...} }
    OR
      { "min_edge":0.0, "items":[...], "limits":{"ml":5,"total":5,"prop":8} }
    """
    min_edge = float(payload.get("min_edge", EDGE_THRESHOLD_DEFAULT))

    if "items" in payload:
        items = payload["items"]
        limits = payload.get("limits", {"ml": 5, "total": 5, "prop": 8})
    else:
        items = [{
            "market": payload.get("market", "ml"),
            "ref_id": payload.get("game_id") or payload.get("ref_id", "UNKNOWN"),
            "sides": payload.get("sides", {}),
            "model_q": payload.get("model_q", {})
        }]
        limits = {"ml": 5, "total": 5, "prop": 8}

    all_cards: List[Dict[str, Any]] = []
    for it in items:
        market = str(it.get("market", "ml")).lower()
        ref_id = it.get("ref_id") or it.get("game_id") or "UNKNOWN"
        sides = it.get("sides", {})
        model_q = it.get("model_q", {})
        cards = _two_way_card(market, ref_id, sides, model_q)
        cards = [c for c in cards if c.get("Edge", 0) >= min_edge * 100.0]
        all_cards.extend(cards)

    def topn(mkt: str, n: int) -> List[Dict[str, Any]]:
        rows = [c for c in all_cards if c["Market"].lower() == mkt]
        rows.sort(key=lambda r: (r["EV_per_$"], r["Edge"], r["Kelly_half"]), reverse=True)
        return rows[:n]

    top_ml     = topn("ml",     limits.get("ml", 5))
    top_totals = topn("total",  limits.get("total", 5))
    top_props  = topn("prop",   limits.get("prop", 8))

    if "items" not in payload:
        single = sorted(all_cards, key=lambda r: r["EV_per_$"], reverse=True)
        return JSONResponse({"cards": single})

    return JSONResponse({
        "top_ml": top_ml,
        "top_totals": top_totals,
        "top_props": top_props,
        "filters": {"edge_threshold": min_edge, "kelly_cap": MAX_KELLY, "limits": limits}
    })

# ------------------------------
# Consensus +EV scan (no external model)
# ------------------------------
def _devig_two_way_from_book(side_a_odds: int, side_b_odds: int) -> Tuple[float, float]:
    pa = american_to_implied(side_a_odds)
    pb = american_to_implied(side_b_odds)
    return devig_two_way(pa, pb)

def _median(xs: List[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    mid = n // 2
    return ys[mid] if n % 2 else 0.5 * (ys[mid - 1] + ys[mid])

def _consensus_fair_from_books(two_way_prices: List[Tuple[int, int]]) -> Tuple[float, float]:
    fair_a_list, fair_b_list = [], []
    for a, b in two_way_prices:
        fa, fb = _devig_two_way_from_book(a, b)
        fair_a_list.append(fa)
        fair_b_list.append(fb)
    return _median(fair_a_list), _median(fair_b_list)

def _collect_two_way_pairs(ml_market: Dict[str, Dict[str, int]], a_key: str, b_key: str) -> List[Tuple[int, int]]:
    pairs = []
    books = set(ml_market.get(a_key, {})) & set(ml_market.get(b_key, {}))
    for bk in books:
        try:
            a = int(ml_market[a_key][bk])
            b = int(ml_market[b_key][bk])
            pairs.append((a, b))
        except Exception:
            continue
    return pairs

def _pairs_for_total(total_market: Dict[str, Dict[str, int]], pts_key: str) -> List[Tuple[int, int]]:
    over_key = f"{pts_key}_over"
    under_key = f"{pts_key}_under"
    if over_key not in total_market or under_key not in total_market:
        return []
    books = set(total_market[over_key]) & set(total_market[under_key])
    pairs = []
    for bk in books:
        try:
            o = int(total_market[over_key][bk])
            u = int(total_market[under_key][bk])
            pairs.append((o, u))
        except Exception:
            continue
    return pairs

@app.get("/picks")
def picks(
    sport: str = "mlb",
    min_ev: float = 0.0,   # per $1 (e.g., 0.01 = +1 cent EV / $1)
    min_edge: float = 0.0, # decimal (0.015 = 1.5%)
    limit: int = 15
):
    """
    Find +EV opportunities vs cross-book consensus (no external model).
    Works for ML and a single totals number per game if available.
    """
    odds_resp = get_odds(sport=sport, markets=["ml", "total"]).body
    import json
    data = json.loads(odds_resp.decode("utf-8"))

    candidates: List[Dict[str, Any]] = []

    for g in data.get("games", []):
        gid = g["game_id"]
        away, home = g.get("away"), g.get("home")
        mkts = g.get("markets", {})

        # Moneylines
        ml = mkts.get("ml", {})
        if isinstance(ml, dict) and away in ml and home in ml and ml[away] and ml[home]:
            pairs = _collect_two_way_pairs(ml, away, home)
            if pairs:
                fair_away, fair_home = _consensus_fair_from_books(pairs)
                # Away offers
                for bk, price in ml.get(away, {}).items():
                    odds = int(price)
                    q = fair_away
                    ev = ev_per_dollar(q, odds)
                    edge = q - american_to_implied(odds)
                    if ev >= min_ev and edge >= min_edge:
                        candidates.append({
                            "Market": "ML",
                            "Game": gid,
                            "Side": away,
                            "Book": bk,
                            "Line": odds,
                            "ConsensusFair%": round(q*100, 1),
                            "EV_per_$": round(ev, 4),
                            "Kelly_half": round(kelly_half(q, odds), 3),
                        })
                # Home offers
                for bk, price in ml.get(home, {}).items():
                    odds = int(price)
                    q = fair_home
                    ev = ev_per_dollar(q, odds)
                    edge = q - american_to_implied(odds)
                    if ev >= min_ev and edge >= min_edge:
                        candidates.append({
                            "Market": "ML",
                            "Game": gid,
                            "Side": home,
                            "Book": bk,
                            "Line": odds,
                            "ConsensusFair%": round(q*100, 1),
                            "EV_per_$": round(ev, 4),
                            "Kelly_half": round(kelly_half(q, odds), 3),
                        })

        # Totals (first available points bucket)
        tot = mkts.get("total")
        if isinstance(tot, dict) and tot:
            pts_keys = sorted({k.split("_")[0] for k in tot.keys() if "_over" in k})
            if pts_keys:
                pts = pts_keys[0]
                pairs = _pairs_for_total(tot, pts)
                if pairs:
                    fair_over, fair_under = _consensus_fair_from_books(pairs)
                    over_key, under_key = f"{pts}_over", f"{pts}_under"

                    for bk, price in tot.get(over_key, {}).items():
                        odds = int(price)
                        q = fair_over
                        ev = ev_per_dollar(q, odds)
                        edge = q - american_to_implied(odds)
                        if ev >= min_ev and edge >= min_edge:
                            candidates.append({
                                "Market": f"Total {pts}",
                                "Game": gid,
                                "Side": "Over",
                                "Book": bk,
                                "Line": odds,
                                "ConsensusFair%": round(q*100, 1),
                                "EV_per_$": round(ev, 4),
                                "Kelly_half": round(kelly_half(q, odds), 3),
                            })

                    for bk, price in tot.get(under_key, {}).items():
                        odds = int(price)
                        q = fair_under
                        ev = ev_per_dollar(q, odds)
                        edge = q - american_to_implied(odds)
                        if ev >= min_ev and edge >= min_edge:
                            candidates.append({
                                "Market": f"Total {pts}",
                                "Game": gid,
                                "Side": "Under",
                                "Book": bk,
                                "Line": odds,
                                "ConsensusFair%": round(q*100, 1),
                                "EV_per_$": round(ev, 4),
                                "Kelly_half": round(kelly_half(q, odds), 3),
                            })

    candidates.sort(key=lambda r: (r["EV_per_$"], r["Kelly_half"]), reverse=True)
    return JSONResponse({"picks": candidates[:max(1, min(limit, 50))], "count": len(candidates)})

# ------------------------------
# News stub
# ------------------------------
@app.get("/news/consensus")
def news(sport: str):
    return JSONResponse({"sport": sport, "consensus": [], "injuries": [], "links": []})
