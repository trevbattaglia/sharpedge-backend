from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from fastapi import Query


app = FastAPI(title="SharpEdge Actions", version="1.0.0")

# ---------- Helpers (math) ----------
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
    k = (b * q - (1.0 - q)) / b if b > 0 else 0.0
    return max(0.0, k / 2.0)

# ---------- Health ----------
@app.get("/health")
def health():
    return {"ok": True, "service": "sharpedge-backend"}

@app.get("/version")
def version():
    return {"version": "1.0.0", "endpoints": [
        "/odds", "/mlb/stats", "/mlb/savant",
        "/fangraphs/projections", "/model/probability",
        "/rank", "/news/consensus"
    ]}

# ---------- Endpoints (stubs) ----------
@app.get("/odds")
def get_odds(sport: str, markets: Optional[List[str]] = None, books: Optional[List[str]] = None):
    """Stub multi-book odds for one MLB game."""
    data = {
        "sport": sport,
        "markets": markets or ["ml", "total"],
        "books": books or ["dk", "mgm", "czu"],
        "games": [
            {
                "game_id": "mlb-nyy-hou-2025-08-11",
                "away": "HOU",
                "home": "NYY",
                "markets": {
                    "ml": {
                        "HOU": {"dk": 110, "mgm": 112, "czu": 115},
                        "NYY": {"dk": -130, "mgm": -128, "czu": -125}
                    },
                    "total": {
                        "8.5_over": {"dk": -105, "mgm": -110},
                        "8.5_under": {"dk": -115, "mgm": -110}
                    }
                }
            }
        ]
    }
    return JSONResponse(data)

@app.get("/mlb/stats")
def mlb_stats(date: Optional[str] = None):
    """Stub MLB StatsAPI-like response; always returns 200."""
    return JSONResponse({
        "date": date or "unknown",
        "games": [
            {
                "game_id": "mlb-nyy-hou-2025-08-11",
                "probables": {"HOU": "Framber Valdez", "NYY": "Carlos Rodon"},
                "bullpen_rest_index": {"HOU": 0.62, "NYY": 0.48},
                "park": "Yankee Stadium",
                "lineup_confirmed": False
            }
        ]
    })

@app.get("/mlb/savant")
def mlb_savant(player_id: str):
    """Stub Savant splits & pitch mix; always returns 200 for any player_id."""
    return JSONResponse({
        "player_id": player_id,
        "pitch_mix": {"sinker": 42.0, "curve": 27.0, "change": 12.0, "cutter": 11.0, "four_seam": 8.0},
        "splits": {"vs_L": {"wOBA": 0.305}, "vs_R": {"wOBA": 0.320}}
    })


@app.get("/fangraphs/projections")
def fangraphs_projections(
    # accept multiple aliases so the Action can't miss
    entity_id: Optional[str] = Query(None, alias="entity_id"),
    entityId: Optional[str] = Query(None, alias="entityId"),
    player_id: Optional[str] = Query(None, alias="player_id"),
    id_: Optional[str] = Query(None, alias="id"),
):
    # choose the first non-empty id provided
    eid = entity_id or entityId or player_id or id_ or "UNKNOWN"

    return JSONResponse({
        "entity_id": eid,
        "projections": {
            "K%": 24.5,
            "BB%": 7.1,
            "ERA": 3.65,
            "wOBA": 0.334,
            "TB_per_game": 1.75
        },
        "meta": {"note": "stub projections"}
    })

@app.post("/model/probability")
def model_probability(payload: Dict[str, Any] = Body(...)):
    """Returns a sample model win prob for a two‑way ML."""
    q = {"HOU": 0.47, "NYY": 0.53}
    return JSONResponse({"q": q, "meta": {"note": "stubbed probs", "echo": payload}})

@app.post("/rank")
def rank(payload: Dict[str, Any] = Body(...)):
    """
    Example payload:
    {
      "market":"ml",
      "game_id":"mlb-nyy-hou-2025-08-11",
      "sides":{"HOU":115,"NYY":-125},
      "model_q":{"HOU":0.47,"NYY":0.53}
    }
    """
    market = payload.get("market", "ml")
    sides = payload.get("sides", {})
    model_q = payload.get("model_q", {})
    keys = list(sides.keys())
    if len(keys) != 2:
        return JSONResponse({"error": "only two-way example in stub", "received": sides}, status_code=400)

    a, b = keys[0], keys[1]
    p_a = american_to_implied(int(sides[a]))
    p_b = american_to_implied(int(sides[b]))
    fair_a, fair_b = devig_two_way(p_a, p_b)

    rows = []
    for team, fair in [(a, fair_a), (b, fair_b)]:
        q = float(model_q.get(team, fair))  # fallback q=fair
        odds = int(sides[team])
        edge = q - fair
        ev = ev_per_dollar(q, odds)
        k = kelly_half(q, odds)
        rows.append({
            "Market": market.upper(),
            "Side": team,
            "Line": odds,
            "Fair%": round(fair * 100.0, 1),
            "Model%": round(q * 100.0, 1),
            "Edge": round(edge * 100.0, 1),
            "EV_per_$": round(ev, 4),
            "Kelly_half": round(k, 3),
            "Notes": "stub calc"
        })
    rows.sort(key=lambda r: r["EV_per_$"], reverse=True)
    return JSONResponse({"cards": rows})

@app.get("/news/consensus")
def news(sport: str):
    """Stub news/consensus response."""
    return JSONResponse({"sport": sport, "consensus": [], "injuries": [], "links": []})
