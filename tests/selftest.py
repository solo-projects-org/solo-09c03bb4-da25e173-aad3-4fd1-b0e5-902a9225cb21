#!/usr/bin/env python3
"""End-to-end self test for the ensemble rainfall probability API.

By default the script spawns TWO uvicorn processes sharing one SQLite file
(dual-instance deployment), waits for health, and exercises every key flow:

  * registration / member-count / threshold / datetime validation
  * idempotent resubmit (same revision+content) and conflicting resubmit rejection
  * revision ordering, grid-shape and non-negativity checks
  * publish blocked until every member is present
  * probability / denominator / all-missing null semantics and used revisions
  * correction immediately expires the old product while history stays queryable
  * two-instance interleaved publish vs. correction -> only consistent products
  * parallel duplicate publishes collapse to one idempotent product
  * injected compute failure leaves no half-written product (atomicity)
  * slow computation never blocks another forecast time's ingestion
  * quality control: tolerance gate, anomaly counting, pending lock, review
  * publish/approve/reject interleavings across both instances
  * failed publish never leaves a half product nor revives an expired one
  * Swagger UI / OpenAPI availability

Only the Python standard library is required.
"""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PASS_COUNT = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS_COUNT
    if not cond:
        raise AssertionError(f"FAIL: {name} {detail}")
    PASS_COUNT += 1
    print(f"  PASS  {name}")


def http(method: str, base: str, path: str, body: Optional[dict] = None) -> Tuple[int, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"raw": raw}


def approx(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def matrix_approx(a: list, b: list) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if len(ra) != len(rb):
            return False
        for x, y in zip(ra, rb):
            if not approx(x, y):
                return False
    return True


# --------------------------------------------------------------------------- #
# dual-instance process management
# --------------------------------------------------------------------------- #
def spawn_servers() -> Tuple[list, str, str, str, str, str]:
    tmp = tempfile.mkdtemp(prefix="rain-selftest-")
    db_path = os.path.join(tmp, "rain.db")
    fail_marker = os.path.join(tmp, "fail.marker")
    env = dict(os.environ)
    env.update(
        RAIN_DB_PATH=db_path,
        RAIN_TEST_PUBLISH_DELAY_S="0.6",
        RAIN_TEST_FAIL_MARKER=fail_marker,
        PYTHONPATH=ROOT,
    )
    procs = []
    for port in (8091, 8092):
        log = open(os.path.join(tmp, f"uvicorn-{port}.log"), "w")
        procs.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "app.main:app",
                 "--host", "127.0.0.1", "--port", str(port)],
                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
        )
    base_a, base_b = "http://127.0.0.1:8091", "http://127.0.0.1:8092"
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            s1, _ = http("GET", base_a, "/health")
            s2, _ = http("GET", base_b, "/health")
            if s1 == 200 and s2 == 200:
                return procs, base_a, base_b, fail_marker, tmp, db_path
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("servers did not become healthy")


def shutdown(procs: list) -> None:
    for p in procs:
        p.send_signal(signal.SIGINT)
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()


def main() -> int:
    base_a = os.environ.get("RAIN_BASE_A")
    base_b = os.environ.get("RAIN_BASE_B")
    fail_marker = os.environ.get("RAIN_TEST_FAIL_MARKER", "")
    db_path = os.environ.get("RAIN_TEST_DB_PATH", "")
    spawned = False
    procs: list = []
    if not base_a or not base_b:
        procs, base_a, base_b, fail_marker, _, db_path = spawn_servers()
        spawned = True
        print(f"[setup] spawned dual instances {base_a} / {base_b}")
    hooks_enabled = bool(fail_marker)
    try:
        run(base_a, base_b, fail_marker, hooks_enabled, db_path)
    finally:
        if spawned:
            shutdown(procs)
    print(f"\nALL TESTS PASSED ({PASS_COUNT} assertions)")
    return 0


def register(base: str, members: int = 3, rows: int = 2, cols: int = 2,
             thresholds=(5.0, 10.0), names=None, tolerance=None) -> dict:
    if names is None:
        names = [f"m{i}" for i in range(1, members + 1)]
    body = {
        "forecast_time": "2026-11-16T00:00:00Z",
        "rows": rows, "cols": cols,
        "member_names": names,
        "thresholds": list(thresholds),
    }
    if tolerance is not None:
        body["change_tolerance"] = tolerance
    status, data = http("POST", base, "/groups", body)
    assert status == 201, (status, data)
    return data


def review(base: str, gid: str, member: str, revision: int, action: str) -> Tuple[int, Any]:
    return http("POST", base,
                f"/groups/{gid}/members/{member}/revisions/{revision}/review",
                {"action": action})


def submit(base: str, gid: str, member: str, rain: list, revision: int,
           missing: Optional[list] = None) -> Tuple[int, Any]:
    body: Dict[str, Any] = {"rain": rain, "revision": revision}
    if missing is not None:
        body["missing"] = missing
    return http("PUT", base, f"/groups/{gid}/members/{member}", body)


def db_scalar(db_path: str, sql: str, params: tuple = ()) -> Any:
    conn = sqlite3.connect(db_path, timeout=10)
    try:
        row = conn.execute(sql, params).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def run_qc_tests(base_a: str, base_b: str, fail_marker: str,
                 hooks_enabled: bool, db_path: str) -> None:
    print("\n[13] no tolerance configured -> original submit behaviour")
    g = register(base_a, members=2, rows=2, cols=2, thresholds=[5.0])
    gid = g["group_id"]
    check("register without tolerance echoes null", g["change_tolerance"] is None)
    s, d = submit(base_a, gid, "m1", [[100.0, 0.0], [0.0, 0.0]], 1)
    check("first submission adopted", s == 200 and d["status"] == "adopted", str(d))
    s, d = submit(base_b, gid, "m2", [[0.0, 0.0], [0.0, 0.0]], 1)
    check("m2 first submission adopted", s == 200 and d["status"] == "adopted")
    s, p0 = http("POST", base_a, f"/groups/{gid}/publish")
    check("baseline product published", s == 200, str(p0))
    s, d = submit(base_b, gid, "m1", [[40.0, 0.0], [0.0, 0.0]], 2)
    check("huge correction still adopted without tolerance",
          s == 200 and d["status"] == "adopted" and "anomaly_cells" not in d, str(d))
    s, d = http("GET", base_a, f"/groups/{gid}/products/current")
    check("untolerated correction expires product as before", s == 404)
    s, d = http("POST", base_a, "/groups", {
        "forecast_time": "2026-11-16T00:00:00Z", "rows": 1, "cols": 1,
        "member_names": ["a", "b"], "thresholds": [1.0], "change_tolerance": -0.1})
    check("negative tolerance rejected", s == 422, str(d))

    print("\n[14] tolerance gate: hold, count anomalies, do not touch product")
    gq = register(base_a, members=2, rows=2, cols=2, thresholds=[120.0], tolerance=2.0)
    gidq = gq["group_id"]
    check("tolerance echoed on registration", gq["change_tolerance"] == 2.0)
    # first submissions are always adopted even with large values
    s, d = submit(base_a, gidq, "m1", [[100.0, 0.0], [0.0, 0.0]], 1)
    check("first submission adopted despite tolerance",
          s == 200 and d["status"] == "adopted", str(d))
    s, _ = submit(base_b, gidq, "m2", [[0.0, 0.0], [0.0, 0.0]], 1)
    check("m2 first submission adopted", s == 200)
    s, p1 = http("POST", base_a, f"/groups/{gidq}/publish")
    check("product p1 published", s == 200 and p1["thresholds"][0]["probability"][0][0] == 0.0)

    s, d = submit(base_a, gidq, "m1", [[101.0, 0.0], [0.0, 0.0]], 2)
    check("correction within tolerance adopted", s == 200 and d["status"] == "adopted", str(d))
    s, _ = http("GET", base_b, f"/groups/{gidq}/products/current")
    check("adopted correction still expires current product", s == 404)
    s, p2 = http("POST", base_b, f"/groups/{gidq}/publish")
    check("p2 uses adopted rev2 and unchanged probability",
          s == 200 and p2["product_id"] != p1["product_id"]
          and p2["product_id"] == http("GET", base_a, f"/groups/{gidq}/products/current")[1]["product_id"]
          and p2["members_used"][0]["revision"] == 2, str(p2.get("members_used")))

    s, d = submit(base_b, gidq, "m1", [[150.0, 0.0], [0.0, 0.0]], 3)
    check("correction beyond tolerance held for review",
          s == 200 and d["status"] == "pending" and d["anomaly_cells"] == 1, str(d))
    s, cur = http("GET", base_a, f"/groups/{gidq}/products/current")
    check("pending revision does not expire current product",
          s == 200 and cur["product_id"] == p2["product_id"], str(cur))
    s, st = http("GET", base_b, f"/groups/{gidq}")
    m1 = [m for m in st["members"] if m["member_id"] == "m1"][0]
    check("status distinguishes adopted vs pending revision",
          m1["latest_revision"] == 2 and m1["latest_adopted_revision"] == 2
          and m1["pending_revision"]["revision"] == 3
          and m1["pending_revision"]["anomaly_cells"] == 1, str(m1))
    check("status flags pending review", st["has_pending_review"] is True)
    check("status still reports all members present", st["all_members_present"] is True)
    s, p3 = http("POST", base_a, f"/groups/{gidq}/publish")
    check("publish ignores pending revision and reuses current product",
          s == 200 and p3["product_id"] == p2["product_id"], str(p3["product_id"]))

    s, dup = submit(base_a, gidq, "m1", [[150.0, 0.0], [0.0, 0.0]], 3)
    check("same pending revision + same content is idempotent",
          s == 200 and dup["duplicate"] is True and dup["status"] == "pending"
          and dup["anomaly_cells"] == 1, str(dup))
    s, d = submit(base_b, gidq, "m1", [[151.0, 0.0], [0.0, 0.0]], 3)
    check("same pending revision + different content still rejected",
          s == 409 and d["error"]["code"] == "REVISION_CONTENT_CONFLICT", str(d))
    s, d = submit(base_a, gidq, "m1", [[152.0, 0.0], [0.0, 0.0]], 4)
    check("new revision while pending rejected",
          s == 409 and d["error"]["code"] == "PENDING_REVIEW", str(d))
    s, d = submit(base_a, gidq, "m1", [[100.0, 0.0], [0.0, 0.0]], 2)
    check("lower revision while pending rejected",
          s == 409 and d["error"]["code"] == "REVISION_CONFLICT", str(d))
    s, d = review(base_a, gidq, "m1", 99, "approve")
    check("review of unknown revision -> 404",
          s == 404 and d["error"]["code"] == "REVISION_NOT_FOUND", str(d))
    s, d = review(base_a, gidq, "m1", 2, "approve")
    check("review of adopted revision -> 409",
          s == 409 and d["error"]["code"] == "REVISION_NOT_PENDING", str(d))
    s, d = http("POST", base_a, f"/groups/{gidq}/members/m1/revisions/3/review",
                {"action": "maybe"})
    check("invalid review action rejected", s == 422, str(d))

    print("\n[15] approve adopts, expires old product, later publish uses it")
    s, rv = review(base_b, gidq, "m1", 3, "approve")
    check("approve succeeds", s == 200 and rv["outcome"] == "approved", str(rv))
    check("approve reports expired product", rv["expired_product_id"] == p2["product_id"], str(rv))
    check("member now has no pending revision", rv["member"]["pending"] is None
          and rv["member"]["latest_adopted_revision"] == 3)
    s, d = http("GET", base_a, f"/groups/{gidq}/products/current")
    check("approved revision expires current product", s == 404)
    s, p4 = http("POST", base_a, f"/groups/{gidq}/publish")
    by = p4["thresholds"][0]
    check("publish after approval uses approved revision",
          s == 200 and p4["members_used"][0]["revision"] == 3
          and matrix_approx(by["probability"], [[0.5, 0.0], [0.0, 0.0]]), str(by))
    s, d = review(base_a, gidq, "m1", 3, "reject")
    check("approve-then-reject of same revision rejected",
          s == 409 and d["error"]["code"] == "REVISION_NOT_PENDING", str(d))

    print("\n[16] reject keeps current product, burns the revision number")
    gr = register(base_b, members=2, rows=2, cols=2, thresholds=[5.0], tolerance=1.0)
    gidr = gr["group_id"]
    submit(base_a, gidr, "m1", [[10.0, 0.0], [0.0, 0.0]], 1)
    submit(base_b, gidr, "m2", [[0.0, 0.0], [0.0, 0.0]], 1)
    s, pr = http("POST", base_a, f"/groups/{gidr}/publish")
    check("reject-path baseline product", s == 200, str(pr))
    s, d = submit(base_b, gidr, "m1", [[99.0, 0.0], [0.0, 0.0]], 2)
    check("second anomalous correction held", s == 200 and d["anomaly_cells"] == 1, str(d))
    s, rv = review(base_a, gidr, "m1", 2, "reject")
    check("reject succeeds", s == 200 and rv["outcome"] == "rejected", str(rv))
    check("reject does not expire product", rv["expired_product_id"] is None)
    s, cur = http("GET", base_a, f"/groups/{gidr}/products/current")
    check("current product retained after reject",
          s == 200 and cur["product_id"] == pr["product_id"], str(cur))
    s, again = http("POST", base_b, f"/groups/{gidr}/publish")
    check("publish after reject reuses the same product",
          s == 200 and again["product_id"] == pr["product_id"])
    s, d = review(base_b, gidr, "m1", 2, "approve")
    check("late approve after reject is refused",
          s == 409 and d["error"]["code"] == "REVISION_REJECTED", str(d))
    s, d = submit(base_a, gidr, "m1", [[99.0, 0.0], [0.0, 0.0]], 2)
    check("resending rejected revision is refused",
          s == 409 and d["error"]["code"] == "REVISION_REJECTED", str(d))
    s, d = submit(base_a, gidr, "m1", [[10.0, 0.0], [0.0, 0.0]], 3)
    check("higher revision after reject is adopted (diff 0)",
          s == 200 and d["status"] == "adopted", str(d))
    s, d = http("GET", base_b, f"/groups/{gidr}/products/current")
    check("adopted higher revision expires product", s == 404)
    s, pr3 = http("POST", base_a, f"/groups/{gidr}/publish")
    check("publish uses revision 3", s == 200 and pr3["members_used"][0]["revision"] == 3)
    s, d = review(base_b, gidr, "m1", 2, "approve")
    check("late approve after newer revision cannot revive rejected version",
          s == 409 and d["error"]["code"] == "REVISION_REJECTED", str(d))
    s, cur = http("GET", base_a, f"/groups/{gidr}/products/current")
    check("current product still the revision-3 product",
          s == 200 and cur["product_id"] == pr3["product_id"]
          and cur["members_used"][0]["revision"] == 3, str(cur))

    print("\n[17] tolerance boundary and anomaly counting")
    gb = register(base_a, members=2, rows=2, cols=2, thresholds=[5.0], tolerance=2.0)
    gidb = gb["group_id"]
    submit(base_a, gidb, "m1", [[0.0, 0.0], [0.0, 0.0]], 1)
    submit(base_b, gidb, "m2", [[0.0, 0.0], [0.0, 0.0]], 1)
    s, d = submit(base_a, gidb, "m1", [[2.0, 0.0], [0.0, 0.0]], 2)
    check("change equal to tolerance is adopted", s == 200 and d["status"] == "adopted", str(d))
    s, d = submit(base_b, gidb, "m1", [[4.0, 0.0], [0.0, 0.0]], 3)
    check("change equal to tolerance vs latest adopted is adopted",
          s == 200 and d["status"] == "adopted", str(d))
    s, d = submit(base_a, gidb, "m1", [[6.0000001, 0.0], [0.0, 0.0]], 4)
    check("change just beyond tolerance is held",
          s == 200 and d["status"] == "pending" and d["anomaly_cells"] == 1, str(d))
    s, _ = review(base_b, gidb, "m1", 4, "reject")
    check("boundary pending rejected", s == 200)
    s, d = submit(base_a, gidb, "m1", [[7.0, 3.0], [3.0, 0.0]], 5)
    check("multi-cell anomaly count reported",
          s == 200 and d["status"] == "pending" and d["anomaly_cells"] == 3, str(d))
    s, _ = review(base_b, gidb, "m1", 5, "reject")
    check("multi-cell pending rejected", s == 200)
    s, d = submit(base_a, gidb, "m1", [[4.0, 0.0], [0.0, 0.0]], 6,
                  missing=[[1, 0], [0, 0]])
    check("valid->missing swap is held even inside tolerance",
          s == 200 and d["status"] == "pending" and d["anomaly_cells"] == 1, str(d))
    s, _ = review(base_b, gidb, "m1", 6, "reject")
    check("swap pending rejected", s == 200)
    s, d = submit(base_a, gidb, "m1", [[4.0, 0.0], [0.0, 0.0]], 7,
                  missing=[[0, 1], [0, 0]])
    check("missing->valid swap is held",
          s == 200 and d["status"] == "pending" and d["anomaly_cells"] == 1, str(d))
    s, _ = review(base_b, gidb, "m1", 7, "reject")
    check("missing->valid pending rejected", s == 200)
    s, d = submit(base_a, gidb, "m1", [[4.0, 0.0], [0.0, 0.0]], 8,
                  missing=[[0, 0], [0, 0]])
    check("rejected missing state does not become the comparison baseline",
          s == 200 and d["status"] == "adopted", str(d))

    print("\n[18] dual-instance interleaving: publish vs approve / reject")
    gi = register(base_a, members=2, rows=8, cols=8, thresholds=[5.0], tolerance=1.0)
    gidi = gi["group_id"]
    submit(base_a, gidi, "m1", [[10.0] * 8 for _ in range(8)], 1)
    submit(base_b, gidi, "m2", [[10.0] * 8 for _ in range(8)], 1)
    s, pi = http("POST", base_a, f"/groups/{gidi}/publish")
    check("interleave baseline published", s == 200, str(pi))
    submit(base_b, gidi, "m1", [[11.0] * 8 for _ in range(8)], 2)   # adopted
    s, d = submit(base_a, gidi, "m2", [[80.0] * 8 for _ in range(8)], 2)
    check("m2 anomalous revision held", s == 200 and d["status"] == "pending", str(d))

    pub: Dict[str, Any] = {}

    def pub_thread() -> None:
        pub["result"] = http("POST", base_b, f"/groups/{gidi}/publish")

    t = threading.Thread(target=pub_thread)
    t.start()
    time.sleep(0.15)
    s, rv = review(base_a, gidi, "m2", 2, "approve")
    check("approve lands while publish is computing", s == 200, str(rv))
    t.join()
    check("publish returns after retrying over the approval",
          pub["result"][0] == 200, str(pub["result"]))
    s, cur = http("GET", base_a, f"/groups/{gidi}/products/current")
    s2, st = http("GET", base_b, f"/groups/{gidi}")
    adopted = {m["member_id"]: m["latest_revision"] for m in st["members"]}
    check("current product uses the complete adopted revision set",
          s == 200 and {m["member_id"]: m["revision"] for m in cur["members_used"]} == adopted
          and adopted == {"m1": 2, "m2": 2}, f"{cur.get('members_used')} vs {adopted}")
    s, listing = http("GET", base_b, f"/groups/{gidi}/products")
    check("only one current product after approve/publish race",
          sum(1 for p in listing["products"] if p["is_current"]) == 1)
    check("no mixed product from the pre-approval attempt",
          all(p["revision_map"] != {"0": 2, "1": 1} for p in listing["products"]), str(listing))

    gj = register(base_b, members=2, rows=8, cols=8, thresholds=[5.0], tolerance=1.0)
    gidj = gj["group_id"]
    submit(base_b, gidj, "m1", [[10.0] * 8 for _ in range(8)], 1)
    submit(base_a, gidj, "m2", [[10.0] * 8 for _ in range(8)], 1)
    http("POST", base_b, f"/groups/{gidj}/publish")
    submit(base_a, gidj, "m1", [[11.0] * 8 for _ in range(8)], 2)
    s, d = submit(base_b, gidj, "m2", [[80.0] * 8 for _ in range(8)], 2)
    check("second group's m2 held", s == 200 and d["status"] == "pending", str(d))
    pub2: Dict[str, Any] = {}

    def pub2_thread() -> None:
        pub2["result"] = http("POST", base_a, f"/groups/{gidj}/publish")

    t = threading.Thread(target=pub2_thread)
    t.start()
    time.sleep(0.15)
    s, rv = review(base_b, gidj, "m2", 2, "reject")
    check("reject lands while publish is computing",
          s == 200 and rv["outcome"] == "rejected", str(rv))
    t.join()
    check("publish after concurrent reject succeeds", pub2["result"][0] == 200, str(pub2["result"]))
    s, cur = http("GET", base_b, f"/groups/{gidj}/products/current")
    check("rejected revision never enters a product",
          s == 200 and {m["member_id"]: m["revision"] for m in cur["members_used"]} == {"m1": 2, "m2": 1},
          str(cur.get("members_used")))
    s, listing = http("GET", base_a, f"/groups/{gidj}/products")
    check("no product references the rejected revision",
          all(p["revision_map"].get("1") != 2 for p in listing["products"]), str(listing))

    print("\n[19] concurrent approve vs reject: exactly one review wins")
    gc = register(base_a, members=2, rows=2, cols=2, thresholds=[5.0], tolerance=1.0)
    gidc = gc["group_id"]
    submit(base_a, gidc, "m1", [[10.0, 0.0], [0.0, 0.0]], 1)
    submit(base_b, gidc, "m2", [[0.0, 0.0], [0.0, 0.0]], 1)
    http("POST", base_a, f"/groups/{gidc}/publish")
    s, d = submit(base_b, gidc, "m1", [[50.0, 0.0], [0.0, 0.0]], 2)
    check("review-race revision held", s == 200 and d["status"] == "pending", str(d))
    outcomes: Dict[str, Any] = {}

    def do_review(action: str, instance: str) -> None:
        outcomes[action] = review(instance, gidc, "m1", 2, action)

    r1 = threading.Thread(target=do_review, args=("approve", base_a))
    r2 = threading.Thread(target=do_review, args=("reject", base_b))
    r1.start(); r2.start(); r1.join(); r2.join()
    statuses = {action: st for action, (st, _) in outcomes.items()}
    check("exactly one review action wins",
          sorted(statuses.values()) == [200, 409], str(outcomes))
    winner = [a for a, st in statuses.items() if st == 200][0]
    loser = "reject" if winner == "approve" else "approve"
    s, d = review(base_a, gidc, "m1", 2, loser)
    check("losing action replayed later is refused", s == 409, str(d))
    s, st = http("GET", base_b, f"/groups/{gidc}")
    m1 = [m for m in st["members"] if m["member_id"] == "m1"][0]
    check("no pending revision remains after the race", m1["pending_revision"] is None, str(m1))
    if winner == "approve":
        check("approve winner adopted the revision", m1["latest_adopted_revision"] == 2, str(m1))
    else:
        check("reject winner keeps prior adopted revision", m1["latest_adopted_revision"] == 1, str(m1))

    print("\n[20] failed publish neither leaves half a product nor revives expired")
    gf = register(base_a, members=2, rows=4, cols=4, thresholds=[5.0], tolerance=1.0)
    gidf = gf["group_id"]
    submit(base_a, gidf, "m1", [[6.0] * 4 for _ in range(4)], 1)
    submit(base_b, gidf, "m2", [[4.0] * 4 for _ in range(4)], 1)
    s, pf = http("POST", base_a, f"/groups/{gidf}/publish")
    check("failure-path baseline product", s == 200, str(pf))
    submit(base_b, gidf, "m1", [[6.5] * 4 for _ in range(4)], 2)  # adopted within tolerance
    s, d = http("GET", base_a, f"/groups/{gidf}/products/current")
    check("correction expired baseline product", s == 404)
    if hooks_enabled:
        with open(fail_marker, "w") as fh:
            fh.write("fail")
        s, d = http("POST", base_b, f"/groups/{gidf}/publish")
        os.remove(fail_marker)
        check("failed publish returns 500", s == 500 and d["error"]["code"] == "COMPUTE_FAILED")
        s, d = http("GET", base_a, f"/groups/{gidf}/products/current")
        check("failed publish does not revive expired product", s == 404, str(d))
        s, listing = http("GET", base_b, f"/groups/{gidf}/products")
        check("failed publish created no product row",
              [p["product_id"] for p in listing["products"]] == [pf["product_id"]], str(listing))
        check("failed publish left no current product",
              all(p["is_current"] is False for p in listing["products"]))
        if db_path:
            rows = db_scalar(db_path, "SELECT COUNT(*) FROM products WHERE group_id = ?", (gidf,))
            check("database has exactly the baseline product row", rows == 1, f"rows={rows}")
            orphans = db_scalar(
                db_path,
                "SELECT COUNT(*) FROM product_thresholds t"
                " LEFT JOIN products p ON p.product_id = t.product_id"
                " WHERE p.product_id IS NULL")
            check("no orphan threshold rows after failed publish", orphans == 0, f"orphans={orphans}")
        s, pf2 = http("POST", base_a, f"/groups/{gidf}/publish")
        check("publish succeeds after failure cleared",
              s == 200 and pf2["is_current"] and pf2["members_used"][0]["revision"] == 2, str(pf2))
    else:
        print("  SKIP  failure injection (RAIN_TEST_FAIL_MARKER not configured)")

    print("\n[21] churn: submits across instances with concurrent publishes")
    gz = register(base_b, members=3, rows=16, cols=16, thresholds=[5.0, 20.0], tolerance=1000.0)
    gidz = gz["group_id"]
    for i, inst in enumerate((base_a, base_b, base_a)):
        submit(inst, gidz, f"m{i + 1}", [[10.0] * 16 for _ in range(16)], 1)
    churn_errors: list = []
    pub_status: list = []

    def churn(member: str, instance: str, seed: float) -> None:
        for r in range(2, 7):
            s, d = submit(instance, gidz, member,
                          [[10.0 + seed + r * 0.01] * 16 for _ in range(16)], r)
            if s != 200:
                churn_errors.append((member, r, s, d))
            time.sleep(0.01)

    def publish_loop(instance: str) -> None:
        for _ in range(3):
            s, d = http("POST", instance, f"/groups/{gidz}/publish")
            pub_status.append((s, d.get("error", {}).get("code") if isinstance(d, dict) else None))
            time.sleep(0.05)

    threads = [
        threading.Thread(target=churn, args=("m1", base_a, 0.0)),
        threading.Thread(target=churn, args=("m2", base_b, 0.5)),
        threading.Thread(target=publish_loop, args=(base_a,)),
        threading.Thread(target=publish_loop, args=(base_b,)),
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    check("all churn submissions adopted", churn_errors == [], str(churn_errors[:2]))
    check("interleaved publishes never fail with server errors",
          all(s in (200, 409, 503) for s, _ in pub_status), str(pub_status))
    s, final = http("POST", base_a, f"/groups/{gidz}/publish")
    check("final publish succeeds", s == 200 and final["is_current"], str(final))
    s, st = http("GET", base_b, f"/groups/{gidz}")
    adopted = {m["member_id"]: m["latest_adopted_revision"] for m in st["members"]}
    s, cur = http("GET", base_a, f"/groups/{gidz}/products/current")
    check("final current product matches complete adopted set",
          s == 200 and {m["member_id"]: m["revision"] for m in cur["members_used"]} == adopted,
          f"{cur.get('members_used')} vs {adopted}")
    s, listing = http("GET", base_b, f"/groups/{gidz}/products")
    check("exactly one current product after churn",
          sum(1 for p in listing["products"] if p["is_current"]) == 1)
    check("every retained product covers all three members",
          all(len(p["revision_map"]) == 3 for p in listing["products"]), str(listing["products"]))
    check("no product references a revision newer than adopted",
          all(all(int(v) <= adopted[f"m{int(k) + 1}"] for k, v in p["revision_map"].items())
              for p in listing["products"]))

    print("\n[22] schema migration from the pre-QC database layout")
    if db_path:
        old_db = os.path.join(os.path.dirname(db_path), "legacy.db")
        conn = sqlite3.connect(old_db)
        conn.executescript(
            """
            CREATE TABLE groups (
                group_id TEXT PRIMARY KEY, forecast_time TEXT NOT NULL,
                rows INTEGER NOT NULL, cols INTEGER NOT NULL,
                thresholds TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT 'x');
            CREATE TABLE members (
                group_id TEXT NOT NULL, member_idx INTEGER NOT NULL,
                member_id TEXT NOT NULL, PRIMARY KEY (group_id, member_idx),
                UNIQUE (group_id, member_id));
            CREATE TABLE member_revisions (
                group_id TEXT NOT NULL, member_idx INTEGER NOT NULL,
                revision INTEGER NOT NULL, content_hash TEXT NOT NULL,
                rain_blob BLOB NOT NULL, miss_blob BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT 'x',
                PRIMARY KEY (group_id, member_idx, revision));
            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY AUTOINCREMENT, group_id TEXT NOT NULL,
                revision_map TEXT NOT NULL, is_current INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT 'x', UNIQUE (group_id, revision_map));
            CREATE TABLE product_thresholds (
                product_id INTEGER NOT NULL, threshold REAL NOT NULL,
                probability_blob BLOB NOT NULL, denom_blob BLOB NOT NULL,
                PRIMARY KEY (product_id, threshold));
            INSERT INTO groups VALUES ('old', '2026-01-01T00:00:00+00:00', 1, 1, '[5.0]', 'x');
            INSERT INTO members VALUES ('old', 0, 'm1');
            INSERT INTO member_revisions VALUES ('old', 0, 1, 'h', x'00', x'00', 'x');
            """
        )
        conn.commit()
        conn.close()
        env = dict(os.environ)
        env.update(RAIN_DB_PATH=old_db, PYTHONPATH=ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", "from app.db import init_db; init_db(); init_db()"],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=60)
        check("migration runs twice without error", proc.returncode == 0, proc.stderr[-400:])
        conn = sqlite3.connect(old_db)
        try:
            group_cols = {r[1] for r in conn.execute("PRAGMA table_info(groups)")}
            rev_cols = {r[1] for r in conn.execute("PRAGMA table_info(member_revisions)")}
            rev_row = conn.execute(
                "SELECT status, anomaly_cells FROM member_revisions"
                " WHERE group_id = 'old'").fetchone()
            indexes = {r[1] for r in conn.execute("PRAGMA index_list(member_revisions)")}
        finally:
            conn.close()
        check("legacy group gained change_tolerance", "change_tolerance" in group_cols)
        check("legacy revisions gained QC columns",
              {"status", "anomaly_cells", "reviewed_at"} <= rev_cols, str(rev_cols))
        check("legacy revision rows are adopted with zero anomalies",
              rev_row == ("adopted", 0), str(rev_row))
        check("pending-uniqueness index created", "idx_member_revisions_one_pending" in indexes,
              str(indexes))


def run(base_a: str, base_b: str, fail_marker: str, hooks_enabled: bool,
        db_path: str = "") -> None:
    print("\n[1] validation on registration")
    bad = {"forecast_time": "2026-11-16T00:00:00Z", "rows": 2, "cols": 2,
           "member_names": ["only-one"], "thresholds": [5.0]}
    s, d = http("POST", base_a, "/groups", bad)
    check("1 member rejected (need 2-8)", s == 422, str(d))
    bad["member_names"] = [f"m{i}" for i in range(9)]
    s, d = http("POST", base_a, "/groups", bad)
    check("9 members rejected", s == 422)
    bad["member_names"] = ["a", "b"]
    bad["thresholds"] = [-1.0]
    s, d = http("POST", base_a, "/groups", bad)
    check("negative threshold rejected", s == 422)
    bad["thresholds"] = [5.0]
    bad["forecast_time"] = "2026-11-16T00:00:00"  # naive, no timezone
    s, d = http("POST", base_a, "/groups", bad)
    check("naive datetime rejected", s == 422)

    print("\n[2] happy-path group + member submission semantics")
    g = register(base_a, members=3, thresholds=[5.0, 10.0])
    gid = g["group_id"]
    check("registered 3 members", len(g["members"]) == 3)
    s, d = http("GET", base_b, f"/groups/{gid}")
    check("group visible from instance B", s == 200 and d["all_members_present"] is False)

    rain1 = [[0.0, 10.0], [20.0, 0.0]]
    s, d = http("PUT", base_a, f"/groups/{gid}/members/m1",
                {"rain": rain1, "revision": 1})
    check("m1 rev1 accepted", s == 200 and d["duplicate"] is False, str(d))
    s, d2 = http("PUT", base_b, f"/groups/{gid}/members/m1",
                 {"rain": rain1, "revision": 1})
    check("same revision+content resent is idempotent", s == 200 and d2["duplicate"] is True)
    check("idempotent hash stable", d["content_hash"] == d2["content_hash"])
    s, d = http("PUT", base_a, f"/groups/{gid}/members/m1",
                {"rain": [[0.0, 10.0], [20.0, 1.0]], "revision": 1})
    check("same revision different content rejected (异文拒收)",
          s == 409 and d["error"]["code"] == "REVISION_CONTENT_CONFLICT", str(d))
    s, d = http("PUT", base_a, f"/groups/{gid}/members/ghost",
                {"rain": rain1, "revision": 1})
    check("unknown member -> 404", s == 404 and d["error"]["code"] == "MEMBER_NOT_FOUND")
    s, d = http("PUT", base_a, "/groups/nope/members/m1", {"rain": rain1, "revision": 1})
    check("unknown group -> 404", s == 404 and d["error"]["code"] == "GROUP_NOT_FOUND")

    print("\n[3] revision ordering and grid validation on a second group")
    g2 = register(base_b, members=2, thresholds=[5.0])
    gid2 = g2["group_id"]
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
    check("g2 m1 rev1", s == 200)
    s, d = http("PUT", base_b, f"/groups/{gid2}/members/m1",
                {"rain": [[9.0, 9.0], [9.0, 9.0]], "revision": 2})
    check("g2 m1 rev2", s == 200)
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "revision": 1})
    check("lower revision rejected", s == 409 and d["error"]["code"] == "REVISION_CONFLICT")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "revision": 3})
    check("grid shape mismatch rejected", s == 422 and d["error"]["code"] == "GRID_MISMATCH")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[-1.0, 2.0], [3.0, 4.0]], "revision": 3})
    check("negative rain rejected", s == 422 and d["error"]["code"] == "NEGATIVE_RAIN")
    s, d = http("PUT", base_a, f"/groups/{gid2}/members/m1",
                {"rain": [[1.0, 2.0], [3.0, 4.0]], "missing": [[1, 0, 0]], "revision": 3})
    check("missing grid shape mismatch rejected",
          s == 422 and d["error"]["code"] == "GRID_MISMATCH", str(d))

    print("\n[4] publishing requires every member")
    s, d = http("POST", base_a, f"/groups/{gid}/publish")
    check("incomplete group cannot publish",
          s == 409 and d["error"]["code"] == "MEMBERS_INCOMPLETE", str(d))

    print("\n[5] complete group and verify probability math")
    rain2 = [[10.0, 10.0], [0.0, 0.0]]
    rain3 = [[5.0, 0.0], [99.0, 0.0]]
    miss3 = [[0, 0], [1, 1]]  # m3 missing on second row
    s, _ = http("PUT", base_b, f"/groups/{gid}/members/m2",
                {"rain": rain2, "revision": 1})
    check("m2 rev1 via instance B", s == 200)
    s, _ = http("PUT", base_a, f"/groups/{gid}/members/m3",
                {"rain": rain3, "missing": miss3, "revision": 1})
    check("m3 rev1 with missing grid", s == 200)

    s, p1 = http("POST", base_a, f"/groups/{gid}/publish")
    check("publish succeeds when complete", s == 200, str(p1))
    check("product marked current", p1["is_current"] is True)
    check("3 member revisions recorded",
          {m["member_id"]: m["revision"] for m in p1["members_used"]}
          == {"m1": 1, "m2": 1, "m3": 1})
    by_thr = {t["threshold"]: t for t in p1["thresholds"]}
    check("denominators exclude missing members",
          by_thr[5.0]["denominator"] == [[3, 3], [2, 2]],
          str(by_thr[5.0]["denominator"]))
    check("probability threshold=5",
          matrix_approx(by_thr[5.0]["probability"],
                        [[2 / 3, 2 / 3], [0.5, 0.0]]),
          str(by_thr[5.0]["probability"]))
    check("probability threshold=10",
          matrix_approx(by_thr[10.0]["probability"],
                        [[1 / 3, 2 / 3], [0.5, 0.0]]),
          str(by_thr[10.0]["probability"]))
    s, p1b = http("POST", base_b, f"/groups/{gid}/publish")
    check("republish identical revision set is idempotent",
          s == 200 and p1b["product_id"] == p1["product_id"])
    s, cur = http("GET", base_a, f"/groups/{gid}/products/current")
    check("current product query", s == 200 and cur["product_id"] == p1["product_id"])
    s, listing = http("GET", base_b, f"/groups/{gid}/products")
    check("one product in history listing", s == 200 and len(listing["products"]) == 1)
    s, d = http("GET", base_a, f"/groups/{gid2}/products/{p1['product_id']}")
    check("product from other group -> 404", s == 404)

    print("\n[6] all-missing cell yields null probability and zero denominator")
    g3 = register(base_a, members=2, rows=1, cols=2, thresholds=[5.0])
    gid3 = g3["group_id"]
    http("PUT", base_a, f"/groups/{gid3}/members/m1",
         {"rain": [[5.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    http("PUT", base_b, f"/groups/{gid3}/members/m2",
         {"rain": [[10.0, 9.0]], "missing": [[0, 1]], "revision": 1})
    s, p3 = http("POST", base_a, f"/groups/{gid3}/publish")
    t3 = p3["thresholds"][0]
    check("all-missing cell null", t3["probability"] == [[1.0, None]], str(t3))
    check("all-missing cell denominator 0", t3["denominator"] == [[2, 0]])

    print("\n[7] correction expires old product immediately, history retained")
    s, _ = http("PUT", base_b, f"/groups/{gid}/members/m1",
                {"rain": [[0.0, 10.0], [20.0, 5.0]], "revision": 2})
    check("correction rev2 accepted", s == 200)
    s, d = http("GET", base_a, f"/groups/{gid}/products/current")
    check("no stale product served as current warning basis",
          s == 404 and d["error"]["code"] == "NO_CURRENT_PRODUCT", str(d))
    s, listing = http("GET", base_b, f"/groups/{gid}/products")
    check("old product retained but superseded",
          len(listing["products"]) == 1 and listing["products"][0]["is_current"] is False)
    s, hist = http("GET", base_a, f"/groups/{gid}/products/{p1['product_id']}")
    check("historical product still queryable", s == 200 and hist["is_current"] is False)
    s, p2 = http("POST", base_b, f"/groups/{gid}/publish")
    check("republish after correction", s == 200 and p2["product_id"] != p1["product_id"])
    check("new product uses rev2 for corrected member",
          {m["member_id"]: m["revision"] for m in p2["members_used"]}
          == {"m1": 2, "m2": 1, "m3": 1})
    s, cur = http("GET", base_b, f"/groups/{gid}/products/current")
    check("current points at new product", cur["product_id"] == p2["product_id"])

    print("\n[8] dual-instance interleaving: publish vs concurrent correction")
    g4 = register(base_a, members=3, rows=64, cols=64, thresholds=[5.0, 20.0])
    gid4 = g4["group_id"]
    for i, inst in enumerate((base_a, base_b, base_a)):
        name = f"m{i + 1}"
        grid = [[float((i + 1) * (r + c) % 25) for c in range(64)] for r in range(64)]
        s, _ = http("PUT", inst, f"/groups/{gid4}/members/{name}",
                    {"rain": grid, "revision": 1})
        assert s == 200

    result: Dict[str, Any] = {}

    def worker_publish() -> None:
        result["status"], result["body"] = http("POST", base_a, f"/groups/{gid4}/publish")

    t = threading.Thread(target=worker_publish)
    t.start()
    time.sleep(0.15)  # let the publish enter its slow compute phase
    t0 = time.time()
    s, corr = http("PUT", base_b, f"/groups/{gid4}/members/m2",
                   {"rain": [[7.0] * 64 for _ in range(64)], "revision": 2})
    write_elapsed = time.time() - t0
    check("correction write not blocked by in-flight compute", s == 200, str(corr))
    if hooks_enabled:
        check("ingest stayed fast while compute was running", write_elapsed < 0.5,
              f"{write_elapsed:.2f}s")
    t.join()
    check("in-flight publish eventually returns 200", result["status"] == 200, str(result))

    s, status4 = http("GET", base_a, f"/groups/{gid4}")
    latest = {m["member_id"]: m["latest_revision"] for m in status4["members"]}
    s, cur4 = http("GET", base_b, f"/groups/{gid4}/products/current")
    check("a current product exists after interleaving", s == 200)
    cur_map = {m["member_id"]: m["revision"] for m in cur4["members_used"]}
    check("current product matches latest member revisions (no mixed revision group)",
          cur_map == latest, f"{cur_map} vs {latest}")
    s, listing4 = http("GET", base_a, f"/groups/{gid4}/products")
    for prod in listing4["products"]:
        check(f"product {prod['product_id']} has complete revision set",
              len(prod["revision_map"]) == 3)
        if prod["is_current"]:
            check("only one current product",
                  sum(1 for x in listing4["products"] if x["is_current"]) == 1)
    check("current listing entry consistent",
          [x for x in listing4["products"] if x["is_current"]][0]["product_id"]
          == cur4["product_id"])

    print("\n[9] parallel publishes of the same revision set collapse to one product")
    g5 = register(base_b, members=2, rows=8, cols=8, thresholds=[5.0])
    gid5 = g5["group_id"]
    http("PUT", base_a, f"/groups/{gid5}/members/m1",
         {"rain": [[6.0] * 8 for _ in range(8)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid5}/members/m2",
         {"rain": [[0.0] * 8 for _ in range(8)], "revision": 1})
    outs: Dict[str, Any] = {}

    def pub(instance: str) -> None:
        outs[instance] = http("POST", instance, f"/groups/{gid5}/publish")

    threads = [threading.Thread(target=pub, args=(base_a,)),
               threading.Thread(target=pub, args=(base_b,))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    ids = {body["product_id"] for st, body in outs.values() if st == 200}
    check("both parallel publishes succeed", all(st == 200 for st, _ in outs.values()))
    check("both return the same idempotent product", len(ids) == 1, str(ids))
    s, listing5 = http("GET", base_a, f"/groups/{gid5}/products")
    check("exactly one product row for the revision set", len(listing5["products"]) == 1)

    print("\n[10] compute failure leaves no half-written product")
    g6 = register(base_a, members=2, rows=4, cols=4, thresholds=[5.0])
    gid6 = g6["group_id"]
    http("PUT", base_a, f"/groups/{gid6}/members/m1",
         {"rain": [[6.0] * 4 for _ in range(4)], "revision": 1})
    http("PUT", base_b, f"/groups/{gid6}/members/m2",
         {"rain": [[4.0] * 4 for _ in range(4)], "revision": 1})
    if hooks_enabled:
        with open(fail_marker, "w") as fh:
            fh.write("fail")
        s, d = http("POST", base_a, f"/groups/{gid6}/publish")
        os.remove(fail_marker)
        check("injected failure surfaces as 500", s == 500 and d["error"]["code"] == "COMPUTE_FAILED")
        s, d = http("GET", base_b, f"/groups/{gid6}/products/current")
        check("no current product after failed compute", s == 404)
        s, listing6 = http("GET", base_a, f"/groups/{gid6}/products")
        check("no product rows at all after failed compute", listing6["products"] == [])
        s, p6 = http("POST", base_b, f"/groups/{gid6}/publish")
        check("publish succeeds after failure cleared", s == 200 and p6["is_current"])
    else:
        print("  SKIP  failure injection (RAIN_TEST_FAIL_MARKER not configured)")

    print("\n[11] slow compute on one time does not block another time's ingestion")
    g7 = register(base_a, members=2, rows=16, cols=16, thresholds=[5.0])
    gid7 = g7["group_id"]
    http("PUT", base_a, f"/groups/{gid7}/members/m1",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})
    http("PUT", base_a, f"/groups/{gid7}/members/m2",
         {"rain": [[6.0] * 16 for _ in range(16)], "revision": 1})

    bg = {}

    def slow_publish() -> None:
        bg["result"] = http("POST", base_a, f"/groups/{gid7}/publish")

    th = threading.Thread(target=slow_publish)
    th.start()
    time.sleep(0.15)
    t0 = time.time()
    g8 = register(base_b, members=2, rows=3, cols=3, thresholds=[5.0])
    http("PUT", base_b, f"/groups/{g8['group_id']}/members/m1",
         {"rain": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], "revision": 1})
    s, _ = http("PUT", base_b, f"/groups/{g8['group_id']}/members/m2",
                {"rain": [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], "revision": 1})
    ingest_elapsed = time.time() - t0
    check("other time registration+ingest succeeded during slow compute", s == 200)
    if hooks_enabled:
        check("other time ingest stayed fast", ingest_elapsed < 0.5, f"{ingest_elapsed:.2f}s")
    th.join()
    check("slow publish completed", bg["result"][0] == 200, str(bg))

    print("\n[12] Swagger / OpenAPI")
    s, openapi = http("GET", base_a, "/openapi.json")
    check("openapi.json served", s == 200 and "paths" in openapi)
    for path in ("/groups", "/groups/{group_id}/members/{member_id}",
                 "/groups/{group_id}/publish",
                 "/groups/{group_id}/products/current",
                 "/groups/{group_id}/members/{member_id}/revisions/{revision}/review"):
        check(f"path {path} documented", path in openapi["paths"])
    req = urllib.request.Request(base_a + "/docs")
    with urllib.request.urlopen(req, timeout=10) as resp:
        check("Swagger UI served", resp.status == 200 and b"swagger" in resp.read().lower())

    run_qc_tests(base_a, base_b, fail_marker, hooks_enabled, db_path)


if __name__ == "__main__":
    sys.exit(main())
