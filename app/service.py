"""Business logic: registration, member submission and product publishing.

Concurrency model (two API processes sharing one SQLite file)
----------------------------------------------------------------
* Every write happens inside a short ``BEGIN IMMEDIATE`` transaction, so only
  one writer exists at a time. Conflicting writers get ``SQLITE_BUSY`` after
  the busy timeout and receive HTTP 409/503 instead of corrupting state.
* Publishing reads a snapshot, runs the (potentially slow) NumPy computation
  *outside* any write transaction, then re-validates the revision set inside
  an immediate write transaction. The slow CPU work therefore never blocks
  other forecast times from ingesting members.
* A product row and its threshold rows are inserted in one transaction; any
  failure rolls the whole thing back, so no half product can ever be visible.
* Quality control: a correction that moves beyond the registered per-cell
  tolerance (or swaps valid/missing state) is stored as ``pending``, which is
  invisible to publishing and does not expire the current product. A reviewer
  approves (adopts + expires) or rejects (keeps the product) by revision number.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .db import get_conn
from .errors import ApiError
from .schemas import RegisterRequest, ReviewRequest, SubmitRequest
from .storage import (
    compute_probabilities,
    content_hash,
    decode_float,
    decode_int,
    decode_mask,
    encode_blob,
    matrix_to_json,
    to_arrays,
)

# --- test hooks (disabled unless explicitly enabled; never on by default) ---
PUBLISH_DELAY_S = float(os.environ.get("RAIN_TEST_PUBLISH_DELAY_S", "0"))
FAIL_MARKER = os.environ.get("RAIN_TEST_FAIL_MARKER", "")


def _is_busy(exc: Exception) -> bool:
    return isinstance(exc, Exception) and "SQLITE_BUSY" in str(exc)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #
def register_group(payload: RegisterRequest) -> Dict[str, Any]:
    group_id = uuid.uuid4().hex
    thresholds_json = json.dumps(payload.thresholds)
    with get_conn() as conn:
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO groups (group_id, forecast_time, rows, cols, thresholds, change_tolerance)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (
                    group_id,
                    payload.forecast_time.astimezone(timezone.utc).isoformat(),
                    payload.rows,
                    payload.cols,
                    thresholds_json,
                    payload.change_tolerance,
                ),
            )
            conn.executemany(
                "INSERT INTO members (group_id, member_idx, member_id) VALUES (?, ?, ?)",
                [(group_id, idx, name) for idx, name in enumerate(payload.member_names)],
            )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return {
        "group_id": group_id,
        "forecast_time": payload.forecast_time.astimezone(timezone.utc).isoformat(),
        "rows": payload.rows,
        "cols": payload.cols,
        "members": [
            {"member_idx": idx, "member_id": name}
            for idx, name in enumerate(payload.member_names)
        ],
        "thresholds": payload.thresholds,
        "change_tolerance": payload.change_tolerance,
    }


def _get_group(conn, group_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT * FROM groups WHERE group_id = ?", (group_id,)
    ).fetchone()
    if row is None:
        raise ApiError(404, "GROUP_NOT_FOUND", f"group {group_id!r} is not registered")
    d = dict(row)
    d["thresholds"] = json.loads(d["thresholds"])
    return d


def _get_member(conn, group_id: str, member_id: str) -> Dict[str, Any]:
    row = conn.execute(
        "SELECT member_idx, member_id FROM members WHERE group_id = ? AND member_id = ?",
        (group_id, member_id),
    ).fetchone()
    if row is None:
        raise ApiError(404, "MEMBER_NOT_FOUND", f"member {member_id!r} is not part of this group")
    return dict(row)


def _latest_adopted(conn, group_id: str, member_idx: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT revision, content_hash, rain_blob, miss_blob, created_at FROM member_revisions"
        " WHERE group_id = ? AND member_idx = ? AND status = 'adopted'"
        " ORDER BY revision DESC LIMIT 1",
        (group_id, member_idx),
    ).fetchone()
    return dict(row) if row is not None else None


def _pending_revision(conn, group_id: str, member_idx: int) -> Optional[Dict[str, Any]]:
    row = conn.execute(
        "SELECT revision, content_hash, anomaly_cells, created_at FROM member_revisions"
        " WHERE group_id = ? AND member_idx = ? AND status = 'pending'"
        " ORDER BY revision DESC LIMIT 1",
        (group_id, member_idx),
    ).fetchone()
    return dict(row) if row is not None else None


def count_anomaly_cells(
    prev_rain: np.ndarray,
    prev_miss: np.ndarray,
    rain: np.ndarray,
    miss: np.ndarray,
    tolerance: float,
) -> int:
    """Cells that make a correction suspicious ("疑似错位").

    A cell is anomalous when its validity changes (valid <-> missing) or when
    both versions are valid and the rainfall moves by more than ``tolerance``.
    Missing-cell values carry no meaning and are not compared.
    """
    if len(prev_miss) != len(miss):
        raise ApiError(422, "GRID_MISMATCH", "missing grid shape does not match rain grid")
    prev_valid = prev_miss == 0
    valid = miss == 0
    validity_swap = prev_valid != valid
    both_valid = prev_valid & valid
    exceeds = both_valid & (np.abs(rain - prev_rain) > float(tolerance))
    return int(np.count_nonzero(validity_swap | exceeds))



# --------------------------------------------------------------------------- #
# Member submission
# --------------------------------------------------------------------------- #
def submit_member(group_id: str, member_id: str, payload: SubmitRequest) -> Dict[str, Any]:
    with get_conn() as conn:
        group = _get_group(conn, group_id)
        member = _get_member(conn, group_id, member_id)
        rain_flat, miss_flat = to_arrays(
            payload.rain, payload.missing, group["rows"], group["cols"]
        )
        digest = content_hash(rain_flat, miss_flat)
        rain_blob = encode_blob(rain_flat)
        miss_blob = encode_blob(miss_flat)
        idx = member["member_idx"]
        tolerance = group.get("change_tolerance")

        try:
            conn.execute("BEGIN IMMEDIATE")
            # "latest" is the newest revision of any state: pending revisions
            # occupy their revision number and keep the sequence monotonic.
            prev = conn.execute(
                "SELECT revision, content_hash, status FROM member_revisions"
                " WHERE group_id = ? AND member_idx = ?"
                " ORDER BY revision DESC LIMIT 1",
                (group_id, idx),
            ).fetchone()
            pending = _pending_revision(conn, group_id, idx)
            duplicate = False
            if prev is not None:
                if payload.revision < prev["revision"]:
                    conn.execute("ROLLBACK")
                    raise ApiError(
                        409,
                        "REVISION_CONFLICT",
                        f"revision {payload.revision} is not greater than latest {prev['revision']}",
                    )
                if payload.revision == prev["revision"]:
                    if prev["status"] == "rejected":
                        conn.execute("ROLLBACK")
                        raise ApiError(
                            409,
                            "REVISION_REJECTED",
                            f"revision {payload.revision} was rejected by quality control;"
                            " submit a higher revision",
                        )
                    if prev["content_hash"] == digest:
                        duplicate = True  # 同版同文重送：幂等，不重复入库
                    else:
                        conn.execute("ROLLBACK")
                        raise ApiError(
                            409,
                            "REVISION_CONTENT_CONFLICT",
                            f"revision {payload.revision} already exists with different content (异文拒收)",
                        )

            stored_status = prev["status"] if duplicate else "adopted"
            stored_anomaly = 0
            if duplicate:
                row = conn.execute(
                    "SELECT anomaly_cells FROM member_revisions"
                    " WHERE group_id = ? AND member_idx = ? AND revision = ?",
                    (group_id, idx, payload.revision),
                ).fetchone()
                stored_anomaly = int(row["anomaly_cells"]) if row else 0
            anomaly_cells = 0
            if not duplicate:
                if pending is not None:
                    # 同一成员待核期间拒绝新修订（含递 revision 的更正）
                    conn.execute("ROLLBACK")
                    raise ApiError(
                        409,
                        "PENDING_REVIEW",
                        f"member {member_id!r} revision {pending['revision']} is awaiting quality "
                        "review; approve or reject it before submitting a new revision",
                    )
                latest_adopted = _latest_adopted(conn, group_id, idx)
                if tolerance is not None and latest_adopted is not None:
                    prev_rain = decode_float(
                        latest_adopted["rain_blob"], group["rows"] * group["cols"]
                    )
                    prev_miss = decode_mask(
                        latest_adopted["miss_blob"], group["rows"] * group["cols"]
                    )
                    anomaly_cells = count_anomaly_cells(
                        prev_rain, prev_miss, rain_flat, miss_flat, float(tolerance)
                    )
                    if anomaly_cells > 0:
                        # 暂存待核：不参与发布，也不使现行产品过期
                        stored_status = "pending"
                        stored_anomaly = anomaly_cells

                conn.execute(
                    "INSERT INTO member_revisions"
                    " (group_id, member_idx, revision, content_hash, rain_blob, miss_blob,"
                    "  status, anomaly_cells)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        group_id, idx, payload.revision, digest, rain_blob, miss_blob,
                        stored_status, stored_anomaly,
                    ),
                )
                if stored_status == "adopted":
                    # 更正成员：旧产品立即过期（历史行保留可查）
                    conn.execute(
                        "UPDATE products SET is_current = 0 WHERE group_id = ? AND is_current = 1",
                        (group_id,),
                    )
            conn.execute("COMMIT")
        except ApiError:
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise

    out = {
        "group_id": group_id,
        "member_id": member_id,
        "member_idx": idx,
        "revision": payload.revision,
        "content_hash": digest,
        "duplicate": duplicate,
        "status": stored_status,
    }
    if stored_status == "pending":
        out["anomaly_cells"] = stored_anomaly
    return out

# --------------------------------------------------------------------------- #
# Quality-control review of held revisions
# --------------------------------------------------------------------------- #
def review_revision(
    group_id: str, member_id: str, revision: int, payload: ReviewRequest
) -> Dict[str, Any]:
    with get_conn() as conn:
        _get_group(conn, group_id)
        member = _get_member(conn, group_id, member_id)
        idx = member["member_idx"]
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT status, anomaly_cells FROM member_revisions"
                " WHERE group_id = ? AND member_idx = ? AND revision = ?",
                (group_id, idx, revision),
            ).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                raise ApiError(
                    404,
                    "REVISION_NOT_FOUND",
                    f"member {member_id!r} has no revision {revision}",
                )
            if row["status"] != "pending":
                conn.execute("ROLLBACK")
                if row["status"] == "rejected":
                    code = "REVISION_REJECTED"
                else:
                    code = "REVISION_NOT_PENDING"
                raise ApiError(
                    409,
                    code,
                    f"revision {revision} is already {row['status']} and cannot be reviewed again",
                )

            if payload.action == "approve":
                # An approval that loses the race against a newer adopted revision
                # (only reachable on legacy data) must not resurrect an old version.
                newest = conn.execute(
                    "SELECT revision FROM member_revisions"
                    " WHERE group_id = ? AND member_idx = ? AND status = 'adopted'"
                    " ORDER BY revision DESC LIMIT 1",
                    (group_id, idx),
                ).fetchone()
                if newest is not None and int(newest["revision"]) > revision:
                    conn.execute("ROLLBACK")
                    raise ApiError(
                        409,
                        "REVISION_SUPERSEDED",
                        f"revision {revision} is older than adopted revision {newest['revision']}",
                    )
                product = conn.execute(
                    "SELECT product_id FROM products WHERE group_id = ? AND is_current = 1",
                    (group_id,),
                ).fetchone()
                if product is not None:
                    conn.execute(
                        "UPDATE products SET is_current = 0 WHERE group_id = ? AND is_current = 1",
                        (group_id,),
                    )
                conn.execute(
                    "UPDATE member_revisions SET status = 'adopted',"
                    " reviewed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')"
                    " WHERE group_id = ? AND member_idx = ? AND revision = ?",
                    (group_id, idx, revision),
                )
                expired_product_id = int(product["product_id"]) if product else None
                outcome = "approved"
            elif payload.action == "reject":
                # Rejected content is discarded; the current product, if any,
                # stays untouched. A tombstone row keeps the revision number
                # burned so a resend cannot silently re-adopt the same version.
                conn.execute(
                    "UPDATE member_revisions SET status = 'rejected',"
                    " reviewed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')"
                    " WHERE group_id = ? AND member_idx = ? AND revision = ? AND status = 'pending'",
                    (group_id, idx, revision),
                )
                expired_product_id = None
                outcome = "rejected"
            else:  # pragma: no cover - schema Literal already restricts this
                conn.execute("ROLLBACK")
                raise ApiError(422, "VALIDATION_ERROR", "action must be approve or reject")
            conn.execute("COMMIT")
        except ApiError:
            raise
        except Exception:
            conn.execute("ROLLBACK")
            raise

    latest = _member_state(group_id, member_id)
    return {
        "group_id": group_id,
        "member_id": member_id,
        "revision": revision,
        "action": payload.action,
        "outcome": outcome,
        "expired_product_id": expired_product_id,
        "member": latest,
    }


def _member_state(group_id: str, member_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        member = _get_member(conn, group_id, member_id)
        idx = member["member_idx"]
        adopted = _latest_adopted(conn, group_id, idx)
        pending = _pending_revision(conn, group_id, idx)
        return {
            "member_idx": idx,
            "member_id": member_id,
            "latest_adopted_revision": int(adopted["revision"]) if adopted else None,
            "pending": (
                {
                    "revision": int(pending["revision"]),
                    "anomaly_cells": int(pending["anomaly_cells"]),
                    "submitted_at": pending["created_at"],
                }
                if pending
                else None
            ),
        }


# --------------------------------------------------------------------------- #
# Publishing
# --------------------------------------------------------------------------- #
def _read_snapshot(conn, group_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[int, int]]:
    group = _get_group(conn, group_id)
    member_rows = conn.execute(
        "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
        (group_id,),
    ).fetchall()
    latest: List[Dict[str, Any]] = []
    revision_map: Dict[int, int] = {}
    for m in member_rows:
        rev = conn.execute(
            "SELECT revision FROM member_revisions"
            " WHERE group_id = ? AND member_idx = ? AND status = 'adopted'"
            " ORDER BY revision DESC LIMIT 1",
            (group_id, m["member_idx"]),
        ).fetchone()
        if rev is None:
            raise ApiError(
                409,
                "MEMBERS_INCOMPLETE",
                f"member {m['member_id']!r} has no adopted submission; all members must be present and adopted before publishing",
            )
        revision_map[int(m["member_idx"])] = int(rev["revision"])
        latest.append({"member_idx": int(m["member_idx"]), "member_id": m["member_id"]})
    return group, latest, revision_map


def _canonical_revision_map(revision_map: Dict[int, int]) -> str:
    return json.dumps(revision_map, sort_keys=True, separators=(",", ":"))


def _revision_set_matches(
    conn,
    product_id: int,
    per_threshold: Dict[float, Tuple[np.ndarray, np.ndarray]],
    thresholds: List[float],
    n_cells: int,
) -> bool:
    """True when the stored product grids equal a freshly computed result.

    Revision numbers can be reused after a rejected pending revision is
    discarded, so a product row keyed by a revision map is only reused when its
    actual probability/denominator grids still match the adopted content.
    """
    stored = {
        float(r["threshold"]): (r["probability_blob"], r["denom_blob"])
        for r in conn.execute(
            "SELECT threshold, probability_blob, denom_blob"
            " FROM product_thresholds WHERE product_id = ?",
            (product_id,),
        )
    }
    if set(stored) != {float(t) for t in thresholds}:
        return False
    for thr, (prob, denom) in per_threshold.items():
        if float(thr) not in stored:
            return False
        s_prob, s_denom = stored[float(thr)]
        if not np.array_equal(decode_float(s_prob, n_cells), prob, equal_nan=True):
            return False
        if not np.array_equal(decode_int(s_denom, n_cells), denom):
            return False
    return True


def publish_product(group_id: str) -> Dict[str, Any]:
    # Bounded outer loop: each iteration is one consistent revision-set attempt.
    for attempt in range(1, 4):
        # 1) snapshot under a read transaction (no write lock held during compute)
        with get_conn() as conn:
            try:
                conn.execute("BEGIN")
                group, members, revision_map = _read_snapshot(conn, group_id)
                revs = conn.execute(
                    "SELECT group_id, member_idx, revision, rain_blob, miss_blob"
                    " FROM member_revisions WHERE group_id = ?",
                    (group_id,),
                ).fetchall()
                existing = conn.execute(
                    "SELECT product_id FROM products"
                    " WHERE group_id = ? AND revision_map = ? AND is_current = 1",
                    (group_id, _canonical_revision_map(revision_map)),
                ).fetchone()
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise

        if existing is not None:
            # Fast path: the same adopted revision set is already published.
            # Re-read the row so a correction that expired it between the
            # snapshot and this load is not reported as the current product.
            loaded = _load_product(existing["product_id"])
            if loaded["is_current"]:
                return loaded

        wanted = {
            (int(r["member_idx"]), int(r["revision"])): (
                decode_float(r["rain_blob"], group["rows"] * group["cols"]),
                decode_mask(r["miss_blob"], group["rows"] * group["cols"]),
            )
            for r in revs
        }
        rains = [wanted[(idx, revision_map[idx])][0] for idx in sorted(revision_map)]
        misses = [wanted[(idx, revision_map[idx])][1] for idx in sorted(revision_map)]

        # 2) slow compute happens OUTSIDE the write lock: other groups keep ingesting
        if PUBLISH_DELAY_S > 0:
            time.sleep(PUBLISH_DELAY_S)
        if FAIL_MARKER and os.path.exists(FAIL_MARKER):
            # deterministic failure injection for atomicity self-tests
            raise ApiError(500, "COMPUTE_FAILED", "injected computation failure (test hook)")
        try:
            per_threshold = compute_probabilities(rains, misses, group["thresholds"])
        except ApiError:
            raise
        except Exception as exc:  # calculation failure must not leave half a result
            raise ApiError(500, "COMPUTE_FAILED", f"probability computation failed: {exc}") from exc

        canonical = _canonical_revision_map(revision_map)

        # 3) re-validate under the write lock and persist atomically
        write_attempts = 0
        should_retry_snapshot = False
        with get_conn() as conn:
            while True:
                write_attempts += 1
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    latest_map: Dict[int, int] = {}
                    for m in members:
                        rev = conn.execute(
                            "SELECT revision FROM member_revisions"
                            " WHERE group_id = ? AND member_idx = ? AND status = 'adopted'"
                            " ORDER BY revision DESC LIMIT 1",
                            (group_id, m["member_idx"]),
                        ).fetchone()
                        if rev is None:
                            conn.execute("ROLLBACK")
                            raise ApiError(409, "MEMBERS_INCOMPLETE", "adopted revision disappeared before publish commit")
                        latest_map[int(m["member_idx"])] = int(rev["revision"])

                    if latest_map != revision_map:
                        # an adopted correction landed while we were computing: this
                        # product would be stale; retry from the new consistent set
                        conn.execute("ROLLBACK")
                        if attempt >= 3:
                            raise ApiError(
                                409,
                                "REVISION_CHANGED",
                                "member revisions changed during publish; retry the request",
                            )
                        should_retry_snapshot = True
                        break

                    row = conn.execute(
                        "SELECT product_id, is_current FROM products"
                        " WHERE group_id = ? AND revision_map = ?",
                        (group_id, canonical),
                    ).fetchone()
                    if row is not None and int(row["is_current"]) == 1:
                        conn.execute("ROLLBACK")
                        return _load_product(int(row["product_id"]))

                    # supersede every previous product of this group (history retained)
                    conn.execute(
                        "UPDATE products SET is_current = 0 WHERE group_id = ? AND is_current = 1",
                        (group_id,),
                    )
                    if row is not None:
                        # Same revision set existed but was superseded: only revisit
                        # it if it really was built from these exact adopted
                        # contents. A revision number can be reused after a rejected
                        # pending revision was discarded, so the row alone is not
                        # proof that its probability grids match today's content.
                        same_contents = _revision_set_matches(
                            conn, int(row["product_id"]), per_threshold,
                            group["thresholds"], group["rows"] * group["cols"],
                        )
                        if not same_contents:
                            # A revision number was reused with different content
                            # (only possible with legacy/imported data). Refuse
                            # instead of resurrecting an expired product or
                            # silently rewriting retained history.
                            conn.execute("ROLLBACK")
                            raise ApiError(
                                409,
                                "PRODUCT_CONTENT_CONFLICT",
                                "a historical product for this revision set was built from "
                                "different content; publish under new revision numbers",
                            )
                        product_id = int(row["product_id"])
                        conn.execute(
                            "UPDATE products SET is_current = 1 WHERE product_id = ?",
                            (product_id,),
                        )
                        conn.execute("COMMIT")
                        return _load_product(product_id)
                    cur = conn.execute(
                        "INSERT INTO products (group_id, revision_map, is_current)"
                        " VALUES (?, ?, 1)",
                        (group_id, canonical),
                    )
                    product_id = int(cur.lastrowid)
                    conn.executemany(
                        "INSERT INTO product_thresholds"
                        " (product_id, threshold, probability_blob, denom_blob)"
                        " VALUES (?, ?, ?, ?)",
                        [
                            (
                                product_id,
                                float(thr),
                                encode_blob(per_threshold[float(thr)][0]),
                                encode_blob(per_threshold[float(thr)][1]),
                            )
                            for thr in group["thresholds"]
                        ],
                    )
                    conn.execute("COMMIT")
                    return _load_product(product_id)
                except ApiError:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
                except Exception as exc:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    if _is_busy(exc) and write_attempts < 3:
                        time.sleep(0.05 * write_attempts)
                        continue
                    if _is_busy(exc):
                        raise ApiError(503, "DATABASE_BUSY", "database locked by another writer; retry") from exc
                    raise
        if should_retry_snapshot:
            continue

    raise ApiError(409, "REVISION_CHANGED", "member revisions kept changing during publish; retry the request")


# --------------------------------------------------------------------------- #
# Queries
# --------------------------------------------------------------------------- #
def _load_product(product_id: int) -> Dict[str, Any]:
    with get_conn() as conn:
        prow = conn.execute(
            "SELECT * FROM products WHERE product_id = ?", (product_id,)
        ).fetchone()
        if prow is None:
            raise ApiError(404, "PRODUCT_NOT_FOUND", f"product {product_id} not found")
        group = _get_group(conn, prow["group_id"])
        rows, cols = group["rows"], group["cols"]
        n = rows * cols
        revision_map = {int(k): v for k, v in json.loads(prow["revision_map"]).items()}
        member_rows = conn.execute(
            "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
            (prow["group_id"],),
        ).fetchall()
        thresholds_out: List[Dict[str, Any]] = []
        for trow in conn.execute(
            "SELECT threshold, probability_blob, denom_blob"
            " FROM product_thresholds WHERE product_id = ? ORDER BY threshold",
            (product_id,),
        ):
            prob = decode_float(trow["probability_blob"], n)
            denom = decode_int(trow["denom_blob"], n)
            thresholds_out.append(
                {
                    "threshold": float(trow["threshold"]),
                    "probability": matrix_to_json(prob, rows, cols),
                    "denominator": denom.reshape(rows, cols).tolist(),
                }
            )
        member_revisions = [
            {
                "member_idx": int(m["member_idx"]),
                "member_id": m["member_id"],
                "revision": int(revision_map[int(m["member_idx"])]),
            }
            for m in member_rows
        ]
        return {
            "product_id": int(prow["product_id"]),
            "group_id": prow["group_id"],
            "forecast_time": group["forecast_time"],
            "is_current": bool(prow["is_current"]),
            "created_at": prow["created_at"],
            "revision_map": revision_map,
            "members_used": member_revisions,
            "thresholds": thresholds_out,
        }


def get_current_product(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        _get_group(conn, group_id)  # 404 on unknown group
        row = conn.execute(
            "SELECT product_id FROM products WHERE group_id = ? AND is_current = 1",
            (group_id,),
        ).fetchone()
        if row is None:
            raise ApiError(
                404,
                "NO_CURRENT_PRODUCT",
                "no current product for this group (publish first, or the previous product was superseded)",
            )
        product_id = int(row["product_id"])
    return _load_product(product_id)


def list_products(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        _get_group(conn, group_id)
        rows = conn.execute(
            "SELECT product_id, revision_map, is_current, created_at"
            " FROM products WHERE group_id = ? ORDER BY product_id",
            (group_id,),
        ).fetchall()
        return {
            "group_id": group_id,
            "products": [
                {
                    "product_id": int(r["product_id"]),
                    "revision_map": {int(k): v for k, v in json.loads(r["revision_map"]).items()},
                    "is_current": bool(r["is_current"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ],
        }


def get_product(group_id: str, product_id: int) -> Dict[str, Any]:
    product = _load_product(product_id)
    if product["group_id"] != group_id:
        raise ApiError(404, "PRODUCT_NOT_FOUND", f"product {product_id} does not belong to this group")
    return product


def get_group_status(group_id: str) -> Dict[str, Any]:
    with get_conn() as conn:
        group = _get_group(conn, group_id)
        members = conn.execute(
            "SELECT member_idx, member_id FROM members WHERE group_id = ? ORDER BY member_idx",
            (group_id,),
        ).fetchall()
        out_members = []
        complete = True
        any_pending = False
        for m in members:
            adopted = _latest_adopted(conn, group_id, m["member_idx"])
            pending = _pending_revision(conn, group_id, m["member_idx"])
            any_pending = any_pending or pending is not None
            if adopted is None:
                complete = False
                out_members.append(
                    {
                        "member_idx": int(m["member_idx"]),
                        "member_id": m["member_id"],
                        "latest_revision": None,
                        "latest_adopted_revision": None,
                        "pending_revision": None,
                    }
                )
            else:
                out_members.append(
                    {
                        "member_idx": int(m["member_idx"]),
                        "member_id": m["member_id"],
                        # kept for compatibility: newest *adopted* revision
                        "latest_revision": int(adopted["revision"]),
                        "latest_adopted_revision": int(adopted["revision"]),
                        "content_hash": adopted["content_hash"],
                        "updated_at": adopted["created_at"],
                        "pending_revision": (
                            {
                                "revision": int(pending["revision"]),
                                "anomaly_cells": int(pending["anomaly_cells"]),
                                "content_hash": pending["content_hash"],
                                "submitted_at": pending["created_at"],
                            }
                            if pending
                            else None
                        ),
                    }
                )
        current = conn.execute(
            "SELECT product_id FROM products WHERE group_id = ? AND is_current = 1",
            (group_id,),
        ).fetchone()
        return {
            "group_id": group_id,
            "forecast_time": group["forecast_time"],
            "rows": group["rows"],
            "cols": group["cols"],
            "thresholds": group["thresholds"],
            "change_tolerance": group.get("change_tolerance"),
            "members": out_members,
            "all_members_present": complete,
            "has_pending_review": any_pending,
            "current_product_id": int(current["product_id"]) if current else None,
        }


def list_groups() -> Dict[str, Any]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT group_id, forecast_time, rows, cols, thresholds, change_tolerance, created_at"
            " FROM groups ORDER BY created_at"
        ).fetchall()
        return {
            "groups": [
                {
                    "group_id": r["group_id"],
                    "forecast_time": r["forecast_time"],
                    "rows": r["rows"],
                    "cols": r["cols"],
                    "thresholds": json.loads(r["thresholds"]),
                    "change_tolerance": r["change_tolerance"],
                    "created_at": r["created_at"],
                }
                for r in rows
            ]
        }
