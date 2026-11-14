from __future__ import annotations

import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from training_portal import main as app_main
from training_portal.service import Portal


@pytest.fixture()
def portal(tmp_path):
    instance = Portal(tmp_path / "data")
    instance.start_workers()
    yield instance
    instance.shutdown()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    instance = Portal(tmp_path / "data")
    monkeypatch.setattr(app_main, "portal", instance)
    with TestClient(app_main.app) as test_client:
        yield test_client, instance
    instance.shutdown()


CSV_SPRING = """batch,respondent_id,text
B1,R1,春天第一段访谈正文
B1,R2,春天第二段访谈正文
B1,R3,春天第三段访谈正文
B2,R4,另一个批次的正文
"""


def upload(instance: Portal, csv_text: str, name: str = "u.csv") -> dict:
    return instance.ingest_upload(name, csv_text.encode("utf-8"))


def wait_for(instance: Portal, job_id: int, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for job in instance.jobs():
            if job["id"] == job_id:
                if job["status"] in ("completed", "blocked", "failed"):
                    return job
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not settle")


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------
def test_upload_parses_headers_and_reports_counts(portal):
    result = upload(portal, CSV_SPRING)
    assert result["accepted"] == 4
    assert result["batches"] == ["B1", "B2"]
    assert portal.stats()["utterances"] == 4


def test_upload_accepts_chinese_headers(portal):
    csv_text = "访谈批次,受访者编号,文本\nB1,R1,中文表头也支持\n"
    assert upload(portal, csv_text)["accepted"] == 1


def test_upload_rejects_missing_column(portal):
    from training_portal.service import ValidationError

    with pytest.raises(ValidationError):
        upload(portal, "batch,text\nB1,缺一列\n")


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------
def test_generation_produces_csv_for_selected_batch(portal):
    upload(portal, CSV_SPRING)
    video = portal.create_job("B1")
    job = wait_for(portal, video)
    assert job["status"] == "completed"

    export = portal.exports()[0]
    handle, name, error = portal.open_download(export["id"])
    assert error is None and handle is not None
    content = handle.read().decode("utf-8")
    handle.close()
    assert "春天第一段访谈正文" in content
    assert "另一个批次的正文" not in content
    assert content.splitlines()[0] == "batch,respondent_id,text"


def test_generation_of_empty_batch_is_rejected(portal):
    from training_portal.service import ValidationError

    with pytest.raises(ValidationError):
        portal.create_job("does-not-exist")


# ---------------------------------------------------------------------------
# revocation
# ---------------------------------------------------------------------------
def test_revocation_deletes_rows_and_locks_existing_exports(portal):
    upload(portal, CSV_SPRING)
    job_id = portal.create_job("B1")
    wait_for(portal, job_id)
    export_id = portal.exports()[0]["id"]

    result = portal.revoke("R2")
    assert result["deleted_rows"] == 1
    assert result["affected_exports"] == 1

    handle, _name, error = portal.open_download(export_id)
    assert handle is None
    assert "撤回" in error
    assert not [e for e in portal.exports() if e["id"] == export_id and e["status"] == "ready"]


def test_revocation_keeps_unaffected_files_downloadable(portal):
    upload(portal, CSV_SPRING)
    job_b1 = portal.create_job("B1")
    job_b2 = portal.create_job("B2")
    wait_for(portal, job_b1)
    wait_for(portal, job_b2)
    exports = {e["batch"]: e for e in portal.exports()}

    portal.revoke("R1")

    handle, _name, error = portal.open_download(exports["B2"]["id"])
    assert error is None and handle is not None
    handle.close()


def test_revocation_blocks_later_upload_of_same_id(portal):
    upload(portal, CSV_SPRING)
    portal.revoke("R1")

    result = upload(portal, "batch,respondent_id,text\nB9,R1,重新上传同号文本\n")
    assert result["accepted"] == 0
    assert result["skipped_revoked"] == 1
    assert "B9" not in [b["name"] for b in portal.batches()]


def test_revocation_blocks_original_text_under_new_id(portal):
    upload(portal, CSV_SPRING)
    portal.revoke("R1")

    csv_text = "batch,respondent_id,text\nB9,R1-ALT,春天第一段访谈正文\n"
    result = upload(portal, csv_text)
    assert result["accepted"] == 0
    assert result["skipped_revoked"] == 1


def test_revocation_is_persistent_across_restart(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    upload(first, CSV_SPRING)
    first.revoke("R2")
    first.shutdown()

    second = Portal(data_dir)
    try:
        handle, _name, error = second.open_download(1) if second.exports() else (None, "", "")
        assert second.stats()["revocations"] == 1
        result = upload(second, "batch,respondent_id,text\nB1,R2,换新文本再试\n")
        assert result["skipped_revoked"] == 1
        assert handle is None
    finally:
        second.shutdown()


@pytest.mark.parametrize("respondent", ["R1", "R2", "R3", "R4"])
def test_revocation_of_any_respondent_leaves_consistent_state(portal, respondent):
    upload(portal, CSV_SPRING)
    job = portal.create_job("B1")
    wait_for(portal, job)
    portal.revoke(respondent)

    in_batch = respondent in {"R1", "R2", "R3"}
    for export in portal.exports():
        if export["status"] == "ready":
            handle, _name, error = portal.open_download(export["id"])
            assert error is None
            handle.close()

    export_id = portal.exports()[0]["id"]
    handle, _n, error = portal.open_download(export_id)
    if in_batch:
        assert handle is None and "撤回" in error
    else:
        # A respondent from another batch must not take this file down.
        assert handle is not None and error is None
        handle.close()


# ---------------------------------------------------------------------------
# mid-generation revocation
# ---------------------------------------------------------------------------
def test_revocation_during_generation_prevents_publication(client):
    test_client, instance = client
    repair = "\n".join(f"B1,R{i},第{i}段较长的访谈正文内容用于演示生成过程中的撤回" for i in range(200))
    upload(instance, "batch,respondent_id,text\n" + repair)
    instance.db.set_settings({"pause_generation": "1"})

    r = test_client.post("/generate", data={"batch": "B1"})
    assert r.status_code == 200

    deadline = time.monotonic() + 5
    paused = None
    while time.monotonic() < deadline:
        paused = next((j for j in instance.jobs() if j["status"] == "paused"), None)
        if paused:
            break
        time.sleep(0.05)
    assert paused is not None, "job never reached the pause gate"

    r = test_client.post("/revoke", data={"respondent_id": "R37"})
    assert r.status_code == 200

    instance.db.set_settings({"pause_generation": "0"})
    deadline = time.monotonic() + 5
    settled = None
    while time.monotonic() < deadline:
        settled = instance.jobs()[0]
        if settled["status"] in ("blocked", "completed", "failed"):
            break
        time.sleep(0.05)
    assert settled["status"] == "blocked"

    assert instance.exports() == []
    assert "R37" in r.text or "撤回" in r.text


# ---------------------------------------------------------------------------
# delete failures, pending cleanup, restarts
# ---------------------------------------------------------------------------
def test_delete_failure_leaves_pending_and_blocks_download(portal):
    upload(portal, CSV_SPRING)
    job = portal.create_job("B1")
    wait_for(portal, job)
    export_id = portal.exports()[0]["id"]

    portal.db.set_settings({"delete_failures_remaining": 2})
    portal.revoke("R2")

    deadline = time.monotonic() + 5
    export = None
    while time.monotonic() < deadline:
        export = next(e for e in portal.exports() if e["id"] == export_id)
        if export["status"] == "pending_delete" and export["delete_attempts"] >= 1:
            break
        time.sleep(0.05)
    assert export["status"] == "pending_delete"
    assert export["last_delete_error"]
    assert export["delete_attempts"] >= 1

    handle, _name, error = portal.open_download(export_id)
    assert handle is None and error

    # The file still exists on disk but is not reachable under a download name
    # until the cleanup budget is exhausted.
    path = portal.export_dir / export["stored_name"]
    assert path.exists()

    portal.db.set_settings({"delete_failures_remaining": 0})
    portal.retry_cleanup_now()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        export = next(e for e in portal.exports() if e["id"] == export_id)
        if export["status"] == "deleted":
            break
        time.sleep(0.05)
    assert export["status"] == "deleted"
    assert not path.exists()


def test_restart_continues_pending_cleanup_and_never_reopens(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    first.start_workers()
    try:
        upload(first, CSV_SPRING)
        job = first.create_job("B1")
        wait_for(first, job)
        export_id = first.exports()[0]["id"]
        first.db.set_settings({"delete_always_fail": "1"})
        first.revoke("R1")
        time.sleep(0.3)
        pending = next(e for e in first.exports() if e["id"] == export_id)
        assert pending["status"] == "pending_delete"
        assert (first.export_dir / pending["stored_name"]).exists()
        handle, _n, error = first.open_download(export_id)
        assert handle is None
    finally:
        first.shutdown()

    second = Portal(data_dir)
    second.start_workers()
    try:
        # The switch is still on after restart: cleanup retries and still fails,
        # and the file remains non-downloadable.
        time.sleep(0.4)
        row = next(e for e in second.exports() if e["id"] == export_id)
        assert row["status"] == "pending_delete"
        handle, _n, error = second.open_download(export_id)
        assert handle is None and error

        # Clearing the switch lets the retry succeed; the file is gone for good.
        second.db.set_settings({"delete_always_fail": "0"})
        second.retry_cleanup_now()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            row = next(e for e in second.exports() if e["id"] == export_id)
            if row["status"] == "deleted":
                break
            time.sleep(0.05)
        assert row["status"] == "deleted"
        assert not (second.export_dir / row["stored_name"]).exists()
    finally:
        second.shutdown()


def test_startup_sweeps_orphan_files_and_demotes_missing_files(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    first.start_workers()
    try:
        upload(first, CSV_SPRING)
        job = first.create_job("B1")
        wait_for(first, job)
        stored = first.exports()[0]["stored_name"]
    finally:
        first.shutdown()

    # Simulate an interrupted delete that left the file behind and an
    # interrupted publication that left a temp file.
    assert (data_dir / "exports" / stored).unlink() is None
    (data_dir / "exports" / "orphan.csv").write_text("leaked", encoding="utf-8")
    (data_dir / "tmp" / "half-written.csv").write_text("leaked", encoding="utf-8")

    second = Portal(data_dir)
    second.start_workers()
    try:
        assert not (data_dir / "exports" / "orphan.csv").exists()
        assert not list((data_dir / "tmp").glob("*.csv"))
        row = second.exports()[0]
        assert row["status"] == "pending_delete"
        handle, _n, error = second.open_download(row["id"])
        assert handle is None
    finally:
        second.shutdown()


def test_restart_fails_interrupted_jobs_and_keeps_queued(tmp_path):
    data_dir = tmp_path / "data"
    first = Portal(data_dir)
    try:
        upload(first, CSV_SPRING)
        # Simulate a crash while the generator held a job mid-run.
        with first.db.transaction() as conn:
            conn.execute(
                "INSERT INTO jobs(batch, status, requested_at) VALUES ('B1', 'running', ?)",
                ("2026-01-01T00:00:00+00:00",),
            )
            conn.execute(
                "INSERT INTO jobs(batch, status, requested_at) VALUES ('B2', 'queued', ?)",
                ("2026-01-01T00:00:00+00:00",),
            )
    finally:
        first.shutdown()

    second = Portal(data_dir)
    second.start_workers()
    try:
        jobs = {job["id"]: job for job in second.jobs()}
        crashed = next(j for j in jobs.values() if j["batch"] == "B1")
        queued = next(j for j in jobs.values() if j["batch"] == "B2")
        assert crashed["status"] == "failed"
        assert "重启" in crashed["message"]

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            queued = next(j for j in second.jobs() if j["batch"] == "B2")
            if queued["status"] in ("completed", "blocked", "failed"):
                break
            time.sleep(0.05)
        assert queued["status"] == "completed"
    finally:
        second.shutdown()


def test_download_blocks_text_withdrawn_before_export(portal):
    """Text ingested under another id must take an export down after revocation.

    This is the adversarial ordering: the duplicate copy is exported while its
    owner still has consent, then the real owner withdraws.
    """
    upload(
        portal,
        "batch,respondent_id,text\n"
        "B1,R-A,同一段访谈原文\n"
        "B1,R-B,另一段访谈原文\n"
        "B1,R-OTHER,同一段访谈原文\n",
    )
    job = portal.create_job("B1")
    wait_for(portal, job)
    export_id = portal.exports()[0]["id"]

    handle, _n, error = portal.open_download(export_id)
    assert error is None
    handle.close()

    portal.revoke("R-A")

    handle, _n, error = portal.open_download(export_id)
    assert handle is None and error
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        row = next(e for e in portal.exports() if e["id"] == export_id)
        if row["status"] == "deleted":
            break
        time.sleep(0.05)
    assert row["status"] == "deleted"


# ---------------------------------------------------------------------------
# byte-fingerprint upload dedup
# ---------------------------------------------------------------------------
def test_same_bytes_reupload_ingests_once_and_returns_first_receipt(portal):
    raw = CSV_SPRING.encode("utf-8")
    first = portal.ingest_upload("spring.csv", raw)
    assert first["reupload"] is False
    assert first["upload_id"] == 1

    second = portal.ingest_upload("spring.csv", raw)
    third = portal.ingest_upload("a-different-name.csv", raw)

    assert second["reupload"] is True
    assert third["reupload"] is True
    # The original receipt id and first-import statistics are always returned.
    for result in (second, third):
        assert result["upload_id"] == first["upload_id"]
        assert result["total"] == first["total"]
        assert result["accepted"] == first["accepted"]
        assert result["batches"] == first["batches"]
    assert second["reupload_count"] == 1
    assert third["reupload_count"] == 2

    assert portal.stats()["utterances"] == 4
    assert len(portal.uploads()) == 1
    receipt = portal.uploads()[0]
    assert receipt["reupload_count"] == 2
    assert receipt["file_fingerprint"] == first["fingerprint"]


def test_same_filename_different_bytes_is_a_new_upload(portal):
    raw_a = CSV_SPRING.encode("utf-8")
    raw_b = (CSV_SPRING + "B3,R9,同名文件里多出来的新内容\n").encode("utf-8")

    first = portal.ingest_upload("same-name.csv", raw_a)
    second = portal.ingest_upload("same-name.csv", raw_b)

    assert second["reupload"] is False
    assert second["upload_id"] != first["upload_id"]
    assert second["fingerprint"] != first["fingerprint"]
    # 4 rows in the first file + 5 in the second: nothing was deduplicated.
    assert portal.stats()["utterances"] == 9
    assert len(portal.uploads()) == 2


def test_concurrent_identical_uploads_ingest_exactly_once(portal):
    raw = CSV_SPRING.encode("utf-8")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: portal.ingest_upload("race.csv", raw), range(8)))

    assert portal.stats()["utterances"] == 4
    assert len(portal.uploads()) == 1
    receipt_ids = {r["upload_id"] for r in results}
    assert receipt_ids == {1}
    assert sorted(r["reupload"] for r in results) == [False] + [True] * 7
    assert portal.uploads()[0]["reupload_count"] == 7


def test_fingerprint_and_receipt_survive_restart(tmp_path):
    data_dir = tmp_path / "data"
    raw = CSV_SPRING.encode("utf-8")
    first = Portal(data_dir)
    initial = first.ingest_upload("spring.csv", raw)
    first.shutdown()

    second = Portal(data_dir)
    try:
        again = second.ingest_upload("spring.csv", raw)
        assert again["reupload"] is True
        assert again["upload_id"] == initial["upload_id"]
        assert again["accepted"] == initial["accepted"]
        assert second.stats()["utterances"] == 4
        assert len(second.uploads()) == 1
    finally:
        second.shutdown()


def test_reupload_after_full_revocation_returns_history_without_restoring(portal):
    raw = CSV_SPRING.encode("utf-8")
    initial = portal.ingest_upload("spring.csv", raw)
    for respondent in ("R1", "R2", "R3", "R4"):
        portal.revoke(respondent)

    assert portal.stats()["utterances"] == 0
    # The receipt is retained even though every corpus row is gone.
    receipts = portal.uploads()
    assert len(receipts) == 1
    assert receipts[0]["id"] == initial["upload_id"]

    again = portal.ingest_upload("spring.csv", raw)
    assert again["reupload"] is True
    assert again["upload_id"] == initial["upload_id"]
    # No corpus is restored and the historical stats are still reported.
    assert portal.stats()["utterances"] == 0
    assert again["accepted"] == initial["accepted"] == 4
    assert len(portal.uploads()) == 1


def test_partial_revocation_reupload_never_writes_other_peoples_rows(portal):
    raw = CSV_SPRING.encode("utf-8")
    initial = portal.ingest_upload("spring.csv", raw)
    portal.revoke("R1")
    assert portal.stats()["utterances"] == 3

    again = portal.ingest_upload("spring.csv", raw)
    assert again["reupload"] is True
    assert again["upload_id"] == initial["upload_id"]
    # R2/R3/R4 are still consenting, but their rows must not be written again.
    assert portal.stats()["utterances"] == 3
    with portal.db.read() as conn:
        remaining = {
            row["respondent_id"]
            for row in conn.execute("SELECT DISTINCT respondent_id FROM utterances")
        }
    assert remaining == {"R2", "R3", "R4"}


def test_failed_import_leaves_no_receipt_and_can_succeed_later(portal):
    from training_portal.service import ValidationError

    with pytest.raises(ValidationError):
        portal.ingest_upload("broken.csv", b"batch,text\nB1,missing-column\n")
    with pytest.raises(ValidationError):
        portal.ingest_upload("empty.csv", b"batch,respondent_id,text\n")
    assert portal.uploads() == []
    assert portal.stats()["utterances"] == 0

    # The same (now valid) bytes are a genuine first import, not a re-upload.
    result = portal.ingest_upload("fixed.csv", CSV_SPRING.encode("utf-8"))
    assert result["reupload"] is False
    assert len(portal.uploads()) == 1


def test_legacy_receipt_without_fingerprint_stays_usable_and_never_matches(tmp_path):
    data_dir = tmp_path / "data"
    legacy = Portal(data_dir)
    # Simulate a receipt written by the pre-fingerprint release: no fingerprint,
    # matching the NULL column produced by the ALTER TABLE migration.
    with legacy.db.transaction() as conn:
        conn.execute(
            "INSERT INTO uploads(filename, uploaded_at, total_rows, accepted_rows, "
            "skipped_revoked, skipped_invalid, batch_count, file_fingerprint, "
            "batch_names, reupload_count, last_seen_at) "
            "VALUES ('legacy.csv', ?, 3, 3, 0, 0, 1, NULL, '[]', 0, ?)",
            ("2025-12-01T00:00:00+00:00", "2025-12-01T00:00:00+00:00"),
        )
    legacy.shutdown()

    portal = Portal(data_dir)
    try:
        receipts = portal.uploads()
        assert len(receipts) == 1
        assert receipts[0]["file_fingerprint"] is None

        # No fingerprint means no guessed identity: the new file is imported,
        # and the unique index must not reject the second receipt.
        result = portal.ingest_upload("legacy.csv", CSV_SPRING.encode("utf-8"))
        assert result["reupload"] is False
        assert len(portal.uploads()) == 2
        assert portal.stats()["utterances"] == 4
    finally:
        portal.shutdown()


def test_schema_migration_adds_fingerprint_columns_to_old_database(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True)
    conn = sqlite3.connect(data_dir / "portal.sqlite3")
    conn.executescript(
        """
        CREATE TABLE uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            uploaded_at TEXT NOT NULL,
            total_rows INTEGER NOT NULL,
            accepted_rows INTEGER NOT NULL,
            skipped_revoked INTEGER NOT NULL,
            skipped_invalid INTEGER NOT NULL,
            batch_count INTEGER NOT NULL
        );
        """
    )
    conn.execute(
        "INSERT INTO uploads VALUES (1, 'old.csv', '2025-12-01T00:00:00+00:00', "
        "1, 1, 0, 0, 1)"
    )
    conn.commit()
    conn.close()

    portal = Portal(data_dir)
    try:
        row = portal.uploads()[0]
        assert row["file_fingerprint"] is None
        assert row["reupload_count"] == 0
        assert row["last_seen_at"] == "2025-12-01T00:00:00+00:00"
        # A new upload after migration gets a fingerprint and dedup works.
        raw = CSV_SPRING.encode("utf-8")
        first = portal.ingest_upload("new.csv", raw)
        again = portal.ingest_upload("new.csv", raw)
        assert first["reupload"] is False and again["reupload"] is True
    finally:
        portal.shutdown()


def test_upload_page_marks_new_import_and_reupload(client):
    test_client, instance = client
    raw = CSV_SPRING.encode("utf-8")
    instance.ingest_upload("spring.csv", raw)
    instance.ingest_upload("spring.csv", raw)

    html = test_client.get("/").text
    assert "重传 ×1" in html
    assert "重传识别" in test_client.post(
        "/upload", files={"file": ("spring.csv", raw, "text/csv")}
    ).text


def test_upload_endpoint_reports_failed_import_without_receipt(client):
    test_client, instance = client
    response = test_client.post(
        "/upload",
        files={"file": ("broken.csv", "batch,text\nB1,缺列\n".encode("utf-8"), "text/csv")},
    )
    assert response.status_code == 200
    assert "缺少必需列" in response.text
    assert instance.uploads() == []
