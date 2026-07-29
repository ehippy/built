"""logging_config.py — the in-memory ring buffer every built.* logger funnels
into (main.py calls configure_logging() at import time), and the /ui/logs view
polls. Assertions here only ever check deltas or entries strictly after a
captured cutoff seq, never absolute buffer state — the buffer is a
process-global singleton shared across the whole test session, and other tests
(e.g. test_archiver.py) legitimately log into it too."""

import logging

from built.logging_config import _MAX_BUFFERED_RECORDS, configure_logging, get_logs

# conftest.py already calls configure_logging() for the whole test session; these
# tests exercise that call directly too where relevant.


async def test_captures_a_log_record_from_any_built_submodule():
    logger = logging.getLogger("built.orchestrator.archiver")
    prior = get_logs()
    cutoff = prior[-1].seq if prior else 0

    logger.info("archived %d done card(s)", 3)

    new_logs = get_logs(since_seq=cutoff)
    assert len(new_logs) == 1
    assert new_logs[0].message == "archived 3 done card(s)"
    assert new_logs[0].level == "INFO"
    assert new_logs[0].logger_name == "built.orchestrator.archiver"


async def test_since_seq_returns_only_strictly_newer_entries():
    logging.getLogger("built.x").info("one")
    cutoff = get_logs()[-1].seq
    logging.getLogger("built.x").info("two")
    logging.getLogger("built.x").info("three")

    newer = get_logs(since_seq=cutoff)
    assert [e.message for e in newer] == ["two", "three"]


async def test_ring_buffer_is_bounded_to_the_most_recent_entries():
    logger = logging.getLogger("built.x")
    for i in range(_MAX_BUFFERED_RECORDS + 50):
        logger.info("line %d", i)

    logs = get_logs(limit=_MAX_BUFFERED_RECORDS + 100)
    assert len(logs) == _MAX_BUFFERED_RECORDS
    # Oldest entries fell off — the buffer keeps the most recent N, not the first N.
    assert logs[-1].message == f"line {_MAX_BUFFERED_RECORDS + 49}"


async def test_configure_logging_is_idempotent():
    app_logger = logging.getLogger("built")
    handlers_before = list(app_logger.handlers)
    configure_logging()
    configure_logging()
    assert app_logger.handlers == handlers_before
