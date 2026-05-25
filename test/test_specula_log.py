import logging
import importlib
import unittest

from specula.log import (
    get_specula_logger,
    init_logging,
    SpeculaLogAdapter,
    SpeculaLogFormatter,
    INIT_PLACEHOLDER_NAME,
    reset_logging,
)


class TestSpeculaLogging(unittest.TestCase):

    def setUp(self):
        """Reset logging before each test to avoid global state issues."""
        logging.shutdown()
        importlib.reload(logging)
        reset_logging()

    # ------------------------
    # get_specula_logger
    # ------------------------

    def test_get_specula_logger_returns_adapter(self):
        logger = get_specula_logger("test_logger")
        self.assertIsInstance(logger, SpeculaLogAdapter)

    # ------------------------
    # init_logging
    # ------------------------

    def test_init_logging_sets_process_rank(self):
        init_logging(process_rank=42)
        root = logging.getLogger()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )

        for handler in root.handlers:
            for f in handler.filters:
                f.filter(record)

        self.assertEqual(record.process_rank, 42)


    # ------------------------
    # SpeculaLogAdapter
    # ------------------------

    def test_set_instance_name_sets_extra(self):
        base_logger = logging.getLogger("test")
        adapter = SpeculaLogAdapter(base_logger)

        adapter.set_instance_name("instance42")

        self.assertEqual(adapter.extra["instance_name"], "instance42")

    def test_mpi_debug_logs_at_correct_level(self):
        base_logger = logging.getLogger("test")
        adapter = SpeculaLogAdapter(base_logger)

        with self.assertLogs(level=5) as cm:
            adapter.mpi_debug("mpi debug message")

        self.assertTrue(any(
            "mpi debug message" in msg for msg in cm.output
        ))

    def test_mpi_send_debug_logs_at_correct_level(self):
        base_logger = logging.getLogger("test")
        adapter = SpeculaLogAdapter(base_logger)

        with self.assertLogs(level=5) as cm:
            adapter.mpi_send_debug("send debug")

        self.assertTrue(any(
            "send debug" in msg for msg in cm.output
        ))



class TestSpeculaLogFormatter(unittest.TestCase):
    def setUp(self):
        self.formatter = SpeculaLogFormatter(
            fmt_with_rank="%(levelname)s [rank %(process_rank)s] [%(display_name)s]: %(message)s",
            fmt_without_rank="%(levelname)s [%(display_name)s]: %(message)s",
        )

    def make_record(self, level=logging.INFO, msg="hello", **extra):
        record = logging.LogRecord(
            name="test.module",
            level=level,
            pathname=__file__,
            lineno=1,
            msg=msg,
            args=(),
            exc_info=None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    # --- process_rank behavior ---

    def test_without_process_rank_uses_simple_format(self):
        record = self.make_record()
        output = self.formatter.format(record)

        self.assertIn("[test.module]", output)
        self.assertNotIn("rank", output)

    def test_with_process_rank_uses_rank_format(self):
        record = self.make_record(process_rank=3)
        output = self.formatter.format(record)

        self.assertIn("[rank 3]", output)

    # --- instance_name behavior ---

    def test_instance_name_replaces_name_info_level(self):
        record = self.make_record(instance_name="worker-1", level=logging.INFO)
        output = self.formatter.format(record)

        self.assertIn("[worker-1]", output)
        self.assertNotIn("test.module (worker-1)", output)

    def test_instance_name_and_name_shown_in_debug(self):
        record = self.make_record(instance_name="worker-1", level=logging.DEBUG)
        output = self.formatter.format(record)

        self.assertIn("test.module (worker-1)", output)

    def test_no_instance_name_falls_back_to_logger_name(self):
        record = self.make_record(level=logging.INFO)
        output = self.formatter.format(record)

        self.assertIn("[test.module]", output)

    # --- combined behavior ---

    def test_rank_and_instance_name_info(self):
        record = self.make_record(
            level=logging.INFO,
            process_rank=1,
            instance_name="worker-1",
        )
        output = self.formatter.format(record)

        self.assertIn("[rank 1]", output)
        self.assertIn("[worker-1]", output)
        self.assertNotIn("test.module (worker-1)", output)

    def test_rank_and_instance_name_debug(self):
        record = self.make_record(
            level=logging.DEBUG,
            process_rank=2,
            instance_name="worker-2",
        )
        output = self.formatter.format(record)

        self.assertIn("[rank 2]", output)
        self.assertIn("test.module (worker-2)", output)


