import logging
import importlib
import unittest

from specula.log import (
    get_specula_logger,
    init_logging,
    SpeculaFilter,
    SpeculaAdapter,
    INIT_PLACEHOLDER_NAME,
)


class TestSpeculaLogging(unittest.TestCase):

    def setUp(self):
        """Reset logging before each test to avoid global state issues."""
        logging.shutdown()
        importlib.reload(logging)

    # ------------------------
    # get_specula_logger
    # ------------------------

    def test_get_specula_logger_returns_adapter(self):
        logger = get_specula_logger("test_logger")
        self.assertIsInstance(logger, SpeculaAdapter)

    # ------------------------
    # init_logging
    # ------------------------

    def test_init_logging_adds_filter(self):
        init_logging(process_rank=1)
        root = logging.getLogger()

        self.assertTrue(root.handlers)

        has_filter = any(
            any(isinstance(f, SpeculaFilter) for f in h.filters)
            for h in root.handlers
        )
        self.assertTrue(has_filter)

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
    # log_format tests
    # ------------------------

    def test_init_logging_default_format_without_rank(self):
        init_logging(process_rank=None)
        root = logging.getLogger()

        formatter = root.handlers[0].formatter
        fmt = formatter._fmt

        self.assertIn("%(asctime)s", fmt)
        self.assertIn("%(levelname)s", fmt)
        self.assertIn("%(name)s", fmt)
        self.assertIn("%(message)s", fmt)
        self.assertNotIn("process_rank", fmt)

    def test_init_logging_default_format_with_rank(self):
        init_logging(process_rank=7)
        root = logging.getLogger()

        formatter = root.handlers[0].formatter
        fmt = formatter._fmt

        self.assertIn("%(asctime)s", fmt)
        self.assertIn("%(levelname)s", fmt)
        self.assertIn("%(name)s", fmt)
        self.assertIn("%(message)s", fmt)
        self.assertIn("%(process_rank)s", fmt)

    def test_init_logging_custom_format(self):
        custom_format = "%(levelname)s - %(message)s"
        init_logging(log_format=custom_format)

        root = logging.getLogger()
        formatter = root.handlers[0].formatter

        self.assertEqual(formatter._fmt, custom_format)

    # ------------------------
    # SpeculaFilter
    # ------------------------

    def test_filter_replaces_name_with_instance_name(self):
        filt = SpeculaFilter(process_rank=0)

        record = logging.LogRecord(
            name="MyClass",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )
        record.instance_name = "instance1"

        filt.filter(record)

        self.assertEqual(record.name, "instance1")

    def test_filter_keeps_class_name_for_debug(self):
        filt = SpeculaFilter(process_rank=0)

        record = logging.LogRecord(
            name="MyClass",
            level=logging.DEBUG,
            pathname=__file__,
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )
        record.instance_name = "instance1"

        filt.filter(record)

        self.assertEqual(record.name, "MyClass - instance1")

    def test_filter_placeholder_name_behavior(self):
        filt = SpeculaFilter(process_rank=0)

        record = logging.LogRecord(
            name="MyClass",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="msg",
            args=(),
            exc_info=None,
        )
        record.instance_name = INIT_PLACEHOLDER_NAME

        filt.filter(record)

        self.assertEqual(record.name, f"MyClass - {INIT_PLACEHOLDER_NAME}")

    # ------------------------
    # SpeculaAdapter
    # ------------------------

    def test_set_instance_name_sets_extra(self):
        base_logger = logging.getLogger("test")
        adapter = SpeculaAdapter(base_logger)

        adapter.set_instance_name("instance42")

        self.assertEqual(adapter.extra["instance_name"], "instance42")

    def test_mpi_debug_logs_at_correct_level(self):
        base_logger = logging.getLogger("test")
        adapter = SpeculaAdapter(base_logger)

        with self.assertLogs(level=5) as cm:
            adapter.mpi_debug("mpi debug message")

        self.assertTrue(any(
            "mpi debug message" in msg for msg in cm.output
        ))

    def test_mpi_send_debug_logs_at_correct_level(self):
        base_logger = logging.getLogger("test")
        adapter = SpeculaAdapter(base_logger)

        with self.assertLogs(level=5) as cm:
            adapter.mpi_send_debug("send debug")

        self.assertTrue(any(
            "send debug" in msg for msg in cm.output
        ))



