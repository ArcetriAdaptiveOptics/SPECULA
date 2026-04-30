import time
import queue
import multiprocessing as mp
from unittest.mock import MagicMock


from specula.processing_objects.specula_input import SpeculaInput
from specula.scalar_values import FloatValue, IntValue, StringValue


def _dummy_task(q):
    q.put(("x", 99))


class TestSpeculaInput:

    def test_outputs_created(self):
        output_list = ["a:int", "b:float", "c:str"]
        obj = SpeculaInput(output_list=output_list)

        assert "a" in obj.outputs
        assert "b" in obj.outputs
        assert "c" in obj.outputs
        assert len(obj.outputs) == 3

    def test_trigger_updates_output_value(self):
        output_list = ["x:int"]
        obj = SpeculaInput(output_list=output_list)
        obj.q = mp.Queue()

        obj.current_time = 42
        obj.q.put(("x", 123))
        time.sleep(0.001)  # Allow task switch

        obj.trigger_code()

        assert obj.outputs["x"].value == 123
        assert obj.outputs["x"].generation_time == 42

    def test_trigger_handles_multiple_values(self):
        output_list = ["x:int", "y:int"]
        obj = SpeculaInput(output_list=output_list)
        obj.q = mp.Queue()

        obj.current_time = 10
        obj.q.put(("x", 1))
        obj.q.put(("y", 2))
        time.sleep(0.001)  # Allow task switch

        obj.trigger_code()

        assert obj.outputs["x"].value == 1
        assert obj.outputs["y"].value == 2

    # capfd is a pytest fixture, handled automatically
    # when running tests

    def test_trigger_ignores_unknown_output(self):
        obj = SpeculaInput(output_list=["x:int"])
        obj.q = mp.Queue()

        obj.q.put(("dummy", 5))
        time.sleep(0.001)    # Allow task switch

        obj.logger.log = MagicMock()  # Mock logger to capture error messages

        obj.trigger_code()

        assert "Unknown output" in obj.logger.log.call_args[0][1]  # Check that error was logged

    def test_set_input_task_process(self):
        obj = SpeculaInput(output_list=["x:int"])

        obj.set_input_task(_dummy_task)

        long_timeout = 10
        name = None
        start = time.time()

        # wait briefly for process to enqueue value
        while time.time() < start + long_timeout:
            try:
                name, value = obj.q.get(timeout=1)
                break
            except queue.Empty:
                pass

        if name is None:
            raise TimeoutError(f'Value from input task not received after {long_timeout} seconds')

        assert name == "x"
        assert value == 99

        obj.p.terminate()
        obj.p.join()
