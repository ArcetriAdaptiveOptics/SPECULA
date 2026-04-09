import sys
import time
import queue
import multiprocessing as mp


from specula.processing_objects.specula_input import SpeculaInput


def _dummy_task(q):
    q.put(("x", 99))


class TestSpeculaInput:

    def test_outputs_created(self):
        obj = SpeculaInput(output_list=["a", "b"])

        assert "a" in obj.outputs
        assert "b" in obj.outputs
        assert len(obj.outputs) == 2

    def test_trigger_updates_output_value(self):
        obj = SpeculaInput(output_list=["x"])
        obj.q = mp.Queue()

        obj.current_time = 42
        obj.q.put(("x", 123))
        time.sleep(0.001)  # Allow task switch

        obj.trigger_code()

        assert obj.outputs["x"].value == 123
        assert obj.outputs["x"].generation_time == 42

    def test_trigger_handles_multiple_values(self):
        obj = SpeculaInput(output_list=["x", "y"])
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

    def test_trigger_ignores_unknown_output(self, capfd):
        obj = SpeculaInput(output_list=["x"])
        obj.q = mp.Queue()

        obj.q.put(("dummy", 5))
        time.sleep(0.001)    # Allow task switch

        obj.trigger_code()

        captured = capfd.readouterr()
        assert "Unknown output dummy" in captured.out

    def test_set_input_task_process(self):
        obj = SpeculaInput(output_list=["x"])

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
