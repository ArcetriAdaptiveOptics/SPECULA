import threading

import pytest

import specula
specula.init(-1)  # Default target device

from specula.processing_objects.specula_input import SpeculaInput


class TestSpeculaInput:

    def test_outputs_created(self):
        output_list = ["a:int", "b:float", "c:str"]
        obj = SpeculaInput(output_list=output_list)

        assert "a" in obj.outputs
        assert "b" in obj.outputs
        assert "c" in obj.outputs
        assert len(obj.outputs) == 3

    def test_trigger_updates_output_value(self):
        obj = SpeculaInput(output_list=["x:int"])

        obj.current_time = 42
        obj.put_input("x", "123")
        obj.trigger_code()

        assert obj.outputs["x"].value == 123
        assert obj.outputs["x"].generation_time == 42

    def test_value_applied_only_at_trigger(self):
        obj = SpeculaInput(output_list=["x:float"])

        obj.put_input("x", "0.5")
        assert obj.outputs["x"].value == 0.0

        obj.current_time = 7
        obj.trigger_code()
        assert obj.outputs["x"].value == 0.5
        assert obj.outputs["x"].generation_time == 7

    def test_trigger_handles_multiple_values(self):
        obj = SpeculaInput(output_list=["x:int", "y:int"])

        obj.current_time = 10
        obj.put_input("x", 1)
        obj.put_input("y", 2)
        obj.trigger_code()

        assert obj.outputs["x"].value == 1
        assert obj.outputs["y"].value == 2

    def test_last_value_wins(self):
        obj = SpeculaInput(output_list=["x:int"])

        obj.put_input("x", 1)
        obj.put_input("x", 2)
        obj.trigger_code()

        assert obj.outputs["x"].value == 2

    def test_output_without_type_rejected(self):
        with pytest.raises(ValueError, match="Unsupported type"):
            SpeculaInput(output_list=["x"])

    def test_unknown_output_rejected(self):
        obj = SpeculaInput(output_list=["x:int"])

        with pytest.raises(KeyError, match="Unknown output"):
            obj.put_input("dummy", 5)
        assert obj.q.empty()

    def test_bad_value_rejected(self):
        obj = SpeculaInput(output_list=["x:int"])

        with pytest.raises(ValueError, match="cannot convert to type int"):
            obj.put_input("x", "abc")
        assert obj.q.empty()

    def test_put_input_from_thread(self):
        obj = SpeculaInput(output_list=["x:int"])

        t = threading.Thread(target=obj.put_input, args=("x", 99))
        t.start()
        t.join()
        obj.trigger_code()

        assert obj.outputs["x"].value == 99
