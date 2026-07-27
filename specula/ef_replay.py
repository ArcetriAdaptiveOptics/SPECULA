
from typing import Dict, List, Optional

from specula.base_replay_analyser import BaseReplayAnalyser


class EfReplay(BaseReplayAnalyser):
    """
    Replay a list of existing ElectricField/Layer outputs (e.g.
    'ef_combinator13.out_ef', 'dm_foc_lift.out_layer') exactly as they were
    during a past simulation run, using Simul.build_targeted_replay on the
    objects that produce them.

    Unlike FieldAnalyser, which synthesizes new off-axis Source objects and
    attaches them to an AtmoPropagation object, EfReplay targets object(s)
    that already exist in the original params.yml -- e.g. an
    ElectricFieldCombinator that sums an AtmoPropagation source's output with
    a disturbance (PhaseScreenCube) before it reaches a WFS. Targeting that
    combinator directly, instead of the propagation object alone, pulls in
    everything it depends on (including the disturbance), so the replayed
    field matches what was actually sensed -- something FieldAnalyser cannot
    do for a genuinely new off-axis direction, since such disturbances carry
    no notion of source direction to re-project onto.
    """

    def __init__(self,
                 data_dir: str,
                 tracking_number: str,
                 output_refs: List[str],
                 start_time: float = 0.1,
                 end_time: Optional[float] = None,
                 display: bool = False,
                 log_level: Optional[str] = None,
                 on_missing_downstream_consumers: str = 'error'):

        super().__init__(data_dir, tracking_number, start_time, end_time, display, log_level,
                          on_missing_downstream_consumers)

        self.output_refs = output_refs
        self._targets = sorted({ref.split('.')[0] for ref in output_refs})

        for obj_name in self._targets:
            if obj_name not in self.params:
                raise KeyError(
                    f"Object '{obj_name}' not found in {self.tn_dir / 'params.yml'}")

        self.replay_output_dir = self._make_output_dir('EFREPLAY')

    @staticmethod
    def _ref_to_filename(ref: str) -> str:
        return ref.replace('.', '_')

    def compute_replay(self, force_recompute: bool = False) -> Dict[str, dict]:
        """
        Replay the requested EF/Layer outputs and return their data.

        Returns a dict keyed by output_ref (e.g. 'ef_combinator13.out_ef'),
        each value a dict with 'data' and 'times' (times is None if the
        DataStore did not save a time vector for that output).
        """
        all_exist = True
        if not force_recompute:
            for ref in self.output_refs:
                path = self.replay_output_dir / f"{self._ref_to_filename(ref)}.fits"
                if not path.exists():
                    all_exist = False
                    break

            if all_exist:
                self.logger.debug(f"Loading existing EfReplay results from: {self.replay_output_dir}")
                return self._load_efreplay_results()

        self.logger.debug(f"Computing EfReplay for {self.output_refs}...")

        replay_params = self._build_replay_params_from_datastore(*self._targets)

        input_list = [f'{self._ref_to_filename(ref)}-{ref}' for ref in self.output_refs]
        replay_params['data_store_efreplay'] = {
            'class': 'DataStore',
            'store_dir': str(self.replay_output_dir),
            'data_format': 'fits',
            'create_tn': False,
            'inputs': {
                'input_list': input_list
            }
        }

        _ = self._run_simulation_with_params(replay_params, self.replay_output_dir)

        return self._load_efreplay_results()

    def _load_efreplay_results(self) -> Dict[str, dict]:
        results = {}
        for ref in self.output_refs:
            path = self.replay_output_dir / f"{self._ref_to_filename(ref)}.fits"
            data, times = self._read_fits_primary_and_times(path)
            results[ref] = {'data': data, 'times': times}
        return results
