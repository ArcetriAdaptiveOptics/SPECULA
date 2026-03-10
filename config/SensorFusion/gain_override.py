import specula
import numpy as np

# Range of gains to test
gains = np.linspace(0.1, 1.0, 10)
output_dir = "gain_override"
base_config = "xao_main.yml"

for gain in gains:
    overrides = ("{"
                "main.total_time: 0.5, "
                f"filter.iir_gain: {gain:.2f}, "
                f"data_store.store_dir: ./output/gain_opt/gain_{gain:.2f}"
                "}")

    specula.main_simul(yml_files=[base_config], overrides=overrides)