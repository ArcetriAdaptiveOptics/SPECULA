# import specula
import numpy as np
import os
from specula.mmlib.yaml_overrides import write_yaml_overrides

# Range of gains to test
gains = np.linspace(0.1, 1.0, 10) #(0.2, 0.9, 8)#
output_dir = "gain_override"
base_config = "cascading.yml" #"xao_main.yml"
gain_calib = "calib_cascading_gain.yml" #"calib_gain.yml"

for gain in gains:
    print(f'Testing gain {gain}')
    overrides = ("{"
                # f"gainramp.scheduled_values: [[0.2],[{gain:.2f}]], "
                # f"gainramp.scheduled_times: [0.01], "
                # f"data_store.store_dir: '/raid1/mmenessini/results/XAO/gain_opt/gain_{gain:.2f}'"
                f"filter2.int_gain: [{gain:.2f}], "
                f"data_store.store_dir: '/raid1/mmenessini/results/Cascading/gain_opt/gain_{gain:.2f}'"
                "}")
    write_yaml_overrides(input_string=overrides)
    os.system(f"specula {base_config} {gain_calib} temp_overrides.yml")