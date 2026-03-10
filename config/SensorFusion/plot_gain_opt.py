import os
import glob
import yaml
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

output_base = "./output/gain_opt"
dirs = sorted(glob.glob(os.path.join(output_base, "gain_*/2*/")))

gains = []
mean_sr = []

for d in dirs:
    # Find the YAML file to get the gain value
    yml_files = glob.glob(os.path.join(d, "*.yml"))
    gain = None
    for yml in yml_files:
        with open(yml, "r") as f:
            yml_data = yaml.safe_load(f)
            if "filter" in yml_data:
                gain = float(yml_data["filter"]["iir_gain"])
                break
    if gain is None:
        # Fallback: parse from directory name
        gain = float(d.split("_")[-1].replace("/", ""))
    # Load sr.fits
    sr_file = os.path.join(d, "sr.fits")
    if os.path.exists(sr_file):
        with fits.open(sr_file) as hdul:
            sr = hdul[0].data
        mean_sr.append(sr[50:].mean())  # Ignore initial transient
        gains.append(gain)
        print(f"Gain {gain:.2f}: mean SR = {sr[50:].mean():.4f}")
    else:
        print(f"Warning: {sr_file} not found.")

# Plot
plt.figure()
plt.plot(gains, mean_sr, marker='o')
plt.xlabel("IIR Gain")
plt.ylabel("Mean Strehl Ratio")
plt.title("Loop Gain Optimization")
plt.grid(True)
plt.show()