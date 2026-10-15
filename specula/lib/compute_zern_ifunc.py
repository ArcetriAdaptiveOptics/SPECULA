
from specula.lib.make_mask import make_mask
from specula.lib.zernike_generator import ZernikeGenerator
from specula.lib.utils import make_orto_modes

def compute_zern_ifunc(dim, nzern, xp, dtype, obsratio=0.0, diaratio=1.0, start_mode=0, mask=None):

    if mask is None:
        mask, idx = make_mask(dim, obsratio, diaratio, get_idx=True, xp=xp)
    else:
        mask = mask.astype(float)
        idx = xp.where(mask)

    mask = mask.astype(dtype)

    # Fill the masked pixels one mode at a time, without
    # keeping all full-frame Zernike polynomials in memory
    zg = ZernikeGenerator(dim, xp=xp, dtype=dtype)
    nzern -= start_mode
    zern_phase_2d = xp.empty((nzern, len(idx[0])), dtype=dtype)
    for i in range(nzern):
        zern_phase_2d[i] = zg.getZernike(i + start_mode + 2, cache=False)[idx]
    zg = None

    # Orthonormalize Zernike modes
    zern_phase_2d = make_orto_modes(zern_phase_2d, xp=xp, dtype=dtype)
    # Remove the average phase (piston) from each Zernike mode and normalize them
    zern_phase_2d -= xp.mean(zern_phase_2d, axis=1, keepdims=True)
    zern_phase_2d /= xp.std(zern_phase_2d, axis=1, keepdims=True)

    return zern_phase_2d, mask
