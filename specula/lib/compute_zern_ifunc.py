
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

    zg = ZernikeGenerator(dim, xp=xp, dtype=dtype)
    zern_phase_3d = xp.stack([zg.getZernike(z) for z in range(2, nzern + 2)])
    zern_phase_3d = zern_phase_3d[start_mode:]
    nzern -= start_mode

    zern_phase_2d = xp.array([zern_phase_3d[i][idx] for i in range(nzern)], dtype=dtype)

    # Orthonormalize Zernike modes
    zern_phase_2d = make_orto_modes(zern_phase_2d, xp=xp, dtype=dtype)
    # Remove the average phase (piston) from each Zernike mode and normalize them
    zern_phase_2d = zern_phase_2d - xp.mean(zern_phase_2d, axis=1, keepdims=True)
    zern_phase_2d = zern_phase_2d / xp.std(zern_phase_2d, axis=1, keepdims=True)

    return zern_phase_2d, mask

