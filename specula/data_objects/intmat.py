

from astropy.io import fits
from specula import cpuArray

from specula.base_data_obj import BaseDataObj
from specula.data_objects.recmat import Recmat

class Intmat(BaseDataObj):
    def __init__(self,
                 intmat,
                 slope_mm = None,
                 slope_rms = None,
                 pupdata_tag: str = '',
                 norm_factor: float= 0.0,
                 target_device_idx=None,
                 precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._intmat = self.xp.array(intmat)
        self._slope_mm = slope_mm
        self._slope_rms = slope_rms
        self._pupdata_tag = pupdata_tag
        self._norm_factor = norm_factor

    def reduce_size(self, n_modes_to_be_discarded):
        nmodes = self._intmat.shape[0]
        if n_modes_to_be_discarded >= nmodes:
            raise ValueError(f'nModesToBeDiscarded should be less than nmodes (<{nmodes})')
        self._intmat = self._intmat[:nmodes - n_modes_to_be_discarded, :]

    def reduce_slopes(self, n_slopes_to_be_discarded):
        nslopes = self._intmat.shape[1]
        if n_slopes_to_be_discarded >= nslopes:
            raise ValueError(f'nSlopesToBeDiscarded should be less than nslopes (<{nslopes})')
        self._intmat = self._intmat[:, :nslopes - n_slopes_to_be_discarded]

    def set_start_mode(self, start_mode):
        nmodes = self._intmat.shape[0]
        if start_mode >= nmodes:
            raise ValueError(f'start_mode should be less than nmodes (<{nmodes})')
        self._intmat = self._intmat[start_mode:, :]

    def save(self, filename, hdr=None):
        if not filename.endswith('.fits'):
            filename += '.fits'
        if hdr is None:
            hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['PUP_TAG'] = self._pupdata_tag
        hdr['NORMFACT'] = self._norm_factor
        # Save fits file
        fits.writeto(filename, cpuArray(self._intmat), hdr, overwrite=True)
        if self._slope_mm is not None:
            fits.append(filename, self._slope_mm)
        if self._slope_rms is not None:
            fits.append(filename, self._slope_rms)

    @staticmethod
    def restore(filename, hdr=None, exten=0, target_device_idx=None):
        intmat = fits.getdata(filename, ext=exten)
        hdr = fits.getheader(filename, ext=exten)
        norm_factor = float(hdr.get('NORMFACT', 0.0))
        pupdata_tag = hdr.get('PUP_TAG', '')
        # Reading additional fits extensions
        num_ext = len(fits.open(filename))
        if num_ext >= exten + 2:
            slope_mm = fits.getdata(filename, ext=exten + 1)
            slope_rms = fits.getdata(filename, ext=exten + 2)
        else:
            slope_mm = slope_rms = None
        return Intmat(intmat, slope_mm, slope_rms, pupdata_tag, norm_factor, target_device_idx=target_device_idx)

    def generate_rec(self, nmodes=None, cut_modes=0, w_vec=None, interactive=False):
        if nmodes is not None:
            intmat = self._intmat[:nmodes, :]
        else:
            intmat = self._intmat
        recmat = self.pseudo_invert(intmat, n_modes_to_drop=cut_modes, w_vec=w_vec, interactive=interactive)
        rec = Recmat(recmat)
        rec.im_tag = self._norm_factor
        return rec

    def pseudo_invert(self, matrix, n_modes_to_drop=0, w_vec=None, interactive=False):
        # TODO handle n_modes_to_drop, and w_vec
        return self.xp.linalg.pinv(matrix)

    def build_from_slopes(self, slopes, disturbance):
        times = list(slopes.keys())
        nslopes = len(slopes[times[0]])
        nmodes = len(disturbance[times[0]])
        intmat = self.xp.zeros((nmodes, nslopes), dtype=self.dtype)
        iter_per_mode = self.xp.zeros(nmodes, dtype=self.dtype)
        slope_mm = self.xp.zeros((nmodes, 2), dtype=self.dtype)
        slope_rms = self.xp.zeros(nmodes, dtype=self.dtype)

        for t in times:
            amp = disturbance[t]
            mode = self.xp.where(amp)[0][0]
            intmat[mode, :] += slopes[t] / amp[mode]
            iter_per_mode[mode] += 1

        for m in range(nmodes):
            if iter_per_mode[m] > 0:
                intmat[m, :] /= iter_per_mode[m]

        im = Intmat(intmat)
        im._slope_mm = slope_mm
        im._slope_rms = slope_rms
        return im
