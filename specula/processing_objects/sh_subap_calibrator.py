import os

from specula.base_processing_obj import BaseProcessingObj
from specula.data_objects.intensity import Intensity
from specula.data_objects.lenslet import Lenslet
from specula.connections import InputValue
from specula.data_objects.subap_data import SubapData


class ShSubapCalibrator(BaseProcessingObj):
    def __init__(self,
                 subap_on_diameter: int,
                 data_dir: str,         # Set by main simul object
                 energy_th: float,
                 output_tag: str = None,
                 tag_template: str = None,
                 overwrite: bool = False,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self._lenslet = Lenslet(subap_on_diameter, target_device_idx=self.target_device_idx)
        self._energy_th = energy_th
        self._data_dir = data_dir
        if tag_template is None and (output_tag is None or output_tag == 'auto'):
            raise ValueError('At least one of tag_template and output_tag must be set')

        if output_tag is None or output_tag == 'auto':
            self._filename = tag_template
        else:
            self._filename = output_tag
        self._overwrite = overwrite
            
        self.inputs['in_i'] = InputValue(type=Intensity)

    def trigger_code(self):
        image = self.local_inputs['in_i'].i
        self.subaps = self._detect_subaps(image, self._energy_th)

    def finalize(self):
        filename = self._filename
        if not filename.endswith('.fits'):
            filename += '.fits'
        file_path = os.path.join(self._data_dir, filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.subaps.save(os.path.join(self._data_dir, filename), overwrite=self._overwrite)

    def _detect_subaps(self, image, energy_th):
        np = image.shape[0]
        mask_subap = self.xp.zeros_like(image)

        idxs = {}
        map = {}
        spot_intensity = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))
        x = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))
        y = self.xp.zeros((self._lenslet.dimx, self._lenslet.dimy))

        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                lens = self._lenslet.get(i, j)
                x[i, j] = np / 2.0 * (1 + lens[0])
                y[i, j] = np / 2.0 * (1 + lens[1])
                np_sub = round(np / 2.0 * lens[2])

                mask_subap *= 0
                mask_subap[int(self.xp.round(x[i, j] - np_sub / 2)):int(self.xp.round(x[i, j] + np_sub / 2)),
                    int(self.xp.round(y[i, j] - np_sub / 2)):int(self.xp.round(y[i, j] + np_sub / 2))] = 1

                spot_intensity[i, j] = self.xp.sum(image * mask_subap)

        count = 0
        for i in range(self._lenslet.dimx):
            for j in range(self._lenslet.dimy):
                if spot_intensity[i, j] > energy_th * self.xp.max(spot_intensity):
                    mask_subap *= 0
                    mask_subap[int(self.xp.round(x[i, j] - np_sub / 2)):int(self.xp.round(x[i, j] + np_sub / 2)),
                        int(self.xp.round(y[i, j] - np_sub / 2)):int(self.xp.round(y[i, j] + np_sub / 2))] = 1
                    idxs[count] = self.xp.where(mask_subap == 1)
                    map[count] = j * self._lenslet.dimx + i
                    count += 1

        if count == 0:
            raise ValueError("Error: no subapertures selected")

        v = self.xp.zeros((len(idxs), np_sub*np_sub), dtype=int)
        m = self.xp.zeros(len(idxs), dtype=int)
        for k, idx in idxs.items():
            v[k] = self.xp.ravel_multi_index(idx, image.shape)
            m[k] = map[k]

        subap_data = SubapData(idxs=v, display_map=m, nx=self._lenslet.dimx, ny=self._lenslet.dimy, energy_th=energy_th,
                           target_device_idx=self.target_device_idx, precision=self.precision)

        return subap_data
    