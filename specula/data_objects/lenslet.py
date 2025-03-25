
from specula.lib.make_xy import make_xy
from specula.base_data_obj import BaseDataObj


class Lenslet(BaseDataObj):
    def __init__(self, n_lenses=1, target_device_idx=None, precision=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)
        self.n_lenses = n_lenses
        self._lenses = []

        if n_lenses > 1:
            x, y = make_xy(n_lenses, 1.0, xp=self.xp)
        else:
            x = [0.0]
            y = [0.0]
        
        subap_size = 2.0 / n_lenses

        for i in range(n_lenses):
            row = []
            for j in range(n_lenses):
                row.append([x[i, j], y[i, j], subap_size])
            self._lenses.append(row)

    @property
    def dimx(self):
        return len(self._lenses)

    @property
    def dimy(self):
        return len(self._lenses[0]) if self._lenses else 0

    def get(self, x, y):
        """Returns the subaperture information at (x, y)"""
        return self._lenses[x][y]

    def save(self, filename, hdr):
        """TODO Invalid code. To be updated.

        Saves the lenslet data to a file with the header information"""
        hdr['VERSION'] = 1
        super().save(filename, hdr)
        self.xp.save(filename, self.xp.array(self._lenses))

    def read(self, filename, hdr, exten=0):
        """TODO Invalid code. To be updated.

        Reads lenslet data from a file and updates object state"""
        super().read(filename, hdr, exten)
        self._lenses = self.xp.load(filename, allow_pickle=True).tolist()
        exten += 1

    @classmethod
    def restore(cls, filename):
        """TODO Invalid code. To be updated.

        Restores a lenslet object from a file"""

        p = cls()
        p.read(filename, hdr={})
        return p


