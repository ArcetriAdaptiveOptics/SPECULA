
import os
from astropy.io import fits


class CalibManager():
    def __init__(self, root_dir):
        """
        Initialize the calibration manager object.

        Parameters:
        root_dir (str): Root path of the calibration tree
        """
        super().__init__()
        self._subdirs = {
            'phasescreen': 'phasescreens/',
            'AtmoRandomPhase': 'phasescreens/',
            'AtmoEvolution': 'phasescreens/',
            'AtmoInfiniteEvolution': 'phasescreens/',
            'slopenull': 'slopenulls/',
            'SnCalibrator': 'slopenulls/',
            'sn': 'slopenulls/',
            'background': 'backgrounds/',
            'pupils': 'pupils/',
            'pupdata': 'pupils',
            'PyrPupdataCalibrator': 'pupils/',
            'subapdata': 'subapdata/',
            'ShSubapCalibrator': 'subapdata/',
            'iir_filter_data': 'filter/',
            'IirFilterData': 'filter/',
            'rec': 'rec/',
            'recmat': 'rec/',
            'intmat': 'im/',
            'ImCalibrator': 'im/',
            'MultiImCalibrator': 'im/',
            'projmat': 'rec/',
            'RecCalibrator': 'rec/',
            'MultiRecCalibrator': 'rec/',
            'im': 'im/',
            'ifunc': 'ifunc/',
            'IFunc': 'ifunc/',
            'ifunc_inv': 'ifunc/',
            'm2c': 'm2c/',
            'M2C': 'm2c/',
            'filter': 'filter/',
            'kernel': 'kernels/',
            'pupilstop': 'pupilstop/',
            'Pupilstop': 'pupilstop/',
            'maskef': 'maskef/',
            'TimeHistory': 'data/',
            'time_hist': 'data/',
            'vibrations': 'vibrations/',
            'data': 'data/',
            'projection': 'popt/'
        }
        self.root_dir = root_dir

    def root_subdir(self, type):
        return os.path.join(self.root_dir, self._subdirs[type])

    def filename(self, subdir, name):
        """
        Build the filename for a given subdir and name.
        """
        fname = os.path.join(self.root_dir, self._subdirs[subdir], name)
        if not fname.endswith('.fits'):
            fname += '.fits'
        return fname

    def writefits(self, subdir, name, data):
        """
        Write data to a FITS file.
        """
        filename = self.filename(subdir, name)
        fits.writeto(filename, data, overwrite=True)

    def readfits(self, subdir, name, get_filename=False):
        """
        Read data from a FITS file.
        """
        filename = self.filename(subdir, name)
        if get_filename:
            return filename
        print('Reading:', filename)
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        return fits.getdata(filename)

    def write_data(self, name, data):
        self.writefits('data', name, data)

    def read_data(self, name, get_filename=False):
        return self.readfits('data', name, get_filename)

    def __repr__(self):
        return 'Calibration manager'

