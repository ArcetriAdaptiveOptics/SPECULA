
import os
from astropy.io import fits

from specula.data_objects.pupdata import PupData
from specula.data_objects.recmat import Recmat

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
            'slopenull': 'slopenulls/',
            'sn': 'slopenulls/',
            'background': 'backgrounds/',
            'pupils': 'pupils/',
            'pupdata': 'pupils',
            'subapdata': 'subaps/',
            'ShSubapCalibrator': 'subapdata/',
            'rec': 'rec/',
            'recmat': 'rec/',
            'ImRecCalibrator': 'rec/',
            'MultiImRecCalibrator': 'rec/',
            'im': 'im/',
            'ifunc': 'ifunc/',
            'm2c': 'm2c/',
            'filter': 'filter/',
            'kernel': 'kernels/',
            'pupilstop': 'pupilstop/',
            'Pupilstop': 'pupilstop/',
            'maskef': 'maskef/',
            'vibrations': 'vibrations/',
            'data': 'data/',
            'projection': 'popt/'
        }
        self._root_dir = root_dir

    @property
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value):
        self._root_dir = value

    def root_subdir(self, type):
        return os.path.join(self.root_dir, self._subdirs[type])

    def setproperty(self, root_dir=None, **kwargs):
        """
        Set properties for the calibration manager.

        Parameters:
        root_dir (str, optional): Root path of the calibration tree
        kwargs (dict): Dictionary of additional properties to set
        """
        if root_dir is not None:
            self._root_dir = root_dir

        for key, value in kwargs.items():
            if key in self._subdirs:
                self._subdirs[key] = value
            else:
                print(f"Warning: Property {key} not recognized")

    def joinpath(self, *pieces):
        """
        Join multiple pieces into a single path.
        """
        return os.path.join(*pieces)

    def filename(self, subdir, name):
        """
        Build the filename for a given subdir and name.
        """
        return self.joinpath(self._root_dir, self._subdirs[subdir], name + '.fits')

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

    def write_phasescreen(self, name, data):
        self.writefits('phasescreen', name, data)

    def read_phasescreen(self, name, get_filename=False):
        return self.readfits('phasescreen', name, get_filename)

    def write_slopenull(self, name, data):
        self.writefits('slopenull', name, data)

    def read_slopenull(self, name, get_filename=False):
        return self.readfits('slopenull', name, get_filename)

    def write_background(self, name, data):
        self.writefits('background', name, data)

    def read_background(self, name, get_filename=False):
        return self.readfits('background', name, get_filename)

    def write_pupilstop(self, name, data):
        self.writefits('pupilstop', name, data)

    def read_pupilstop(self, name, get_filename=False):
        return self.readfits('pupilstop', name, get_filename)

    def read_pupils(self, name, get_filename=False):
        filename = self.filename('pupils', name)
        if get_filename:
            return filename
        return PupData.restore(filename)

    def read_rec(self, name, get_filename=False):
        filename = self.filename('rec', name)
        if get_filename:
            return filename
        return Recmat.restore(filename)

    def write_data(self, name, data):
        self.writefits('data', name, data)

    def read_data(self, name, get_filename=False):
        return self.readfits('data', name, get_filename)

    def read_data_ext(self, name, get_filename=False):
        filename = self.filename('data', name)
        if get_filename:
            return filename
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        output = []
        ext = 1
        while True:
            try:
                data = fits.getdata(filename, ext=ext)
                if data is None:
                    break
                output.append(data)
                ext += 1
            except Exception:
                break
        return output

    def write_vibrations(self, name, data):
        self.writefits('vibrations', name, data)

    def read_vibrations(self, name, get_filename=False):
        return self.readfits('vibrations', name, get_filename)

    def __repr__(self):
        return 'Calibration manager'

