
import os
from astropy.io import fits

from iteras import registry
from iteras.log import get_iteras_logger


class CalibManager():
    def __init__(self, root_dir):
        """
        Initialize the calibration manager object.

        Parameters:
        root_dir (str): Root path of the calibration tree

        Files are stored in a subdirectory of root_dir depending on their
        data type (or on the class that computes them). Plugins register
        their subdirectories with iteras.registry.register_calib_subdirs().
        """
        super().__init__()
        # Subdirectories of the data types, extended by plugins (see iteras.registry)
        self._subdirs = registry.calib_subdirs()
        self.root_dir = root_dir
        self.logger = get_iteras_logger('iteras.calib_manager')

    def root_subdir(self, type):
        """
        Returns the full path to the subdirectory corresponding to the given type.

        Parameters:
            type (str): The key representing the calibration data type.

        Returns:
            str: The absolute path to the subdirectory for the specified type.
        """
        return os.path.join(self.root_dir, self._subdirs[type])

    def filename(self, subdir, name):
        """
        Construct the full file path for a calibration file.

        Parameters:
            subdir (str): The key representing the calibration data type or subdirectory.
            name (str): The base name of the file (without extension).

        Returns:
            str: The absolute path to the FITS file, ensuring the '.fits' extension is present.
        """
        fname = os.path.join(self.root_dir, self._subdirs[subdir], name)
        if not fname.endswith('.fits'):
            fname += '.fits'
        return fname

    def writefits(self, subdir, name, data):
        """
        Write data to a FITS file.

        Parameters:
            subdir (str): The key representing the calibration data type or subdirectory.
            name (str): The base name of the file (without extension).
            data (array-like): The data to be written to the file.
        """
        filename = self.filename(subdir, name)
        fits.writeto(filename, data, overwrite=True)

    def readfits(self, subdir, name):
        """
        Read data from a FITS file.

        Parameters:
            subdir (str): The key representing the calibration data type or subdirectory.
            name (str): The base name of the file (without extension).

        Returns:
            array-like: The data read from the file.
        """
        filename = self.filename(subdir, name)
        self.logger.info(f'Reading: {filename}')
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)
        return fits.getdata(filename)

    def write_data(self, name, data):
        """
        Write data to a FITS file in the 'data' subdirectory.

        Parameters:
            name (str): The base name of the file (without extension).
            data (array-like): The data to be written to the file.
        """
        self.writefits('data', name, data)

    def read_data(self, name):
        """
        Read data from a FITS file in the 'data' subdirectory.

        Parameters:
            name (str): The base name of the file (without extension).

        Returns:
            array-like: The data read from the file.
        """
        return self.readfits('data', name)

    def __repr__(self):
        return 'Calibration manager'

