from astropy.io import fits
import numpy as np

from specula import cpuArray
from specula.base_data_obj import BaseDataObj


class ElectricField(BaseDataObj):
    '''Electric field'''

    def __init__(self,
                 dimx: int,
                 dimy: int,
                 pixel_pitch: float,
                 S0: float=0.0,
                 target_device_idx: int=None,
                 precision: int=None):
        """
        ElectricField data object.

        This class represents a 2D electric field, storing both amplitude and phase information
        for each pixel in a rectangular grid. The field is stored as a 2x(dimx)x(dimy) array,
        where the first index (0) corresponds to amplitude and the second index (1) to phase (in nm).

        Parameters
        ----------
        dimx : int
            Number of pixels along the x-axis (width).
        dimy : int
            Number of pixels along the y-axis (height).
        pixel_pitch : float
            The dimension in meters of a pixel.
        S0 : float, optional
            Flux density in photons/s/m^2/nm (default: 0.0).
        target_device_idx : int, optional
            Device index for computation (default: None).
        precision : int, optional
            Precision for computation (default: None).

        Attributes
        ----------
        pixel_pitch : float
            The pixel pitch in meters.
        S0 : float
            Optional parameter for the field.
        field : xp.ndarray
            The electric field array of shape (2, dimx, dimy), with amplitude and phase.
        """
        super().__init__(precision=precision, target_device_idx=target_device_idx)
        dimx = int(dimx)
        dimy = int(dimy)
        self.pixel_pitch = pixel_pitch
        self.S0 = S0
        A = self.xp.ones((dimx, dimy), dtype=self.dtype)
        phaseInNm = self.xp.zeros((dimx, dimy), dtype=self.dtype)
        self.field = self.xp.stack((A, phaseInNm))
    
    @property
    def A(self):
        return self.field[0]

    @A.setter
    def A(self, value):
        self.field[0, :, :] = self.to_xp(value)

    @property
    def phaseInNm(self):
        return self.field[1]

    @phaseInNm.setter
    def phaseInNm(self, value):
        self.field[1, :, :] = self.to_xp(value)

    def __str__(self):
        return 'A: '+ str(self.field[0]) + 'Phase: ' + str(self.field[1])

    def set_value(self, v):
        '''
        Set new values for phase and amplitude
        
        Arrays are not reallocated
        '''
        # Should not expect a list, but a 2xNxN array

        #assert len(v) == 2, "Input must be a sequence of [amplitude, phase]"
        assert v[0].shape == self.field[0].shape, \
            f"Error: input array shape {v[0].shape} does not match amplitude shape {self.field[0].shape}"
        assert v[1].shape == self.phaseInNm.shape, \
            f"Error: input array shape {v[1].shape} does not match phase shape {self.field[1].shape}"

        self.field[:] = self.to_xp(v)

    def get_value(self):
        return self.field

    def reset(self):
        '''
        Reset to zero phase and unitary amplitude
        
        Arrays are not reallocated
        '''
        self.field[0] *= 0
        self.field[0] += 1
        self.field[1] *= 0

    def resize(self, dimx, dimy, pitch=None):
        '''
        Resize the electric field
        
        The pixel pitch and S0 are not changed
        '''
        dimx = int(dimx)
        dimy = int(dimy)
        self.field = self.xp.zeros((2, dimx, dimy), dtype=self.dtype)
        if pitch is not None:
            self.pixel_pitch = pitch
        self.reset()

    @property
    def size(self):
        return self.field[0].shape

    def checkOther(self, ef2, subrect=None):
        if not isinstance(ef2, ElectricField):
            raise ValueError(f'{ef2} is not an ElectricField instance')
        if subrect is None:
            subrect = [0, 0]
        diff0 = self.size[0] - subrect[0]
        diff1 = self.size[1] - subrect[1]
        if ef2.size[0] != diff0 or ef2.size[1] != diff1:
            raise ValueError(f'{ef2} has size {ef2.size} instead of the required ({diff0}, {diff1})')
        return subrect

    def phi_at_lambda(self, wavelengthInNm, slicey=None, slicex=None):
        """
        Calculate the phase of the electric field at a given wavelength.

        Parameters
        ----------
        wavelengthInNm : float
            The wavelength in nanometers.
        slicey : slice, optional
            The slice along the y-axis (default: None, which means all rows).
        slicex : slice, optional
            The slice along the x-axis (default: None, which means all columns).

        Returns
        -------
        xp.ndarray
            The phase of the electric field at the given wavelength.
        """
        if slicey is None:
            slicey = np.s_[:]
        if slicex is None:
            slicex = np.s_[:]
        return self.field[1,slicey, slicex] * ((2 * self.xp.pi) / wavelengthInNm)

    def ef_at_lambda(self, wavelengthInNm, slicey=None, slicex=None, out=None):
        """
        Calculate the electric field at a given wavelength.

        Parameters
        ----------
        wavelengthInNm : float
            The wavelength in nanometers.
        slicey : slice, optional
            The slice along the y-axis (default: None, which means all rows).
        slicex : slice, optional
            The slice along the x-axis (default: None, which means all columns).
        out : xp.ndarray, optional
            The output array (default: None).

        Returns
        -------
        xp.ndarray
            The electric field at the given wavelength. If out is provided, the result is
            both stored in out and and the same reference is returned.
        """
        if slicey is None:
            slicey = np.s_[:]
        if slicex is None:
            slicex = np.s_[:]
        phi = self.phi_at_lambda(wavelengthInNm, slicey=slicey, slicex=slicex)
        ef = self.xp.exp(1j * phi, dtype=self.complex_dtype, out=out)
        ef *= self.field[0, slicey, slicex]
        return ef

    def product(self, ef2, subrect=None):
        """
        Multiply the electric field by another electric field.

        Parameters
        ----------
        ef2 : ElectricField
            The other electric field.
        subrect : tuple, optional
            The subrectangle to multiply (top-left coordinate into ef2. Default: None, which means all of ef2).
        """
#        subrect = self.checkOther(ef2, subrect=subrect)    # TODO check subrect from atmo_propagation, even in PASSATA it does not seem right
        x2 = subrect[0] + self.size[0]
        y2 = subrect[1] + self.size[1]
        self.field[0] *= ef2.field[0, subrect[0] : x2, subrect[1] : y2]
        self.field[1] += ef2.field[1, subrect[0] : x2, subrect[1] : y2]

    def area(self):
        """
        Calculate the area of the electric field.
        """
        return self.field[0].size * (self.pixel_pitch ** 2)

    def masked_area(self):
        tot = self.xp.sum(self.field[0])
        return (self.pixel_pitch ** 2) * tot

    def square_modulus(self, wavelengthInNm):
        """
        Compute the square modulus (intensity) of the electric field at a given wavelength.

        Parameters
        ----------
        wavelengthInNm : float
            The wavelength in nanometers at which to compute the square modulus.

        Returns
        -------
        xp.ndarray
            The square modulus (intensity) of the electric field at the specified wavelength.
        """
        ef = self.ef_at_lambda(wavelengthInNm)
        return self.xp.real( ef * self.xp.conj(ef) )

    def sub_ef(self, xfrom=None, xto=None, yfrom=None, yto=None, idx=None):
        """
        Extract a subregion of the electric field.

        Parameters
        ----------
        xfrom : int, optional
            Starting index along the x-axis (inclusive).
        xto : int, optional
            Ending index along the x-axis (exclusive).
        yfrom : int, optional
            Starting index along the y-axis (inclusive).
        yto : int, optional
            Ending index along the y-axis (exclusive).
        idx : array-like, optional
            Indices to extract as a subregion. If provided, xfrom/xto/yfrom/yto are ignored.

        Returns
        -------
        ElectricField
            A new ElectricField object representing the extracted subregion.
        """
        if idx is not None:
            idx = self.xp.unravel_index(idx, self.field[0].shape)
            xfrom, xto = self.xp.min(idx[0]), self.xp.max(idx[0])
            yfrom, yto = self.xp.min(idx[1]), self.xp.max(idx[1])
        sub_ef = ElectricField(xto - xfrom, yto - yfrom, self.pixel_pitch, target_device_idx=self.target_device_idx)
        sub_ef.field[0, :] = self.field[0, xfrom:xto, yfrom:yto]
        sub_ef.field[1, :] = self.field[1, xfrom:xto, yfrom:yto]
        sub_ef.S0 = self.S0
        return sub_ef

    def compare(self, ef2):
        """
        Compare this ElectricField object with another ElectricField object.

        Parameters
        ----------
        ef2 : ElectricField
            The ElectricField object to compare with.

        Returns
        -------
        bool
            True if the fields are different, False if they are equal.
        """
        return not (self.xp.array_equal(self.field, ef2.field))

    def get_fits_header(self):
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['OBJ_TYPE'] = 'ElectricField'
        hdr['DIMX'] = self.field[0].shape[0]
        hdr['DIMY'] = self.field[0].shape[1]
        hdr['PIXPITCH'] = self.pixel_pitch
        hdr['S0'] = self.S0
        return hdr

    def save(self, filename, overwrite=True):
        hdr = self.get_fits_header()
        hdu = fits.PrimaryHDU(header=hdr)  # main HDU, empty, only header
        hdul = fits.HDUList([hdu])
        hdul.append(fits.ImageHDU(data=cpuArray(self.field[0]), name='AMPLITUDE'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.field[1]), name='PHASE'))
        hdul.writeto(filename, overwrite=overwrite)
        hdul.close()  # Force close for Windows

    @staticmethod
    def from_header(hdr, target_device_idx=None):
        version = hdr['VERSION']
        if version != 1:
            raise ValueError(f"Error: unknown version {version} in header")
        dimx = hdr['DIMX']
        dimy = hdr['DIMY']
        pitch = hdr['PIXPITCH']
        S0 = hdr['S0']
        ef = ElectricField(dimx, dimy, pitch, S0, target_device_idx=target_device_idx)
        return ef

    @staticmethod
    def restore(filename, target_device_idx=None):
        hdr = fits.getheader(filename)
        if 'OBJ_TYPE' not in hdr or hdr['OBJ_TYPE'] != 'ElectricField':
            raise ValueError(f"Error: file {filename} does not contain an ElectricField object")
        ef = ElectricField.from_header(hdr, target_device_idx=target_device_idx)
        with fits.open(filename) as hdul:
            ef.field[0] = ef.to_xp(hdul[1].data.copy())   # pylint: disable=no-member # created dyamically by pyfits
            ef.field[1] = ef.to_xp(hdul[2].data.copy())   # pylint: disable=no-member # created dyamically by pyfits
        return ef

    def array_for_display(self):
        """
        Returns a 2D array suitable for display: the phase (in nm) masked by the amplitude.

        The returned array is the phase array (self.field[1]) where the amplitude (self.field[0]) is greater than zero.
        The average phase (over nonzero amplitude pixels) is subtracted for visualization purposes.

        Returns
        -------
        frame : xp.ndarray
            2D array of phase values (in nm), mean-subtracted over nonzero amplitude pixels.
        """
        frame = self.field[1] * (self.field[0] > 0).astype(float)
        idx = self.xp.where(self.field[0] > 0)[0]
        # Remove average phase
        frame[idx] -= self.xp.mean(frame[idx])
        return frame
