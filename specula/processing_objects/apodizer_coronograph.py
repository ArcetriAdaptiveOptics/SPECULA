from specula.processing_objects.abstract_coronograph import Coronograph
from specula.data_objects.simul_params import SimulParams
from specula.lib.make_mask import make_mask
from specula import RAD2ASEC


class APPCoronograph(Coronograph):

    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 pupil,
                 contrastInDarkHole:float,
                 iwaInLambdaOverD:float,
                 owaInLambdaOverD:float,
                 fft_res: float = 3.0,
                 make_symmetric: bool = False,
                 beta: float = 0.9,
                 max_its:int = 1000,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        
        fov = wavelengthInNm * 1e-9 / simul_params.pixel_pitch * RAD2ASEC
        if iwaInLambdaOverD is None:
            iwaInLambdaOverD = 0.0 

        super().__init__(simul_params=simul_params,
                         wavelengthInNm=wavelengthInNm,
                         fov = fov,
                         fft_res=fft_res,
                         target_device_idx=target_device_idx, 
                         precision=precision)
        apodizer_phase, self.target_contrast = self.define_apodizing_phase(pupil, contrastInDarkHole,
                                                          iwaInLambdaOverD, owaInLambdaOverD, beta,
                                                          symmetric_dark_hole=make_symmetric, max_its=max_its)
        self.apodizer = self.xp.exp(1j*apodizer_phase, dtype=self.complex_dtype)
        

    def define_apodizing_phase(self, pupil, contrast,
                               iwa:float, owa:float, beta:float,
                            symmetric_dark_hole:bool=False, 
                            max_its:int=1000):
        target_contrast = self.xp.ones([self.fft_totsize,self.fft_totsize],dtype=self.dtype)
        fp_obsratio = iwa / owa
        fp_diaratio = (owa * self.fft_res * self.fov_res) / self.fft_sampling
        where = make_mask(self.fft_totsize, diaratio=fp_diaratio, obsratio=fp_obsratio, xp=self.xp)
        if symmetric_dark_hole is False:
            xc = (iwa * self.fft_res * self.fov_res)/ self.fft_sampling + 1.0
            left = make_mask(self.fft_totsize, diaratio=1.0, xc=xc, xp=self.xp, square=True)
            where = self.xp.logical_and(where,left)
        target_contrast[where.astype(bool)] = contrast
        pad_start = self.fft_padding//2
        pad_pupil = self.xp.zeros([self.fft_totsize, self.fft_totsize],dtype=self.dtype)
        pad_pupil[pad_start:pad_start+self.fft_sampling, 
                    pad_start:pad_start+self.fft_sampling] = self.xp.array(pupil)
        app = generate_app_keller(pad_pupil, self.xp.array(target_contrast), max_iterations=max_its, beta=beta, xp=self.xp, dtype=self.complex_dtype)
        apodizer_phase = self.xp.zeros(pupil.shape,dtype=self.complex_dtype)
        apodizer_phase[pupil>0] = self.xp.angle(app)[pad_pupil>0.0]
        return apodizer_phase, target_contrast

    def make_focal_plane_mask(self):
        return self.xp.ones([self.fft_totsize,self.fft_totsize],dtype=self.dtype)
    
    def make_pupil_plane_mask(self):
        return self.xp.ones([self.fft_sampling,self.fft_sampling],dtype=self.dtype)
    

class PAPLCoronograph(APPCoronograph):
                 
    def __init__(self,
                 simul_params: SimulParams,
                 wavelengthInNm: float,
                 pupil,
                 contrastInDarkHole:float,
                 iwaInLambdaOverD:float,
                 owaInLambdaOverD:float,
                 fpmIWAInLambdaOverD:float,
                 fpmOWAInLambdaOverD:float=None,
                 knife_edge:bool=True,
                 outerStopAsRatioOfPupil:float=1.0,
                 innerStopAsRatioOfPupil:float=0.0,
                 fft_res: float = 3.0,
                 make_symmetric: bool = False,
                 beta: float = 0.9,
                 target_device_idx: int = None,
                 precision: int = None
                ):
        
        if min(innerStopAsRatioOfPupil,outerStopAsRatioOfPupil) < 0.0 or outerStopAsRatioOfPupil < innerStopAsRatioOfPupil:
            raise ValueError(f'Invalid pupil stop sizes: inner size is'
                             f' {innerStopAsRatioOfPupil*1e+2:1.0f}% of pupil,'
                             f' outer size is {outerStopAsRatioOfPupil*1e+2:1.0f}% of pupil')
        
        if knife_edge is True and owaInLambdaOverD is not None:
            raise ValueError('OWA cannot be defined for the knife-edge focal plane mask')
        
        self._knife_edge = knife_edge
        if knife_edge:
            self._fedge = fpmIWAInLambdaOverD
        else:
            self._iwa = fpmIWAInLambdaOverD
            self._owa = fpmOWAInLambdaOverD

        self._inPupilStop = innerStopAsRatioOfPupil
        self._outPupilStop = outerStopAsRatioOfPupil

        super().__init__(simul_params=simul_params,
                        wavelengthInNm=wavelengthInNm,
                        pupil=pupil,
                        contrastInDarkHole=contrastInDarkHole,
                        iwaInLambdaOverD=iwaInLambdaOverD,
                        owaInLambdaOverD=owaInLambdaOverD,
                        fft_res=fft_res,
                        make_symmetric=make_symmetric,
                        beta=beta,
                        target_device_idx=target_device_idx, 
                        precision=precision)
        
    def make_focal_plane_mask(self):
        if self._knife_edge:
            xc = 2*(self._fedge * self.fft_res + self.fft_totsize//2)/ self.fft_totsize
            fp_mask = make_mask(self.fft_totsize, diaratio=1.0, xc=xc, xp=self.xp, square=True)
        else:
            owa_oversampled = self._owa * self.fft_res if self._owa is not None else self.fft_totsize
            fp_obsratio = self._iwa / owa_oversampled
            fp_diaratio = owa_oversampled / self.fft_totsize 
            fp_mask = make_mask(self.fft_totsize, diaratio=fp_diaratio, obsratio=fp_obsratio, xp=self.xp)
        return fp_mask
    
    def make_pupil_stop(self):
        pp_mask = make_mask(self.fft_sampling, diaratio=self._outPupilStop, obsratio=self._inPupilStop, xp=self.xp)
        return pp_mask


    


# Outside the class on purpose, move inside or to its own module if you prefer
def generate_app_keller(pupil, target_contrast, max_iterations:int, 
                        xp, dtype, beta:float=0):
    """
    Function taken from HCIpy (Por et al. 2018):
    https://github.com/ehpor/hcipy/blob/master/hcipy/coronagraphy/apodizing_phase_plate.py

    Accelerated Gerchberg-Saxton-like algorithm for APP design by
    Christoph Keller [Keller2016]_ and based on Douglas-Rachford operator splitting.
    The acceleration was inspired by the paper by Jim Fienup [Fienup1976]_. The
    acceleration can provide speed-ups of up to two orders of magnitude and
    produce better APPs.

    .. [Keller2016] Keller C.U., 2016, "Novel instrument concepts for
        characterizing directly imaged exoplanets", Proc. SPIE 9908,
        Ground-based and Airborne Instrumentation for Astronomy VI, 99089V
        doi: 10.1117/12.2232633; https://doi.org/10.1117/12.2232633

    .. [Fienup1976] J. R. Fienup, 1976, "Reconstruction of an object from the modulus
        of its Fourier transform," Opt. Lett. 3, 27-29

    Parameters
    ----------
    pupil : ndarray(bool)
        Boolean of the pupil aperture mask.
    target_contrast : ndarray(float)
        The required contrast in the focal plane: float mask that is 1.0
        everywhere except for the dark zone where it is the contrast value (e.g. 1e-5).
    max_iterations : int
        The maximum number of iterations.
    beta : float (optional)
        The acceleration parameter. The default is 0 (no acceleration).
        Good values for beta are typically between 0.3 and 0.9. Values larger
        than 1.0 will not work.

    Returns
    -------
    Wavefront
        The APP as a wavefront.

    Raises
    ------
    ValueError
        If beta is not between 0 and 1.
        If fft_res is less than 3.
    """
    if beta < 0 or beta > 1:
        raise ValueError('Beta should be between 0 and 1.')

    # initialize APP with pupil
    app = pupil * xp.exp(1j*xp.zeros(pupil.shape),dtype=dtype)

    # define dark zone as location where contrast is < 1e-1
    dark_zone = target_contrast < 0.1

    old_image = None
    for i in range(max_iterations):
        image = xp.fft.fftshift(xp.fft.fft2(app)) # calculate image plane electric field

        if not xp.any(xp.abs(image)**2 / xp.max(xp.abs(image)**2) > target_contrast):
            break

        new_image = image.copy()
        if beta != 0 and old_image is not None:
            new_image[dark_zone] = old_image[dark_zone] * beta - new_image[dark_zone] * (1 + beta)
        else:
            new_image[dark_zone] = 0
        old_image = new_image.copy()

        app = xp.fft.ifft2(xp.fft.ifftshift(new_image)) # determine pupil electric field
        app[~pupil.astype(bool)] = 0 # enforce pupil
        # app[pupil.astype(bool)] /= xp.abs(app[pupil.astype(bool)]) # enforce unity transmission within pupil
        app = xp.asarray(pupil) * xp.exp(1j*xp.angle(app),dtype=dtype)
    
    psf = xp.abs(image)**2
    contrast =  psf / xp.max(psf)
    ref_psf = xp.abs(xp.fft.fftshift(xp.fft.fft2(pupil)))**2

    if i == max_iterations-1:
        raise Warning(f'Maximum number of iterations ({max_iterations:1.0f}) reached, worst contrast in dark hole is: {xp.log10(xp.max(contrast[dark_zone])):1.1f}')

    print(f'Apodizer computed: average contrast in dark hole is {xp.mean(xp.log10(contrast[dark_zone])):1.1f}, Strehl is {xp.max(psf)/xp.max(ref_psf)*1e+2:1.2f}%')

    return xp.array(app)

        

    