from specula.log import get_specula_logger

# TODO add astropy units

def n_phot(
        mag,
        band=None,
        surf=1.0,
        delta_t=1.0,
        lambda_=None,
        width=None,
        e0=None,
        back_mag=None,
        model='default',
        ):
    '''
    This routine computes from a source magnitude the corresponding
    number of photons for a given band, a given surface and a given time
    interval. This is returned together with the number of photons from
    the sky background.
    The chosen band can be either a Johnson one (from U to M, but also
    a default Na-band), or a user-defined one (specifying the central
    wavelength and the bandwidth).
    The routine can also return the model used for calculations, if the
    attribute get_model() is called (see examples #4 and #5).

    TODO c in the code is approximated as 3e8

    INPUTS:

     mag  = magnitude [float]

    KEYWORDS:

     band     = wavelength Johnson band or Na default band [str].
     delta_t  = integrating time [s] [float], default is 1s.
     surf     = integrating surface [m^2] [float], default is 1m^2.
     lambda_   = central wavelenght of the choosen band [m] [float].
     width    = bandwidth of the choosen band [m] [float].
     e0       = A0 0-magnitude star brightness in the choosen band,
                [J/s/m^2/um] [float].
     back_mag = sky background default magnitude [FLOAT].
     model    = use the specified model instead the the default one (Lena 96).
                     Available model are: 'MAORY-1'   (phase A of Maory MCAO-EELT)

    OUTPUTS:

     Tuple with 2 elements:

     nb_of_photons = the computed source number of photons [float]
     background    = the computed sky background nb of photons per arcsec^2 [float]

    SIDE EFFECTS:
     none.

    RESTRICTIONS:
     none.

    EXAMPLES:
     [1]: compute the number of photons coming from a 5-mag star
          observed in V-band, with 8m-diameter telescope and an integration
          time of 12s:
 
          nb_of_photons, background = n_phot(5.,
                                             band='V',
                                             surf=math.pi*(8.**2)/4,
                                             delta_t=12.)
 
          returns:
             nb_of_photons = 5.9521160e+10 photons
             background    = 63581.948     photons
 
     [2]: compute the same stuff but with a user-defined sky background
          of 19.5-mag:
 
          nb_of_photons, background = n_phot(5.,
                                             band='V',
                                             surf=math.pi*(8.**2)/4,
                                             delta_t=12.,
                                             back_mag=19.5)
          returns:
             nb_of_photons = 5.9521160e+10 photons
             background    = 100549.18     photons
 
     [3]: compute the same stuff but with a user-defined band of
          central wavelength 0.54um and a narrow bandwidth of 0.01um:
 
          nb_of_photons, background = n_phot(5.,
                                             lambda_=5.4E-7,
                                             width=1E-8,
                                             surf=math.pi*(8.**2)/4,
                                             delta_t=12.,
                                             back_mag=19.5)
          returns:
             nb_of_photons = 6.5661736e+09 photons
             background    = 11092.246     photons
 
     [4]: get the sky background default magnitudes table:

          model = n_phot.getmodel()
          model.back_mag_table
 
 
          returns:
             back_mag_table= [22.,21.,20.,19.,17.5,16.,14.,12.,10.,6.,23.5]

 
     [5]: get the bands, central wavelengths, bandwidths, A0 0-mag. star
          brightnesses, and sky background default magnitudes tables:

          model = n_phot.getmodel()
 
          returns:
             model.band_table    =
 [   "U",   "B",   "V",   "R",   "I",   "J",   "H",   "K",   "L",   "M",    "Na"]
             model.lambda_table  =
 [3.6e-7,4.4e-7,5.5e-7,  7e-7,  9e-7,1.3e-6,1.7e-6,2.2e-6,3.4e-6,  5e-06,5.89e-7]
             model.width_table   =
 [6.8e-8,9.8e-8,8.9e-8,2.2e-7,2.4e-7,  3e-7,3.5e-7,  4e-7,5.5e-7,   3e-7,   1e-8]
             model.e0_table      =
 [4.4e-8,7.2e-8,3.9e-8,1.8e-8,8.3e-9,3.4e-9, 7e-10, 4e-10, 8e-11,2.2e-11, 3.9e-8]
             model.back_mag_table=
 [   23.,   22.,   21.,   20.,  19.5,   14.,  13.5,  12.5,    3.,     0.,    23.]
 
    RESTRICTIONS:
     in the case that user-defined bandwidth (keyword width) and
     wavelength (keyword lambda_) are set, it cannot extend beyond a
     routine-defined band. otherwise the results (nb_of_photons and
     background) are wrong.
 
    MODIFICATION HISTORY:
     program written: october 1998,
                        Marcel Carbillet (OAA) [marcel@arcetri.astro.it].
     modifications  : march 1999,
                        Marcel Carbillet (OAA) [marcel@arcetri.astro.it]:
                         - help completed.
                         - background output stuff added.
                      august 2015
                        Guido Agapito (OAA) [agapito@arcetri.astro.it]
                         - added DOUBLE keyword
                         - added verbose keyword
                         - where and closest like function to find the band/lambda
                      may 2019
                        Alfio Puglisi (OAA) [alfio.puglisi@inaf.it]
                         - translated to Python
                         - removed DOUBLE keyword
                      apr 2026
                         Alfio Puglisi (OAA) [alfio.puglisi@inaf.it]
                            - added logging and replaced print statements with logger calls

    '''
    logger = get_specula_logger(__name__)

    mymodel = get_model(model)

    if band is None and lambda_ is None:
        raise ValueError('n_phot: cannot work if both band and lambda are None')

    # Get idx_band and lambda_ based on model tables,
    # and respect user overrides where possible.

    if band is not None:

        # Band given. Get lambda from table if not given.

        try:
            idx_band = mymodel.band_tab.index(band)
        except ValueError:
            raise Exception('n_phot: band %s not found in table' % band)

        if lambda_ is None:
            lambda_ = mymodel.lambda_tab[idx_band]

    else:

        # Band not given. Use lambda to index lambda table and find band

        if lambda_ < min(mymodel.lambda_tab):
            logger.warning('lambda is < of the smallest element in lambda_tab')
        if lambda_ > max(mymodel.lambda_tab):
            logger.warning('lambda is > of the greater element in lambda_tab')

        # Na band is chosen only if lambda is between 588.9 and 589.1 nm,
        # otherwise the band is searched in the other elements of lambda_tab
        # MAORY-1 model never uses Na.

        useNa = abs(lambda_ - 0.589e-6) < 0.0001e-6 and model != 'MAORY-1'
        if useNa:
            values = mymodel.lambda_tab
        else:
            values = mymodel.lambda_tab[0:-1]

        values = list(map(lambda x: abs(x - lambda_), values))
        idx_band = values.index(min(values))

    logger.debug(f'BAND index number       : {idx_band}')
    logger.debug(f'BAND                    : {mymodel.band_tab[idx_band]}')
    logger.debug(f'BAND CENTRAL WAVELENGTH : {mymodel.lambda_tab[idx_band]}')

    if width is None:
        width = mymodel.width_tab[idx_band]  # bandwidth[m]

    if back_mag is None:
        back_mag = mymodel.back_mag_tab[idx_band]  # sky background magnitude

    if e0 is None:
        e0 = mymodel.e0_tab[idx_band]  # A0 star 0-mag. brightness [J/s/m^2/um]

    h = 6.626e-34  # Planck constant [Js]
    c = 3e8  # light velocity [m/s]

    # source number of photons
    nb_of_photons = lambda_ * delta_t * surf * (width * 1e6) * e0 / (h * c) * 10 ** (-mag / 2.5)

    # sky background nb of photons
    background = lambda_ * delta_t * surf * (width * 1e6) * e0 / (h * c) * 10 ** (-back_mag / 2.5)

    return (nb_of_photons, background)


class DefaultModel():
    '''
    Table of magnitudes for each band.

    ref.: P.Lena, Astrophysique : methodes physiques de l'observation,
            pp.95--96, Coll. Savoirs Actuels, InterEd./CNRS-Ed. (1996)

    except Na-band (added)
    '''

    # bands:
    band_tab = ['U', 'B', 'V', 'R', 'I', 'J', 'H', 'K', 'L', 'M', 'Na']

    # wavelength [um]:
    lambda_tab = \
      [0.36, 0.44, 0.55, 0.70, 0.90, 1.25, 1.65, 2.20, 3.40, 5.00, .589]

    # bandwidth [um]:
    width_tab = \
      [.068, .098, .089, .220, .240, .300, .350, .400, .550, .300, .010]

    # A0 0-magnitude star brightness [J/s/m^2/um] * 1e10
    e0_tab = \
      [435., 720., 392., 176., 83.0, 34.0, 7.00, 3.90, 0.81, 0.22, 392.]

    # default sky background magnitudes:
    back_mag_tab = \
      [23., 22., 21., 20., 19.5, 14., 13.5, 12.5, 3., 0., 23.]

    # list(map()) should work on both python2 and 3
    lambda_tab = list(map(lambda x: x * 1e-6, lambda_tab))
    width_tab = list(map(lambda x: x * 1e-6, width_tab))
    e0_tab = list(map(lambda x: x * 1e-10, e0_tab))


class Maory1Model():
    '''based on E-SPE-ESO-276-0206 issue 1 and MAORY Tech Notes 19 Dec 2007'''

    band_tab = ['V', 'R', 'I', ' J', 'H', 'K']
    lambda_tab = [0.54, 0.65, 0.80, 1.25, 1.65, 2.20]  # 1e-6
    width_tab = [.090, .130, .200, .300, .350, .400]  # 1e-6
    e0_tab = [368., 177., 119., 28.6, 10.8, 3.79]  # 1e-10
    back_mag_tab = [21.4, 20.6, 19.7, 16.5, 14.4, 13.0]

    # list(map()) should work on both python2 and 3
    lambda_tab = list(map(lambda x: x * 1e-6, lambda_tab))
    width_tab = list(map(lambda x: x * 1e-6, width_tab))
    e0_tab = list(map(lambda x: x * 1e-10, e0_tab))

# Add models as function attributes


n_phot.default_model = DefaultModel()
n_phot.maory_model = Maory1Model()


def get_model(model='default'):
    '''Callable function attribute to get the model's tables.'''
    if model == 'default':
        return DefaultModel()

    elif model == 'MAORY-1':
        return Maory1Model()

    else:
        raise Exception('n_phot: unknown model %s' % model)


n_phot.get_model = get_model

# Python2-compatible type hints

n_phot.__annotations__ = {
    'mag': float,
    'band': str,
    'surf': float,
    'delta_t': float,
    'lambda_': float,
    'width': float,
    'e0': float,
    'back_mag': float,
    'model': str,
}
