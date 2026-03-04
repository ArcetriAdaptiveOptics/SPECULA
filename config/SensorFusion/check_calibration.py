
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

# from specula.lib.make_mask import make_mask

sn_hdu = fits.open('./calibration/slopenulls/pyr_slope_null.fits')
sn = sn_hdu[1].data
plt.figure()
plt.subplot(2,1,1)
plt.plot(sn)
plt.grid()
plt.title('Slope nulls')
rec_hdu = fits.open('./calibration/rec/pyr_1200modes_rec.fits')
pyr_rec = rec_hdu[1].data
sn_modes = pyr_rec @ sn
plt.subplot(2,1,2)
plt.plot(np.arange(len(sn_modes))+1,abs(sn_modes))
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Slope null modes')
plt.tight_layout()

############################### Pupdata ##########################
npix = 120
np_size = (npix,npix)

pup_hdu = fits.open('./calibration/pupils/pyr_pupdata.fits')
rad = pup_hdu[2].data
cx = pup_hdu[3].data
cy = pup_hdu[4].data
pup_ids = pup_hdu[1].data

pup_masks = np.zeros(np_size)
for j in range(len(rad)):
    f = np.zeros(npix**2)
    np.put(f, pup_ids[:,j], 1)
    f2d = f.reshape(np_size)
    pup_masks += f2d

frame_hdu = fits.open('./calibration/slopenulls/ccd1.fits') #unmod_pyr_frame/ccd.fits')
frame = frame_hdu[0].data[0]

masked_frame = frame/frame.max() + pup_masks

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(masked_frame,origin='lower',cmap='RdBu')
plt.title(f'PWFS pupils\nPupil diameter = {2*np.mean(rad):1.1f} pix')
plt.colorbar()

frame_hdu = fits.open('./calibration/slopenulls/ccd2.fits') #unmod_pyr_frame/ccd.fits')
frame = frame_hdu[0].data[0]

plt.subplot(1,2,2)
plt.imshow(frame,origin='lower',cmap='RdBu')
plt.title(f'ZWFS pupil')
plt.colorbar()

########################## Rec ###############################


rec_hdu = fits.open('./calibration/rec/zwfs_1200modes_rec.fits')
zwfs_rec = rec_hdu[1].data

x = np.arange(np.shape(pyr_rec)[0])
z = np.arange(np.shape(zwfs_rec)[0])

plt.figure()
plt.plot(x,np.diag(pyr_rec @ pyr_rec.T),label='pyWFS')
plt.plot(z,np.diag(zwfs_rec @ zwfs_rec.T),label='z2WFS')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance')



plt.show()
