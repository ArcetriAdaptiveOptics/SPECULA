
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from specula.lib.make_mask import make_mask

sn_hdu = fits.open('./calibration/slopenulls/pyr1.0_slope_null.fits')
pyr1_sn = sn_hdu[1].data
plt.figure()
plt.subplot(2,1,1)
plt.plot(pyr1_sn)
plt.grid()
plt.title('Slope nulls')
rec_hdu = fits.open('./calibration/rec/pyr1.0_1200modes_rec.fits')
pyr1_rec = rec_hdu[1].data
sn1_modes = pyr1_rec @ pyr1_sn
x = np.arange(len(sn1_modes))+1
plt.subplot(2,1,2)
plt.plot(x,abs(sn1_modes))
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Slope null modes')
plt.tight_layout()

sn_hdu = fits.open('./calibration/slopenulls/pyr0.0_slope_null.fits')
pyr0_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/pyr0.5_slope_null.fits')
pyr05_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/pyr3.0_slope_null.fits')
pyr3_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/pyr4.0_slope_null.fits')
pyr4_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/pyr6.0_slope_null.fits')
pyr6_sn = sn_hdu[1].data

rec_hdu = fits.open('./calibration/rec/pyr0.0_1200modes_rec.fits')
pyr0_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/pyr0.5_1200modes_rec.fits')
pyr05_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/pyr3.0_1200modes_rec.fits')
pyr3_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/pyr4.0_1200modes_rec.fits')
pyr4_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/pyr6.0_1200modes_rec.fits')
pyr6_rec = rec_hdu[1].data

sn0_modes = pyr0_rec @ pyr0_sn
sn05_modes = pyr05_rec @ pyr05_sn
sn3_modes = pyr3_rec @ pyr3_sn
sn4_modes = pyr4_rec @ pyr4_sn
sn6_modes = pyr6_rec @ pyr6_sn

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(pyr0_sn,label=r'pyWFS 0.0 $\lambda/D$')
# plt.plot(pyr05_sn,label=r'pyWFS 0.5 $\lambda/D$')
# plt.plot(pyr1_sn,label=r'pyWFS 1.0 $\lambda/D$')
# plt.plot(pyr3_sn,label=r'pyWFS 3.0 $\lambda/D$')
# plt.plot(pyr4_sn,label=r'pyWFS 4.0 $\lambda/D$')
# plt.plot(pyr6_sn,label=r'pyWFS 6.0 $\lambda/D$')
# plt.legend()
# plt.grid()
# plt.title('Slope nulls')
# plt.subplot(2,1,2)
# plt.plot(x,abs(sn0_modes),label=r'pyWFS 0.0 $\lambda/D$')
# plt.plot(x,abs(sn05_modes),label=r'pyWFS 0.5 $\lambda/D$')
# plt.plot(x,abs(sn1_modes),label=r'pyWFS 1.0 $\lambda/D$')
# plt.plot(x,abs(sn3_modes),label=r'pyWFS 3.0 $\lambda/D$')
# plt.plot(x,abs(sn4_modes),label=r'pyWFS 4.0 $\lambda/D$')
# plt.plot(x,abs(sn6_modes),label=r'pyWFS 6.0 $\lambda/D$')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.title('Slope null modes')
# plt.tight_layout()

sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_slope_null.fits')
z1wfs_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z1.5wfs_slope_null.fits')
z15wfs_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z2.0wfs_slope_null.fits')
z2wfs_sn = sn_hdu[1].data

plt.figure()
plt.plot(z1wfs_sn)
plt.plot(z15wfs_sn)
plt.plot(z2wfs_sn)
plt.grid()

############################### Pupdata ##########################
# Pyr
npix = 120
np_size = (npix,npix)

pup_hdu = fits.open('./calibration/pupils/pyr_pupdata.fits')
rad = pup_hdu[2].data
cx = pup_hdu[3].data
cy = pup_hdu[4].data
pup_ids = pup_hdu[1].data

pyr_masks = np.zeros(np_size)
for j in range(len(rad)):
    f = np.zeros(npix**2)
    np.put(f, pup_ids[:,j], 1)
    f2d = f.reshape(np_size)
    pyr_masks += f2d

frame_hdu = fits.open('./calibration/slopenulls/pyr0.0_frame.fits') 
pyr0_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr0.5_frame.fits') 
pyr05_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr1.0_frame.fits') 
pyr1_frame = frame_hdu[0].data[0]

# ZWFS
frame_hdu = fits.open('./calibration/slopenulls/z2.0wfs_frame.fits')
z2wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_frame.fits')
z1wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.5wfs_frame.fits')
z15wfs_frame = frame_hdu[0].data[0]

ccd_size = 120
zwfs_mask = make_mask(np_size=ccd_size, diaratio = 40/ccd_size, obsratio=0.0)

masked_frame = lambda frame, mask: frame/frame.max() + mask

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.imshow(masked_frame(pyr0_frame,pyr_masks),origin='lower',cmap='RdBu')
plt.title(r'pyWFS 0.0 $\lambda/D$ pupils'+f'\nPupil diameter = {2*np.mean(rad):1.1f} pix')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(masked_frame(z1wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
plt.title(r'ZWFS 1.0 $\lambda/D$ pupil')
plt.colorbar()

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.imshow(masked_frame(pyr05_frame,pyr_masks),origin='lower',cmap='RdBu')
plt.title(r'pyWFS 0.5 $\lambda/D$ pupils'+f'\nPupil diameter = {2*np.mean(rad):1.1f} pix')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(masked_frame(z15wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
plt.title(r'ZWFS 1.5 $\lambda/D$ pupil')
plt.colorbar()

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.imshow(masked_frame(pyr1_frame,pyr_masks),origin='lower',cmap='RdBu')
plt.title(r'pyWFS 1.0 $\lambda/D$ pupils'+f'\nPupil diameter = {2*np.mean(rad):1.1f} pix')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(masked_frame(z2wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
plt.title(r'ZWFS 2.0 $\lambda/D$ pupil')
plt.colorbar()


###################### Throughput ###########################
frame_hdu = fits.open('./calibration/slopenulls/pyr3.0_frame.fits') 
pyr3_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr4.0_frame.fits') 
pyr4_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr6.0_frame.fits') 
pyr6_frame = frame_hdu[0].data[0]

sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_slope_null.fits')
z1wfs_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z1.5wfs_slope_null.fits')
z15wfs_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z2.0wfs_slope_null.fits')
z2wfs_sn = sn_hdu[1].data

pyr0_thrp = np.sum(pyr0_frame[pyr_masks.astype(bool)])/np.sum(pyr0_frame)
pyr05_thrp = np.sum(pyr05_frame[pyr_masks.astype(bool)])/np.sum(pyr05_frame)
pyr1_thrp = np.sum(pyr1_frame[pyr_masks.astype(bool)])/np.sum(pyr1_frame)
pyr3_thrp = np.sum(pyr3_frame[pyr_masks.astype(bool)])/np.sum(pyr3_frame)
pyr4_thrp = np.sum(pyr4_frame[pyr_masks.astype(bool)])/np.sum(pyr4_frame)
pyr6_thrp = np.sum(pyr6_frame[pyr_masks.astype(bool)])/np.sum(pyr6_frame)

z1wfs_thrp = np.sum(z1wfs_frame[zwfs_mask.astype(bool)])/np.sum(z1wfs_frame)
z15wfs_thrp = np.sum(z15wfs_frame[zwfs_mask.astype(bool)])/np.sum(z15wfs_frame)
z2wfs_thrp = np.sum(z2wfs_frame[zwfs_mask.astype(bool)])/np.sum(z2wfs_frame)

print(pyr0_thrp,pyr05_thrp,pyr1_thrp,pyr3_thrp,pyr4_thrp,pyr6_thrp)
print(z1wfs_thrp,z15wfs_thrp,z2wfs_thrp)
print(np.sum(z1wfs_frame),np.sum(z15wfs_frame),np.sum(z2wfs_frame),np.sum(pyr0_frame),np.sum(pyr6_frame))


########################## Rec ##############################
rec_hdu = fits.open('./calibration/rec/z1.0wfs_1200modes_rec.fits')
z1wfs_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z1.5wfs_1200modes_rec.fits')
z15wfs_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z2.0wfs_1200modes_rec.fits')
z2wfs_rec = rec_hdu[1].data

x = np.arange(np.shape(pyr1_rec)[0])+1
z = np.arange(np.shape(z1wfs_rec)[0])+1

# Rec normalization
def rec_covariance(rec,frame,mask):
    flux = np.sum(frame)
    norm = np.mean(frame[mask.astype(bool)])
    norm_rec = rec / (norm / flux)
    rec_cov = norm_rec @ norm_rec.T
    return np.diag(rec_cov)

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,rec_covariance(pyr0_rec,mask=pyr_masks,frame=pyr0_frame),':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr05_rec,mask=pyr_masks,frame=pyr05_frame),':',label=r'pyWFS 0.5 $\lambda/D$')
plt.plot(x,rec_covariance(pyr1_rec,mask=pyr_masks,frame=pyr1_frame),':',label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr3_rec,mask=pyr_masks,frame=pyr3_frame),':',label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr4_rec,mask=pyr_masks,frame=pyr4_frame),':',label=r'pyWFS 4.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr6_rec,mask=pyr_masks,frame=pyr6_frame),':',label=r'pyWFS 6.0 $\lambda/D$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance')
plt.subplot(1,2,2)
plt.plot(z,rec_covariance(z1wfs_rec,mask=zwfs_mask,frame=z1wfs_frame),':',label='zWFS')
plt.plot(z,rec_covariance(z15wfs_rec,mask=zwfs_mask,frame=z15wfs_frame),':',label='z1.5WFS')
plt.plot(z,rec_covariance(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame),':',label='z2WFS')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance')

plt.show()
