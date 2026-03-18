
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

plt.figure()
plt.subplot(2,1,1)
plt.plot(pyr0_sn,label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(pyr05_sn,label=r'pyWFS 0.5 $\lambda/D$')
plt.plot(pyr1_sn,label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(pyr3_sn,label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(pyr4_sn,label=r'pyWFS 4.0 $\lambda/D$')
plt.plot(pyr6_sn,label=r'pyWFS 6.0 $\lambda/D$')
plt.legend()
plt.grid()
plt.title('Slope nulls')
plt.subplot(2,1,2)
plt.plot(x,abs(sn0_modes),label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,abs(sn05_modes),label=r'pyWFS 0.5 $\lambda/D$')
plt.plot(x,abs(sn1_modes),label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(x,abs(sn3_modes),label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(x,abs(sn4_modes),label=r'pyWFS 4.0 $\lambda/D$')
plt.plot(x,abs(sn6_modes),label=r'pyWFS 6.0 $\lambda/D$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Slope null modes')
plt.tight_layout()

sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_slope_null.fits')
z1wfs_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z1.5wfs_slope_null.fits')
z15wfs_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z2.0wfs_slope_null.fits')
z2wfs_sn = sn_hdu[1].data


rec_hdu = fits.open('./calibration/rec/z1.0wfs_1200modes_rec.fits')
z1wfs_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z1.5wfs_1200modes_rec.fits')
z15wfs_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z2.0wfs_1200modes_rec.fits')
z2wfs_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z2.25wfs_1200modes_rec.fits')
z225wfs_rec = rec_hdu[1].data

z1wfs_sn_modes = z1wfs_rec @ z1wfs_sn
z15wfs_sn_modes = z15wfs_rec @ z15wfs_sn
z2wfs_sn_modes = z2wfs_rec @ z2wfs_sn


plt.figure()
plt.subplot(2,1,1)
plt.plot(z1wfs_sn,label='zWFS')
plt.plot(z15wfs_sn,label='z1.5WFS')
plt.plot(z2wfs_sn,label='z2WFS')
plt.legend()
plt.grid()
plt.title('Slope nulls')
plt.subplot(2,1,2)
plt.plot(x,abs(z1wfs_sn_modes),label='zWFS')
plt.plot(x,abs(z15wfs_sn_modes),label='z1.5WFS')
plt.plot(x,abs(z2wfs_sn_modes),label='z2WFS')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Slope null modes')
plt.tight_layout()

############################### Pupdata ##########################
# Pyr
def get_mask(isPyr:bool=True, Nsubaps:int=48):
    npix = 120
    if isPyr:
        np_size = (npix,npix)
        if Nsubaps == 48:
            pup_hdu = fits.open('./calibration/pupils/pyr_pupdata.fits')
        else:
            pup_hdu = fits.open(f'./calibration/pupils/pyr_pupdata_{Nsubaps:1.0f}x{Nsubaps:1.0f}.fits')
        rad = pup_hdu[2].data
        pup_ids = pup_hdu[1].data
        wfs_mask = np.zeros(np_size)
        for j in range(len(rad)):
            f = np.zeros(npix**2)
            np.put(f, pup_ids[:,j], 1)
            f2d = f.reshape(np_size)
            wfs_mask += f2d
    else:
        wfs_mask = make_mask(np_size=npix, diaratio = Nsubaps/npix, obsratio=0.0) 
    return wfs_mask.astype(bool)

pyr_masks = get_mask(isPyr=True)
zwfs_mask = get_mask(isPyr=False)

frame_hdu = fits.open('./calibration/slopenulls/pyr0.0_frame.fits') 
pyr0_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr0.5_frame.fits') 
pyr05_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr1.0_frame.fits') 
pyr1_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr6.0_frame.fits') 
pyr6_frame = frame_hdu[0].data[0]

# ZWFS
frame_hdu = fits.open('./calibration/slopenulls/z2.0wfs_frame.fits')
z2wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_frame.fits')
z1wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.5wfs_frame.fits')
z15wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.25wfs_frame.fits')
z125wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.75wfs_frame.fits')
z175wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z2.25wfs_frame.fits')
z225wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z5.0wfs_frame.fits')
z5wfs_frame = frame_hdu[0].data[0]

ccd_size = 120
zwfs_mask = make_mask(np_size=ccd_size, diaratio = 48/ccd_size, obsratio=0.0) # xc=0.5/ccd_size, yc=0.5/ccd_size, 

masked_frame = lambda frame, mask: frame/frame.max() + mask

# plt.figure(figsize=(9,4))
# plt.subplot(1,2,1)
# plt.imshow(masked_frame(pyr0_frame,pyr_masks),origin='lower',cmap='RdBu')
# plt.title(r'pyWFS 0.0 $\lambda/D$ pupils'+f'\nPupil diameter = {2*np.mean(rad):1.1f} pix')
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(masked_frame(z1wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 1.0 $\lambda/D$ pupil')
# plt.colorbar()

# plt.figure(figsize=(9,4))
# plt.subplot(1,2,1)
# plt.imshow(masked_frame(pyr05_frame,pyr_masks),origin='lower',cmap='RdBu')
# plt.title(r'pyWFS 0.5 $\lambda/D$ pupils'+f'\nPupil diameter = {2*np.mean(rad):1.1f} pix')
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(masked_frame(z15wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 1.5 $\lambda/D$ pupil')
# plt.colorbar()

# plt.figure(figsize=(9,4))
# plt.subplot(1,2,1)
# plt.imshow(masked_frame(pyr1_frame,pyr_masks),origin='lower',cmap='RdBu')
# plt.title(r'pyWFS 1.0 $\lambda/D$ pupils'+f'\nPupil diameter = {2*np.mean(rad):1.1f} pix')
# plt.colorbar()
# plt.subplot(1,2,2)
# plt.imshow(masked_frame(z2wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 2.0 $\lambda/D$ pupil')
# plt.colorbar()



###################### Throughput ###########################
frame_hdu = fits.open('./calibration/slopenulls/pyr3.0_frame.fits') 
pyr3_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr4.0_frame.fits') 
pyr4_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr6.0_frame.fits') 
pyr6_frame = frame_hdu[0].data[0]


flux = np.sum(pyr6_frame)

pyr0_thrp = np.sum(pyr0_frame[pyr_masks.astype(bool)])/flux
pyr05_thrp = np.sum(pyr05_frame[pyr_masks.astype(bool)])/flux
pyr1_thrp = np.sum(pyr1_frame[pyr_masks.astype(bool)])/flux
pyr3_thrp = np.sum(pyr3_frame[pyr_masks.astype(bool)])/flux
pyr4_thrp = np.sum(pyr4_frame[pyr_masks.astype(bool)])/flux
pyr6_thrp = np.sum(pyr6_frame[pyr_masks.astype(bool)])/flux

z1wfs_thrp = np.sum(z1wfs_frame[zwfs_mask.astype(bool)])/flux
z125wfs_thrp = np.sum(z125wfs_frame[zwfs_mask.astype(bool)])/flux
z15wfs_thrp = np.sum(z15wfs_frame[zwfs_mask.astype(bool)])/flux
z175wfs_thrp = np.sum(z175wfs_frame[zwfs_mask.astype(bool)])/flux
z2wfs_thrp = np.sum(z2wfs_frame[zwfs_mask.astype(bool)])/flux
z225wfs_thrp = np.sum(z225wfs_frame[zwfs_mask.astype(bool)])/flux

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.imshow(masked_frame(z1wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
plt.title(r'ZWFS 1.0 $\lambda/D$ pupil'+f'\nThroughput = {z1wfs_thrp*1e+2:1.1f}%')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(masked_frame(z15wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
plt.title(r'ZWFS 1.5 $\lambda/D$ pupil'+f'\nThroughput = {z15wfs_thrp*1e+2:1.1f}%')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(masked_frame(z2wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
plt.title(r'ZWFS 2.0 $\lambda/D$ pupil'+f'\nThroughput = {z2wfs_thrp*1e+2:1.1f}%')
plt.colorbar()

# plt.figure(figsize=(14,4))
# plt.subplot(1,3,1)
# plt.imshow(masked_frame(z125wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 1.25 $\lambda/D$ pupil'+f'\nThroughput = {z125wfs_thrp*1e+2:1.1f}%')
# plt.colorbar()
# plt.subplot(1,3,2)
# plt.imshow(masked_frame(z175wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 1.75 $\lambda/D$ pupil'+f'\nThroughput = {z175wfs_thrp*1e+2:1.1f}%')
# plt.colorbar()
# plt.subplot(1,3,3)
# plt.imshow(masked_frame(z225wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 2.25 $\lambda/D$ pupil'+f'\nThroughput = {z225wfs_thrp*1e+2:1.1f}%')
# plt.colorbar()

plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plt.imshow(masked_frame(pyr0_frame,pyr_masks),origin='lower',cmap='RdBu')
plt.title(r'pyWFS 0.0 $\lambda/D$ pupils'+f'\nThroughput = {pyr0_thrp*1e+2:1.1f}%')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(masked_frame(pyr1_frame,pyr_masks),origin='lower',cmap='RdBu')
plt.title(r'pyWFS 1.0 $\lambda/D$ pupils'+f'\nThroughput = {pyr1_thrp*1e+2:1.1f}%')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow(masked_frame(pyr6_frame,pyr_masks),origin='lower',cmap='RdBu')
plt.title(r'pyWFS 6.0 $\lambda/D$ pupils'+f'\nThroughput = {pyr6_thrp*1e+2:1.1f}%')
plt.colorbar()


########################## Rec ##############################
rec_hdu = fits.open('./calibration/rec/pyr0.0_1200modes_ml_rec.fits')
pyr0_ml_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z2.0wfs_1200modes_ml_rec.fits')
z2wfs_ml_rec = rec_hdu[1].data
# rec_hdu = fits.open('./calibration/rec/z1.0wfs_1200modes_ml_rec.fits')
# z1wfs_ml_rec = rec_hdu[1].data


x = np.arange(np.shape(pyr1_rec)[0])+1
z = np.arange(np.shape(z1wfs_rec)[0])+1

# Rec normalization
def rec_covariance(rec,frame,mask,flux=None):
    if flux is None:
        flux = np.sum(frame)
    norm = np.mean(frame[mask.astype(bool)])
    norm_rec = rec / (norm / flux)
    rec_cov = norm_rec @ norm_rec.T
    return np.diag(rec_cov)

def rec_phot_cov(rec,frame,mask,sn,flux=None):
    if flux is None:
        flux = np.sum(frame)
    norm = np.mean(frame[mask.astype(bool)])
    phot_noise = np.diag(sn/ (norm / flux))
    rec_cov = rec @ phot_noise @ rec.T
    return np.diag(rec_cov)

pyr0_ron_cov = rec_covariance(pyr0_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux)
pyr0_shot_cov = rec_phot_cov(pyr0_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_sn)

z1wfs_ron_cov = rec_covariance(z1wfs_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux)
z1wfs_shot_cov = rec_phot_cov(z1wfs_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux,sn=z1wfs_sn)

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,pyr0_ron_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr05_rec,mask=pyr_masks,frame=pyr05_frame,flux=flux),':',label=r'pyWFS 0.5 $\lambda/D$')
plt.plot(x,rec_covariance(pyr1_rec,mask=pyr_masks,frame=pyr1_frame,flux=flux),':',label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr3_rec,mask=pyr_masks,frame=pyr3_frame,flux=flux),':',label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr4_rec,mask=pyr_masks,frame=pyr4_frame,flux=flux),':',label=r'pyWFS 4.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr6_rec,mask=pyr_masks,frame=pyr6_frame,flux=flux),':',label=r'pyWFS 6.0 $\lambda/D$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nRON')
plt.subplot(1,2,2)
plt.plot(x,pyr0_shot_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_phot_cov(pyr05_rec,mask=pyr_masks,frame=pyr05_frame,flux=flux,sn=pyr05_sn),':',label=r'pyWFS 0.5 $\lambda/D$')
plt.plot(x,rec_phot_cov(pyr1_rec,mask=pyr_masks,frame=pyr1_frame,flux=flux,sn=pyr1_sn),':',label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(x,rec_phot_cov(pyr3_rec,mask=pyr_masks,frame=pyr3_frame,flux=flux,sn=pyr3_sn),':',label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(x,rec_phot_cov(pyr4_rec,mask=pyr_masks,frame=pyr4_frame,flux=flux,sn=pyr4_sn),':',label=r'pyWFS 4.0 $\lambda/D$')
plt.plot(x,rec_phot_cov(pyr6_rec,mask=pyr_masks,frame=pyr6_frame,flux=flux,sn=pyr6_sn),':',label=r'pyWFS 6.0 $\lambda/D$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nshot noise')

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(z,z1wfs_ron_cov,':',label='zWFS')
plt.plot(z,rec_covariance(z15wfs_rec,mask=zwfs_mask,frame=z15wfs_frame,flux=flux),':',label='z1.5WFS')
plt.plot(z,rec_covariance(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux),':',label='z2WFS')
# plt.plot(z,rec_covariance(z225wfs_rec,mask=zwfs_mask,frame=z225wfs_frame,flux=flux),':',label='z2.25WFS')
# plt.plot(z,rec_covariance(z1wfs_ml_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux),':',label='zWFS (ML)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nRON')
plt.subplot(1,2,2)
plt.plot(z,z1wfs_shot_cov,':',label='zWFS')
plt.plot(z,rec_phot_cov(z15wfs_rec,mask=zwfs_mask,frame=z15wfs_frame,flux=flux,sn=z15wfs_sn),':',label='z1.5WFS')
plt.plot(z,rec_phot_cov(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux,sn=z2wfs_sn),':',label='z2WFS')
# plt.plot(z,rec_phot_cov(z225wfs_rec,mask=zwfs_mask,frame=z225wfs_frame,flux=flux,sn=z2wfs_sn),':',label='z2.25WFS')
# plt.plot(z,rec_phot_cov(z1wfs_ml_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux,sn=z1wfs_sn),':',label='zWFS (ML)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nshot noise')

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,pyr0_ron_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr3_rec,mask=pyr_masks,frame=pyr3_frame,flux=flux),':',label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(z,rec_covariance(z1wfs_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux),':',label='zWFS')
plt.plot(z,rec_covariance(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux),':',label='z2WFS')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nRON')
plt.subplot(1,2,2)
plt.plot(x,pyr0_shot_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_phot_cov(pyr3_rec,mask=pyr_masks,frame=pyr3_frame,flux=flux,sn=pyr3_sn),':',label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(z,rec_phot_cov(z1wfs_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux,sn=z1wfs_sn),':',label='zWFS')
plt.plot(z,rec_phot_cov(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux,sn=z2wfs_sn),':',label='z2WFS')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nShot noise')

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,pyr0_ron_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr0_ml_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label=r'pyWFS 0.0 $\lambda/D$ (ML)')
plt.plot(z,rec_covariance(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux),':',label='z2WFS')
plt.plot(z,rec_covariance(z2wfs_ml_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux),':',label='z2WFS (ML)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nRON')
plt.subplot(1,2,2)
plt.plot(x,pyr0_shot_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_phot_cov(pyr0_ml_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_sn),':',label=r'pyWFS 0.0 $\lambda/D$ (ML)')
plt.plot(z,rec_phot_cov(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux,sn=z2wfs_sn),':',label='z2WFS')
plt.plot(z,rec_phot_cov(z2wfs_ml_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux,sn=z2wfs_sn),':',label='z2WFS (ML)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nShot noise')

################################## NSUBAPS ##############################
sn_hdu = fits.open('./calibration/slopenulls/pyr0.0_36x36_sn.fits')
pyr0_36x36_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/pyr0.0_24x24_sn.fits')
pyr0_24x24_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/pyr0.0_12x12_sn.fits')
pyr0_12x12_sn = sn_hdu[1].data

rec_hdu = fits.open('./calibration/rec/pyr0.0_36x36_600modes_rec.fits')
pyr0_36x36_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/pyr0.0_24x24_300modes_rec.fits')
pyr0_24x24_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/pyr0.0_12x12_75modes_rec.fits')
pyr0_12x12_rec = rec_hdu[1].data

sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_36x36_sn.fits')
z1wfs_36x36_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_24x24_sn.fits')
z1wfs_24x24_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_12x12_sn.fits')
z1wfs_12x12_sn = sn_hdu[1].data

rec_hdu = fits.open('./calibration/rec/z1.0wfs_36x36_600modes_rec.fits')
z1wfs_36x36_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z1.0wfs_24x24_300modes_rec.fits')
z1wfs_24x24_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z1.0wfs_12x12_75modes_rec.fits')
z1wfs_12x12_rec = rec_hdu[1].data


frame_hdu = fits.open('./calibration/slopenulls/pyr0.0_36x36_frame.fits') 
pyr0_36x36_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr0.0_24x24_frame.fits') 
pyr0_24x24_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr0.0_12x12_frame.fits') 
pyr0_12x12_frame = frame_hdu[0].data[0]

frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_36x36_frame.fits') 
z1wfs_36x36_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_24x24_frame.fits') 
z1wfs_24x24_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_12x12_frame.fits') 
z1wfs_12x12_frame = frame_hdu[0].data[0]

pyr_36x36_mask = get_mask(isPyr=True,Nsubaps=36)
pyr_24x24_mask = get_mask(isPyr=True,Nsubaps=24)
pyr_12x12_mask = get_mask(isPyr=True,Nsubaps=12)
zwfs_36x36_mask = get_mask(isPyr=False,Nsubaps=36)
zwfs_24x24_mask = get_mask(isPyr=False,Nsubaps=24)
zwfs_12x12_mask = get_mask(isPyr=False,Nsubaps=12)

print(np.sum(pyr0_36x36_frame),np.sum(pyr0_24x24_frame), np.sum(pyr0_12x12_frame))

pyr0_36x36_thrp = np.sum(pyr0_36x36_frame[pyr_36x36_mask])/flux
pyr0_24x24_thrp = np.sum(pyr0_24x24_frame[pyr_24x24_mask])/flux
pyr0_12x12_thrp = np.sum(pyr0_12x12_frame[pyr_12x12_mask])/flux

z1wfs_36x36_thrp = np.sum(z1wfs_36x36_frame[zwfs_36x36_mask])/flux
z1wfs_24x24_thrp = np.sum(z1wfs_24x24_frame[zwfs_24x24_mask])/flux
z1wfs_12x12_thrp = np.sum(z1wfs_12x12_frame[zwfs_12x12_mask])/flux

print(f'pyWFS 36x36: {pyr0_36x36_thrp*1e+2:1.1f}%, 24x24: {pyr0_24x24_thrp*1e+2:1.1f}%, 12x12: {pyr0_12x12_thrp*1e+2:1.1f}%')
print(f'zWFS 36x36: {z1wfs_36x36_thrp*1e+2:1.1f}%, 24x24: {z1wfs_24x24_thrp*1e+2:1.1f}%, 12x12: {z1wfs_12x12_thrp*1e+2:1.1f}%')

x1800 = np.arange(1800)+1
x1500 = np.arange(1500)+1
x600 = np.arange(600)+1
x300 = np.arange(300)+1
x75 = np.arange(75)+1

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,pyr0_ron_cov,':',label='48x48')
plt.plot(x600,rec_covariance(pyr0_36x36_rec,mask=pyr_36x36_mask,frame=pyr0_36x36_frame,flux=flux),':',label='36x36')
plt.plot(x300,rec_covariance(pyr0_24x24_rec,mask=pyr_24x24_mask,frame=pyr0_24x24_frame,flux=flux),':',label='24x24')
plt.plot(x75,rec_covariance(pyr0_12x12_rec,mask=pyr_12x12_mask,frame=pyr0_12x12_frame,flux=flux),':',label='12x12')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance (RON)\nunmodulated pyramid')
plt.subplot(1,2,2)
plt.plot(x,pyr0_shot_cov,':',label='48x48')
plt.plot(x600,rec_phot_cov(pyr0_36x36_rec,mask=pyr_36x36_mask,frame=pyr0_36x36_frame,flux=flux,sn=pyr0_36x36_sn),':',label='36x36')
plt.plot(x300,rec_phot_cov(pyr0_24x24_rec,mask=pyr_24x24_mask,frame=pyr0_24x24_frame,flux=flux,sn=pyr0_24x24_sn),':',label='24x24')
plt.plot(x75,rec_phot_cov(pyr0_12x12_rec,mask=pyr_12x12_mask,frame=pyr0_12x12_frame,flux=flux,sn=pyr0_12x12_sn),':',label='12x12')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance (shot noise)\nunmodulated pyramid')


plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,z1wfs_ron_cov,':',label='48x48')
plt.plot(x600,rec_covariance(z1wfs_36x36_rec,mask=zwfs_36x36_mask,frame=z1wfs_36x36_frame,flux=flux),':',label='36x36')
plt.plot(x300,rec_covariance(z1wfs_24x24_rec,mask=zwfs_24x24_mask,frame=z1wfs_24x24_frame,flux=flux),':',label='24x24')
plt.plot(x75,rec_covariance(z1wfs_12x12_rec,mask=zwfs_12x12_mask,frame=z1wfs_12x12_frame,flux=flux),':',label='12x12')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance (RON)\nclassical zWFS')
plt.subplot(1,2,2)
plt.plot(x,z1wfs_shot_cov,':',label='48x48')
plt.plot(x600,rec_phot_cov(z1wfs_36x36_rec,mask=zwfs_36x36_mask,frame=z1wfs_36x36_frame,flux=flux,sn=z1wfs_36x36_sn),':',label='36x36')
plt.plot(x300,rec_phot_cov(z1wfs_24x24_rec,mask=zwfs_24x24_mask,frame=z1wfs_24x24_frame,flux=flux,sn=z1wfs_24x24_sn),':',label='24x24')
plt.plot(x75,rec_phot_cov(z1wfs_12x12_rec,mask=zwfs_12x12_mask,frame=z1wfs_12x12_frame,flux=flux,sn=z1wfs_12x12_sn),':',label='12x12')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance (shot noise)\nclassical zWFS')


############################## zWFS phase delay ###############################
rec_hdu = fits.open('./calibration/rec/z1.0wfs_delay0.4_1200modes_rec.fits')
z1wfs_delay04_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z1.0wfs_delay0.3_1200modes_rec.fits')
z1wfs_delay03_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z1.0wfs_delay0.2_1200modes_rec.fits')
z1wfs_delay02_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/z1.0wfs_delay0.1_1200modes_rec.fits')
z1wfs_delay01_rec = rec_hdu[1].data

sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_delay0.4_sn.fits')
z1wfs_delay04_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_delay0.3_sn.fits')
z1wfs_delay03_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_delay0.2_sn.fits')
z1wfs_delay02_sn = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/z1.0wfs_delay0.1_sn.fits')
z1wfs_delay01_sn = sn_hdu[1].data

frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_delay0.4_frame.fits') 
z1wfs_delay04_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_delay0.3_frame.fits') 
z1wfs_delay03_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_delay0.2_frame.fits') 
z1wfs_delay02_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_delay0.1_frame.fits') 
z1wfs_delay01_frame = frame_hdu[0].data[0]

z1wfs_delay04_thrp = np.sum(z1wfs_delay04_frame[zwfs_mask.astype(bool)])/flux
z1wfs_delay03_thrp = np.sum(z1wfs_delay03_frame[zwfs_mask.astype(bool)])/flux
z1wfs_delay02_thrp = np.sum(z1wfs_delay02_frame[zwfs_mask.astype(bool)])/flux
z1wfs_delay01_thrp = np.sum(z1wfs_delay01_frame[zwfs_mask.astype(bool)])/flux

print(f'zWFS delay 0.4: {z1wfs_delay04_thrp*1e+2:1.1f}%, delay 0.3: {z1wfs_delay03_thrp*1e+2:1.1f}%, delay 0.2: {z1wfs_delay02_thrp*1e+2:1.1f}%, delay 0.1: {z1wfs_delay01_thrp*1e+2:1.1f}%')

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,pyr0_ron_cov,':',label=r'unmod pyr')
plt.plot(z,z1wfs_ron_cov,'-.',label=r'0.5 $\pi$')
plt.plot(z,rec_covariance(z1wfs_delay04_rec,mask=zwfs_mask,frame=z1wfs_delay04_frame,flux=flux),'-.',label=r'0.4 $\pi$')
plt.plot(z,rec_covariance(z1wfs_delay03_rec,mask=zwfs_mask,frame=z1wfs_delay03_frame,flux=flux),'-.',label=r'0.3 $\pi$')
plt.plot(z,rec_covariance(z1wfs_delay02_rec,mask=zwfs_mask,frame=z1wfs_delay02_frame,flux=flux),'-.',label=r'0.2 $\pi$')
# plt.plot(z,rec_covariance(z1wfs_delay01_rec,mask=zwfs_mask,frame=z1wfs_delay01_frame,flux=flux),':',label=r'0.1 $\pi$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor cov (RON)\nzWFS vs dot delay')
plt.subplot(1,2,2)
plt.plot(x,pyr0_shot_cov,':',label=r'unmod pyr')
plt.plot(z,z1wfs_shot_cov,'-.',label=r'0.5 $\pi$')
plt.plot(z,rec_phot_cov(z1wfs_delay04_rec,mask=zwfs_mask,frame=z1wfs_delay04_frame,flux=flux,sn=z1wfs_delay04_sn),'-.',label=r'0.4 $\pi$')
plt.plot(z,rec_phot_cov(z1wfs_delay03_rec,mask=zwfs_mask,frame=z1wfs_delay03_frame,flux=flux,sn=z1wfs_delay03_sn),'-.',label=r'0.3 $\pi$')
plt.plot(z,rec_phot_cov(z1wfs_delay02_rec,mask=zwfs_mask,frame=z1wfs_delay02_frame,flux=flux,sn=z1wfs_delay02_sn),'-.',label=r'0.2 $\pi$')
# plt.plot(z,rec_phot_cov(z1wfs_delay01_rec,mask=zwfs_mask,frame=z1wfs_delay01_frame,flux=flux,sn=z1wfs_delay01_sn),':',label=r'0.1 $\pi$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor cov (shot noise)\nzWFS vs dot delay')

# plt.figure(figsize=(18,3.5))
# plt.subplot(1,5,1)
# plt.imshow(masked_frame(z1wfs_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 0.5 $\pi$'+f'\nThroughput = {z1wfs_thrp*1e+2:1.1f}%')
# plt.colorbar()
# plt.subplot(1,5,2)
# plt.imshow(masked_frame(z1wfs_delay04_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 0.4 $\pi$'+f'\nThroughput = {z1wfs_delay04_thrp*1e+2:1.1f}%')
# plt.colorbar()
# plt.subplot(1,5,3)
# plt.imshow(masked_frame(z1wfs_delay03_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 0.3 $\pi$'+f'\nThroughput = {z1wfs_delay03_thrp*1e+2:1.1f}%')
# plt.colorbar()
# plt.subplot(1,5,4)
# plt.imshow(masked_frame(z1wfs_delay02_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 0.2 $\pi$'+f'\nThroughput = {z1wfs_delay02_thrp*1e+2:1.1f}%')
# plt.colorbar()
# plt.subplot(1,5,5)
# plt.imshow(masked_frame(z1wfs_delay01_frame,zwfs_mask),origin='lower',cmap='RdBu')
# plt.title(r'ZWFS 0.1 $\pi$'+f'\nThroughput = {z1wfs_delay01_thrp*1e+2:1.1f}%')
# plt.colorbar()


################################ Nmodes ###############################################

# rec_hdu = fits.open('./calibration/rec/pyr0.0_1800modes_rec.fits')
# pyr0_1800m_rec = rec_hdu[1].data
# rec_hdu = fits.open('./calibration/rec/pyr0.0_1500modes_rec.fits')
# pyr0_1500m_rec = rec_hdu[1].data
# rec_hdu = fits.open('./calibration/rec/pyr0.0_600modes_rec.fits')
# pyr0_600m_rec = rec_hdu[1].data
# rec_hdu = fits.open('./calibration/rec/pyr0.0_300modes_rec.fits')
# pyr0_300m_rec = rec_hdu[1].data
# rec_hdu = fits.open('./calibration/rec/pyr0.0_75modes_rec.fits')
# pyr0_75m_rec = rec_hdu[1].data

# plt.figure(figsize=(9,4))
# plt.subplot(1,2,1)
# # plt.plot(x1500,rec_covariance(pyr0_1500m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label='1500 modes')
# plt.plot(x,pyr0_ron_cov,':',label='1200 modes')
# plt.plot(x600,rec_covariance(pyr0_600m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label='600 modes')
# plt.plot(x300,rec_covariance(pyr0_300m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label='300 modes')
# plt.plot(x75,rec_covariance(pyr0_75m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label='75 modes')
# # plt.plot(x1800,rec_covariance(pyr0_1800m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label='1800 modes')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.title('Reconstructor covariance (RON)\nunmodulated pyramid')
# plt.subplot(1,2,2)
# # plt.plot(x1500,rec_phot_cov(pyr0_1500m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_sn),':',label='1500 modes')
# plt.plot(x,pyr0_shot_cov,':',label='1200 modes')
# plt.plot(x600,rec_phot_cov(pyr0_600m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_sn),':',label='600 modes')
# plt.plot(x300,rec_phot_cov(pyr0_300m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_sn),':',label='300 modes')
# plt.plot(x75,rec_phot_cov(pyr0_75m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_sn),':',label='75 modes')
# # plt.plot(x1800,rec_phot_cov(pyr0_1800m_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_sn),':',label='1800 modes')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.title('Reconstructor covariance (shot noise)\nunmodulated pyramid')


#################### WAVELENGTH #############################
# sn_hdu = fits.open('./calibration/slopenulls/pyr0.0_lambda1000_sn.fits')
# pyr0_lambda1000_sn = sn_hdu[1].data
# sn_hdu = fits.open('./calibration/slopenulls/pyr0.0_lambda1250_sn.fits')
# pyr0_lambda1250_sn = sn_hdu[1].data
# sn_hdu = fits.open('./calibration/slopenulls/pyr0.0_lambda1500_sn.fits')
# pyr0_lambda1500_sn = sn_hdu[1].data

# rec_hdu = fits.open('./calibration/rec/pyr0.0_1200modes_wl1000_rec.fits')
# pyr0_lambda1000_rec = rec_hdu[1].data
# rec_hdu = fits.open('./calibration/rec/pyr0.0_1200modes_wl1250_rec.fits')
# pyr0_lambda1250_rec = rec_hdu[1].data
# rec_hdu = fits.open('./calibration/rec/pyr0.0_1200modes_wl1500_rec.fits')
# pyr0_lambda1500_rec = rec_hdu[1].data

# plt.figure(figsize=(9,4))
# plt.subplot(1,2,1)
# plt.plot(x,pyr0_ron_cov,':',label=r'$\lambda$ = 750nm')
# plt.plot(x,rec_covariance(pyr0_lambda1000_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label=r'$\lambda$ = 1000nm')
# plt.plot(x,rec_covariance(pyr0_lambda1250_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label=r'$\lambda$ = 1250nm')
# plt.plot(x,rec_covariance(pyr0_lambda1500_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux),':',label=r'$\lambda$ = 1500nm')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.title('Reconstructor covariance (RON)\nunmodulated pyramid')
# plt.subplot(1,2,2)
# plt.plot(x,pyr0_shot_cov,':',label=r'$\lambda$ = 750nm')
# plt.plot(x,rec_phot_cov(pyr0_lambda1000_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_lambda1000_sn),':',label=r'$\lambda$ = 1000nm')
# plt.plot(x,rec_phot_cov(pyr0_lambda1250_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_lambda1250_sn),':',label=r'$\lambda$ = 1250nm')
# plt.plot(x,rec_phot_cov(pyr0_lambda1500_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_lambda1500_sn),':',label=r'$\lambda$ = 1500nm')
# plt.legend()
# plt.xscale('log')
# plt.yscale('log')
# plt.grid()
# plt.title('Reconstructor covariance (shot noise)\nunmodulated pyramid')

plt.show()
