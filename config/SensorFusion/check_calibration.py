
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
sn_hdu = fits.open('./calibration/slopenulls/pyr0.0_sn_pc_s0.9_c.fits')
pyr0_sn_pc = sn_hdu[1].data
sn_hdu = fits.open('./calibration/slopenulls/pyr3.0_slope_null.fits')
pyr3_sn = sn_hdu[1].data

rec_hdu = fits.open('./calibration/rec/pyr0.0_1200modes_rec.fits')
pyr0_rec = rec_hdu[1].data
rec_hdu = fits.open('./calibration/rec/pyr3.0_1200modes_rec.fits')
pyr3_rec = rec_hdu[1].data

sn0_modes = pyr0_rec @ pyr0_sn
sn0_modes_pc = pyr0_rec @ pyr0_sn_pc
sn3_modes = pyr3_rec @ pyr3_sn


plt.figure()
plt.subplot(2,1,1)
plt.plot(pyr0_sn,label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(abs(pyr0_sn_pc),'--',label=r'pyWFS 0.0 $\lambda/D$ (PC)')
plt.plot(pyr1_sn,label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(pyr3_sn,label=r'pyWFS 3.0 $\lambda/D$')
plt.legend()
plt.grid()
plt.title('Slope nulls')
plt.subplot(2,1,2)
plt.plot(x,abs(sn0_modes),label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,abs(sn0_modes_pc),'--',label=r'pyWFS 0.0 $\lambda/D$ (PC)')
plt.plot(x,abs(sn1_modes),label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(x,abs(sn3_modes),label=r'pyWFS 3.0 $\lambda/D$')
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
frame_hdu = fits.open('./calibration/slopenulls/pyr1.0_frame.fits') 
pyr1_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/pyr3.0_frame.fits') 
pyr3_frame = frame_hdu[0].data[0]

# ZWFS
frame_hdu = fits.open('./calibration/slopenulls/z2.0wfs_frame.fits')
z2wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.0wfs_frame.fits')
z1wfs_frame = frame_hdu[0].data[0]
frame_hdu = fits.open('./calibration/slopenulls/z1.5wfs_frame.fits')
z15wfs_frame = frame_hdu[0].data[0]

ccd_size = 120
zwfs_mask = make_mask(np_size=ccd_size, diaratio = 48/ccd_size, obsratio=0.0)#14)

masked_frame = lambda frame, mask: frame/frame.max() + mask

###################### Throughput ###########################
flux = np.sum(pyr3_frame)

pyr0_thrp = np.sum(pyr0_frame[pyr_masks.astype(bool)])/flux
pyr1_thrp = np.sum(pyr1_frame[pyr_masks.astype(bool)])/flux
pyr3_thrp = np.sum(pyr3_frame[pyr_masks.astype(bool)])/flux

z1wfs_thrp = np.sum(z1wfs_frame[zwfs_mask.astype(bool)])/flux
z15wfs_thrp = np.sum(z15wfs_frame[zwfs_mask.astype(bool)])/flux
z2wfs_thrp = np.sum(z2wfs_frame[zwfs_mask.astype(bool)])/flux

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
plt.imshow(masked_frame(pyr3_frame,pyr_masks),origin='lower',cmap='RdBu')
plt.title(r'pyWFS 3.0 $\lambda/D$ pupils'+f'\nThroughput = {pyr3_thrp*1e+2:1.1f}%')
plt.colorbar()


########################## Rec ##############################

rec_hdu = fits.open('./calibration/rec/z1.0wfs_1200modes_ml_rec.fits')
z1wfs_ml_rec = rec_hdu[1].data

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
pyr3_ron_cov = rec_covariance(pyr3_rec,mask=pyr_masks,frame=pyr3_frame,flux=flux)
pyr0_phot_cov = rec_phot_cov(pyr0_rec,mask=pyr_masks,frame=pyr0_frame,flux=flux,sn=pyr0_sn)
pyr3_phot_cov = rec_phot_cov(pyr3_rec,mask=pyr_masks,frame=pyr3_frame,flux=flux,sn=pyr3_sn)

z1wfs_ron_cov = rec_covariance(z1wfs_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux)
z1wfs_ron_cov_ml = rec_covariance(z1wfs_ml_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux)
z2wfs_ron_cov = rec_covariance(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux)
z1wfs_shot_cov = rec_phot_cov(z1wfs_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux,sn=z1wfs_sn)
z1wfs_shot_cov_ml = rec_phot_cov(z1wfs_ml_rec,mask=zwfs_mask,frame=z1wfs_frame,flux=flux,sn=z1wfs_sn)
z2wfs_shot_cov = rec_phot_cov(z2wfs_rec,mask=zwfs_mask,frame=z2wfs_frame,flux=flux,sn=z2wfs_sn)

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,pyr0_ron_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_covariance(pyr1_rec,mask=pyr_masks,frame=pyr1_frame,flux=flux),':',label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(x,pyr3_ron_cov,':',label=r'pyWFS 3.0 $\lambda/D$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nRON')
plt.subplot(1,2,2)
plt.plot(x,pyr0_phot_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_phot_cov(pyr1_rec,mask=pyr_masks,frame=pyr1_frame,flux=flux,sn=pyr1_sn),':',label=r'pyWFS 1.0 $\lambda/D$')
plt.plot(x,pyr3_phot_cov,':',label=r'pyWFS 3.0 $\lambda/D$')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nshot noise')

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(z,z1wfs_ron_cov,':',label='zWFS')
plt.plot(z,rec_covariance(z15wfs_rec,mask=zwfs_mask,frame=z15wfs_frame,flux=flux),':',label='z1.5WFS')
plt.plot(z,z2wfs_ron_cov,':',label='z2WFS')
plt.plot(z,z1wfs_ron_cov_ml,':',label='zWFS (ML)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nRON')
plt.subplot(1,2,2)
plt.plot(z,z1wfs_shot_cov,':',label='zWFS')
plt.plot(z,rec_phot_cov(z15wfs_rec,mask=zwfs_mask,frame=z15wfs_frame,flux=flux,sn=z15wfs_sn),':',label='z1.5WFS')
plt.plot(z,z2wfs_shot_cov,':',label='z2WFS')
plt.plot(z,z1wfs_shot_cov_ml,':',label='zWFS (ML)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nshot noise')

# PyWFS vs zWFS

plt.figure(figsize=(9,4))
plt.subplot(1,2,1)
plt.plot(x,pyr0_ron_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,pyr3_ron_cov,':',label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(z,z1wfs_ron_cov,':',label='zWFS')
plt.plot(z,z2wfs_ron_cov,':',label='z2WFS')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nRON')
plt.subplot(1,2,2)
plt.plot(x,pyr0_phot_cov,':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,pyr3_phot_cov,':',label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(z,z1wfs_shot_cov,':',label='zWFS')
plt.plot(z,z2wfs_shot_cov,':',label='z2WFS')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.title('Reconstructor covariance\nShot noise')


def rec_noise(shot_cov,ron_cov, RON=0.5, Nphot=1e+6):
    sigma = ron_cov * RON/Nphot**2 + shot_cov / Nphot
    return sigma

RON = 0.5
Nphot = 2e+5
plt.figure()
plt.plot(x,rec_noise(pyr0_phot_cov,pyr0_ron_cov,RON=RON,Nphot=Nphot),':',label=r'pyWFS 0.0 $\lambda/D$')
plt.plot(x,rec_noise(pyr3_phot_cov,pyr3_ron_cov,RON=RON,Nphot=Nphot),':',label=r'pyWFS 3.0 $\lambda/D$')
plt.plot(x,rec_noise(z1wfs_shot_cov,z1wfs_ron_cov,RON=RON,Nphot=Nphot),':',label='zWFS')
plt.plot(x,rec_noise(z2wfs_shot_cov,z2wfs_ron_cov,RON=RON,Nphot=Nphot),':',label='z2WFS')
plt.plot(x,rec_noise(z1wfs_shot_cov_ml,z1wfs_ron_cov_ml,RON=RON,Nphot=Nphot),':',label='zWFS (ML)')
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.grid()
plt.xlabel('KL mode #')
plt.ylabel(r'[$nm^2$]')
plt.title(f'WFS noise\nRON={RON:1.1f}e-, flux={Nphot:1.0e}ph/frame')

plt.show()
