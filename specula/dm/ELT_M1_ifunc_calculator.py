import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from specula.data_objects.ifunc import IFunc

class ELTM1IFuncCalculator:


    def __init__(self, dim=512, dtype=np.float32):

        self.dim = dim
        self.dtype = dtype
        self.ifs_cube = None

    def M1_modal_base(self):
        segmentation = fits.open(os.path.join(os.path.dirname(__file__), '../data/EltM1SegmMap1015pix38570mm.fits'))[0].data.copy()

        rescalingFactor = segmentation.shape[0]/self.dim
        coord = np.round(np.arange(self.dim)*rescalingFactor).astype(int)
        rsegmentation = np.zeros((self.dim,self.dim))
        for a in range(self.dim):
            for b in range(self.dim):
                rsegmentation[a,b]=segmentation[coord[a],coord[b]]
        segm = rsegmentation.copy().astype(self.dtype)

        pupil = segm>0
        idx_pupil = np.where(pupil)        
        tilt,tip = np.meshgrid(np.linspace(-1,1,segm.shape[0]),
                            np.linspace(-1,1,segm.shape[1]))

        M1Base=[]
        for s in range(int(np.max(segmentation))):
            #generate the piston
            pist_s = segm==(s+1)
            idx_s = np.where(pist_s)
            pist_s = pist_s.astype(self.dtype)
            pist_s[idx_s] = pist_s[idx_s] / np.sqrt(np.mean((pist_s[idx_s])**2))
            M1Base.append(pist_s[idx_pupil].copy())

            #generate the segment tip
            tip_s = tip*pist_s
            tip_s[idx_s]*= 1/np.std(tip_s[idx_s])#normalise to 1 (choose your unit) RMS
            tip_s[idx_s]-= np.mean(tip_s[idx_s])#remove the average offset
            M1Base.append(tip_s[idx_pupil].copy())

            #generate the segment tilt
            tilt_s = tilt*pist_s
            tilt_s[idx_s]*=1/np.std(tilt_s[idx_s])#normalise to 1 (choose your unit) RMS
            tilt_s[idx_s]-=np.mean(tilt_s[idx_s])#remove the average offset
            M1Base.append(tilt_s[idx_pupil].copy())

        self.ifs_cube = np.asarray(M1Base, dtype=self.dtype).T
        self.mask = pupil.astype(self.dtype)

    def save_results(self, filename):
        self.M1_modal_base()

        # Salva la maschera e ifs_2d usando la classe IFunc
        ifunc = IFunc(ifunc=self.ifs_cube, mask=self.mask, target_device_idx=-1, precision=0)
        ifunc.save(filename)

    def plot_results(self):
        #Nr of total modes = 3x798 (= ptt for each segment)
        n_modes = self.ifs_cube.shape[1]
        mb = np.zeros((self.dim, self.dim, n_modes)) 
        mb[np.where(self.mask>0)] = self.ifs_cube

        fig, ax = plt.subplots(3, 3)
        fig.suptitle('First three segments')
        for i, a in enumerate(ax.flatten()):
            a.imshow(mb[:, :, i], origin='lower')

# Example usage of the class
# from specula.dm.ELT_M1_ifunc_calculator import ELTM1IFuncCalculator
# dim = 480  # Pupil dimension
# calculator = ELTM1IFuncCalculator(dim)
# calculator.save_results('~/ifunc_elt_m1.fits')
# calculator.plot_results()