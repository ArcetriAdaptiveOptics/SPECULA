from functools import lru_cache

from specula import cpuArray
from specula.base_data_obj import BaseDataObj

from astropy.io import fits
import numpy as np

class SsrFilterData(BaseDataObj):
    """:class:`~specula.data_objects.ssr_filter_data.SsrFilterData` - State Space Representation Filter Data.

    This class stores discrete-time state-space filter coefficients in the format:
    x[k+1] = A*x[k] + B*u[k]
    y[k]   = C*x[k] + D*u[k]
    
    where:
    - A: state transition matrix (n_states x n_states)
    - B: input matrix (n_states x n_inputs)
    - C: output matrix (n_outputs x n_states)
    - D: feedthrough matrix (n_outputs x n_inputs)
    - x: state vector (n_states,)
    - u: input vector (n_inputs,)
    - y: output vector (n_outputs,)
    """

    def __init__(self,
                 A,
                 B,
                 C,
                 D,
                 n_modes=None,
                 target_device_idx: int=None,
                 precision: int=None):
        super().__init__(target_device_idx=target_device_idx, precision=precision)

        # Handle filter setup with n_modes expansion
        if n_modes is not None:
            n_modes = np.atleast_1d(n_modes)

            # Expand matrices for each mode
            A_list = []
            B_list = []
            C_list = []
            D_list = []

            for i, n in enumerate(n_modes):
                for _ in range(n):
                    A_list.append(A[i] if isinstance(A, list) else A)
                    B_list.append(B[i] if isinstance(B, list) else B)
                    C_list.append(C[i] if isinstance(C, list) else C)
                    D_list.append(D[i] if isinstance(D, list) else D)

            self.A = A_list
            self.B = B_list
            self.C = C_list
            self.D = D_list
        else:
            # Store as lists of matrices (one per filter)
            self.A = A if isinstance(A, list) else [A]
            self.B = B if isinstance(B, list) else [B]
            self.C = C if isinstance(C, list) else [C]
            self.D = D if isinstance(D, list) else [D]

        # Convert to appropriate array type and dtype
        self.A = [self.to_xp(a, dtype=self.dtype) for a in self.A]
        self.B = [self.to_xp(b, dtype=self.dtype) for b in self.B]
        self.C = [self.to_xp(c, dtype=self.dtype) for c in self.C]
        self.D = [self.to_xp(d, dtype=self.dtype) for d in self.D]

        # Validate dimensions
        self._validate_dimensions()

    def _validate_dimensions(self):
        """Validate that all matrices have consistent dimensions."""
        for i in range(self.nfilter):
            A_shape = self.A[i].shape
            B_shape = self.B[i].shape
            C_shape = self.C[i].shape
            D_shape = self.D[i].shape

            # Check A is square
            if A_shape[0] != A_shape[1]:
                raise ValueError(f"Filter {i}: A must be square,"
                                 f" got shape {A_shape}")

            n_states = A_shape[0]

            # Check B dimensions
            if B_shape[0] != n_states:
                raise ValueError(f"Filter {i}: B first dimension must"
                                 f" match A dimensions")

            n_inputs = B_shape[1] if len(B_shape) > 1 else 1

            # Check C dimensions
            if C_shape[1] != n_states:
                raise ValueError(f"Filter {i}: C second dimension must"
                                 f" match A dimensions")

            n_outputs = C_shape[0]

            # Check D dimensions
            expected_D_shape = (n_outputs, n_inputs)
            if D_shape != expected_D_shape:
                raise ValueError(f"Filter {i}: D shape {D_shape} doesn't"
                                 f" match expected {expected_D_shape}")

    @property
    def nfilter(self):
        """Number of filters."""
        return len(self.A)

    def get_state_size(self, filter_idx=None):
        """Get the state vector size for a specific filter or all filters."""
        if filter_idx is not None:
            return self.A[filter_idx].shape[0]
        return [a.shape[0] for a in self.A]

    def get_input_size(self, filter_idx=None):
        """Get the input vector size for a specific filter or all filters."""
        if filter_idx is not None:
            return self.B[filter_idx].shape[1] if len(self.B[filter_idx].shape) > 1 else 1
        return [b.shape[1] if len(b.shape) > 1 else 1 for b in self.B]

    def get_output_size(self, filter_idx=None):
        """Get the output vector size for a specific filter or all filters."""
        if filter_idx is not None:
            return self.C[filter_idx].shape[0]
        return [c.shape[0] for c in self.C]

    def save(self, filename):
        """Save filter data to FITS file."""
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NFILTER'] = self.nfilter

        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])

        # Save each filter's matrices
        for i in range(self.nfilter):
            hdul.append(fits.ImageHDU(data=cpuArray(self.A[i]), name=f'A_{i}'))
            hdul.append(fits.ImageHDU(data=cpuArray(self.B[i]), name=f'B_{i}'))
            hdul.append(fits.ImageHDU(data=cpuArray(self.C[i]), name=f'C_{i}'))
            hdul.append(fits.ImageHDU(data=cpuArray(self.D[i]), name=f'D_{i}'))

        hdul.writeto(filename, overwrite=True)
        hdul.close()

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restore filter data from FITS file."""
        with fits.open(filename) as hdul:
            nfilter = hdul[0].header['NFILTER']

            A_list = []
            B_list = []
            C_list = []
            D_list = []

            for i in range(nfilter):
                A_list.append(hdul[f'A_{i}'].data)
                B_list.append(hdul[f'B_{i}'].data)
                C_list.append(hdul[f'C_{i}'].data)
                D_list.append(hdul[f'D_{i}'].data)

        return SsrFilterData(A_list, B_list, C_list, D_list, 
                           target_device_idx=target_device_idx)

    @staticmethod
    def from_gain(gain, target_device_idx=None):
        """Create a simple proportional controller: y[k] = gain * u[k]."""
        gain = np.atleast_1d(gain)
        n = len(gain)

        A_list = []
        B_list = []
        C_list = []
        D_list = []

        for i in range(n):
            # No internal state for pure gain
            A_list.append(np.zeros((1, 1)))
            B_list.append(np.zeros((1, 1)))
            C_list.append(np.zeros((1, 1)))
            D_list.append(np.array([[gain[i]]]))

        return SsrFilterData(A_list, B_list, C_list, D_list,
                           target_device_idx=target_device_idx)

    @staticmethod
    def from_integrator(gain, dt=1.0, target_device_idx=None):
        """Create a discrete integrator: x[k+1] = x[k] + dt*gain*u[k], y[k] = x[k]."""
        gain = np.atleast_1d(gain)
        n = len(gain)

        A_list = []
        B_list = []
        C_list = []
        D_list = []

        for i in range(n):
            # State equation: x[k+1] = x[k] + dt*gain*u[k]
            A_list.append(np.array([[1.0]]))
            B_list.append(np.array([[dt * gain[i]]]))
            # Output equation: y[k] = x[k]
            C_list.append(np.array([[1.0]]))
            D_list.append(np.array([[0.0]]))

        return SsrFilterData(A_list, B_list, C_list, D_list,
                           target_device_idx=target_device_idx)

    def get_eigenvalues(self, filter_idx=None):
        """Get eigenvalues of A matrix for stability analysis."""
        if filter_idx is not None:
            return self.xp.linalg.eigvals(self.A[filter_idx])
        return [self.xp.linalg.eigvals(a) for a in self.A]

    def is_stable(self, filter_idx=None):
        """Check stability: all eigenvalues must be inside unit circle."""
        if filter_idx is not None:
            eigenvalues = self.get_eigenvalues(filter_idx)
            return bool(self.xp.all(self.xp.abs(eigenvalues) < 1.0))
        return [self.is_stable(i) for i in range(self.nfilter)]
