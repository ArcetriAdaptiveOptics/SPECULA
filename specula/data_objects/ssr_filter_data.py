from specula import cpuArray
from specula.base_data_obj import BaseDataObj
from scipy.linalg import block_diag
from astropy.io import fits
import numpy as np

class SsrFilterData(BaseDataObj):
    """:class:`~specula.data_objects.ssr_filter_data.SsrFilterData`
    State Space Representation Filter Data.

    This class stores discrete-time state-space filter coefficients in the format:
    x[k+1] = A*x[k] + B*u[k]
    y[k]   = C*x[k'] + D*u[k]
    
    where x[k'] is either x[k] or x[k+1] depending on output_uses_new_state argument
    of SsrFilter class.
    
    All filters are combined into single block-diagonal matrices:
    - A: block-diagonal state transition matrix (total_states x total_states)
    - B: input matrix mapping each input to its states (total_states x nfilter)
    - C: output matrix mapping states to outputs (nfilter x total_states)
    - D: diagonal feedthrough matrix (nfilter x nfilter)
    - x: concatenated state vector (total_states,)
    - u: input vector (nfilter,)
    - y: output vector (nfilter,)
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

        # Convert inputs and determine if block-diagonal construction is needed
        A, needs_bd_A = self._ensure_matrix_list(A)
        B, needs_bd_B = self._ensure_matrix_list(B)
        C, needs_bd_C = self._ensure_matrix_list(C)
        D, needs_bd_D = self._ensure_matrix_list(D)

        needs_block_diagonal = needs_bd_A or needs_bd_B or needs_bd_C or needs_bd_D

        # If n_modes is provided, expand and build block-diagonal
        if n_modes is not None:
            A, B, C, D = self._expand_with_n_modes(A, B, C, D, n_modes)
        # Build block-diagonal only if needed
        elif needs_block_diagonal:
            A, B, C, D = self._build_block_diagonal(A, B, C, D)
        else:
            # Single matrix: unwrap from list
            A, B, C, D = A[0], B[0], C[0], D[0]

        # A, B, C, D should now be single matrices
        self.A = self.to_xp(A, dtype=self.dtype)
        self.B = self.to_xp(B, dtype=self.dtype)
        self.C = self.to_xp(C, dtype=self.dtype)
        self.D = self.to_xp(D, dtype=self.dtype)

        # Extract metadata from matrix dimensions
        self.total_states = self.A.shape[0]
        self.nfilter = self.B.shape[1]  # Number of inputs = number of filters

        # Validate dimensions
        self._validate_dimensions()

    def _ensure_matrix_list(self, x):
        """Convert input to list of 2D numpy arrays.
        
        Handles:
        - Scalars: 1 -> [[[1]]]
        - 1D lists: [1, 2] -> [[[1]], [[2]]]
        - 2D arrays: [[1]] -> [[[1]]]
        - 2D lists: [[1,2],[3,4]] -> [[[1,2],[3,4]]]
        - Lists of 2D: [[[1]], [[2]]] -> [[[1]], [[2]]]
        - Lists of 0D arrays: [np.array(0.9), np.array(0.8)] -> [[[0.9]], [[0.8]]]

        Returns
        -------
        matrices : list of ndarray
            List of 2D matrices
        needs_block_diagonal : bool
            True if input was multiple filters that need block-diagonal construction
            False if input was already a single matrix or needs no processing
        """
        # Check if it's an array (numpy or cupy)
        if isinstance(x, self.xp.ndarray) or isinstance(x, np.ndarray):
            # Convert to numpy for processing (cpuArray handles both np and cp)
            x_np = cpuArray(x)

            if x_np.ndim == 2:
                return [x_np], False  # Already a complete matrix
            elif x_np.ndim == 0:  # scalar
                return [np.array([[x_np.item()]])], False
            elif x_np.ndim == 1:
                # 1D array of scalars -> multiple filters
                return [np.array([[xi]]) for xi in x_np], True
            else:
                raise ValueError(f"Unexpected array dimension: {x_np.ndim}")

        # Single scalar -> single filter
        if isinstance(x, (int, float, np.number)):
            return [np.array([[x]])], False

        # If list
        if isinstance(x, list):
            # Empty list
            if len(x) == 0:
                raise ValueError("Empty matrix list")

            # Check first element to determine structure
            first = x[0]

            # List of arrays (numpy or cupy)
            if isinstance(first, self.xp.ndarray) or isinstance(first, np.ndarray):
                first_np = cpuArray(first)

                if first_np.ndim == 2:
                    # List of 2D arrays -> convert all to numpy
                    matrices = [cpuArray(xi) for xi in x]
                    return matrices, len(matrices) > 1
                elif first_np.ndim == 0:
                    # List of 0D arrays (scalars) -> treat as list of scalars
                    matrices = [np.array([[cpuArray(xi).item()]]) for xi in x]
                    return matrices, len(matrices) > 1
                elif first_np.ndim == 1:
                    # List of 1D arrays -> unclear intent, raise error
                    raise ValueError("List of 1D arrays is ambiguous. "
                                   "Use list of 2D matrices or flatten to single 1D array.")
                else:
                    raise ValueError(f"Unexpected array dimension in list: {first_np.ndim}")

            # List of lists -> need to determine if single 2D or list of 2D
            if isinstance(first, list):
                # Disallow empty inner lists
                if len(first) == 0:
                    raise ValueError("Empty row in matrix list")

                # Check if first element is also a list (nested structure)
                if isinstance(first[0], list):
                    # Could be:
                    # 1. List of 2D matrices: [[[1]], [[2]]]
                    # 2. Single 3D structure: [[[1,2],[3,4]]]

                    # Check all elements are non-empty lists of lists
                    if all(isinstance(xi, list) and len(xi) > 0 \
                        and isinstance(xi[0], list) for xi in x):
                        # List of 2D matrices -> multiple filters
                        matrices = [np.array(xi) for xi in x]
                        return matrices, len(matrices) > 1
                    else:
                        raise ValueError("Inconsistent nested list structure")
                else:
                    # Single 2D matrix: [[1, 2], [3, 4]]
                    return [np.array(x)], False

            # List of scalars [1, 2, 3] -> [[[1]], [[2]], [[3]]]
            if isinstance(first, (int, float, np.number)):
                matrices = [np.array([[xi]]) for xi in x]
                return matrices, len(matrices) > 1

        raise ValueError(f"Cannot convert {type(x)} to matrix list")

    def _expand_with_n_modes(self, A, B, C, D, n_modes):
        """Expand lists of matrices according to n_modes and build block-diagonal."""
        n_modes = np.atleast_1d(n_modes)

        # Build expanded lists
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

        # Build block-diagonal matrices
        return self._build_block_diagonal(A_list, B_list, C_list, D_list)

    def _build_block_diagonal(self, A_list, B_list, C_list, D_list):
        """Build block-diagonal system matrices from list of individual filter matrices."""
        nfilter = len(A_list)

        # Validate individual matrices
        self._validate_individual_matrices(A_list, B_list, C_list, D_list)

        # Convert to numpy arrays if needed (scipy works only with numpy)
        A_list_np = [cpuArray(A) for A in A_list]
        B_list_np = [cpuArray(B) for B in B_list]
        C_list_np = [cpuArray(C) for C in C_list]
        D_list_np = [cpuArray(D) for D in D_list]

        # Build block-diagonal A using scipy
        A_block = block_diag(*A_list_np)

        # Build B matrix: each filter's B goes in its state block, column i
        total_states = A_block.shape[0]
        B_block = np.zeros((total_states, nfilter))

        state_offset = 0
        for i, B_i in enumerate(B_list_np):
            n_states = B_i.shape[0]
            B_block[state_offset:state_offset+n_states, i] = B_i[:, 0]
            state_offset += n_states

        # Build C matrix: row i picks from filter i's state block
        C_block = np.zeros((nfilter, total_states))

        state_offset = 0
        for i, C_i in enumerate(C_list_np):
            n_states = C_i.shape[1]
            C_block[i, state_offset:state_offset+n_states] = C_i[0, :]
            state_offset += n_states

        # Build D matrix: diagonal with each filter's feedthrough
        D_block = np.diag([D_i[0, 0] for D_i in D_list_np])

        return A_block, B_block, C_block, D_block

    def _validate_individual_matrices(self, A_list, B_list, C_list, D_list):
        """Validate dimensions of individual filter matrices before building block-diagonal."""
        if not len(A_list) == len(B_list) == len(C_list) == len(D_list):
            raise ValueError("All matrix lists must have same length")

        for i, _ in enumerate(A_list):
            A_i = cpuArray(A_list[i])
            B_i = cpuArray(B_list[i])
            C_i = cpuArray(C_list[i])
            D_i = cpuArray(D_list[i])

            A_shape = A_i.shape
            B_shape = B_i.shape
            C_shape = C_i.shape
            D_shape = D_i.shape

            # Check all matrices are 2D
            if len(A_shape) != 2:
                raise ValueError(f"Filter {i}: A must be 2D, got shape {A_shape}")
            if len(B_shape) != 2:
                raise ValueError(f"Filter {i}: B must be 2D, got shape {B_shape}")
            if len(C_shape) != 2:
                raise ValueError(f"Filter {i}: C must be 2D, got shape {C_shape}")
            if len(D_shape) != 2:
                raise ValueError(f"Filter {i}: D must be 2D, got shape {D_shape}")

            # Check A is square
            if A_shape[0] != A_shape[1]:
                raise ValueError(f"Filter {i}: A must be square, got shape {A_shape}")

            n_states = A_shape[0]

            # Check B dimensions (must have single input per filter)
            if B_shape[0] != n_states:
                raise ValueError(f"Filter {i}: B first dimension must match A")
            if B_shape[1] != 1:
                raise ValueError(f"Filter {i}: B must have shape (n_states, 1), got {B_shape}")

            # Check C dimensions (must have single output per filter)
            if C_shape[1] != n_states:
                raise ValueError(f"Filter {i}: C second dimension must match A")
            if C_shape[0] != 1:
                raise ValueError(f"Filter {i}: C must have shape (1, n_states), got {C_shape}")

            # Check D dimensions (must be scalar feedthrough)
            if D_shape != (1, 1):
                raise ValueError(f"Filter {i}: D must be (1,1), got {D_shape}")

    def _validate_dimensions(self):
        """Validate that block-diagonal matrices have consistent dimensions."""
        A_shape = self.A.shape
        B_shape = self.B.shape
        C_shape = self.C.shape
        D_shape = self.D.shape

        # Check A is square
        if len(A_shape) != 2 or A_shape[0] != A_shape[1]:
            raise ValueError(f"A must be square 2D matrix, got shape {A_shape}")

        total_states = A_shape[0]

        # Check B dimensions
        if B_shape[0] != total_states:
            raise ValueError(f"B first dimension {B_shape[0]} must match"
                             f" A dimension {total_states}")

        nfilter = B_shape[1]

        # Check C dimensions
        if C_shape != (nfilter, total_states):
            raise ValueError(f"C shape {C_shape} must be ({nfilter}, {total_states})")

        # Check D dimensions (diagonal matrix)
        if D_shape != (nfilter, nfilter):
            raise ValueError(f"D shape {D_shape} must be ({nfilter}, {nfilter})")

    def save(self, filename):
        """Save filter data to FITS file."""
        hdr = fits.Header()
        hdr['VERSION'] = 1
        hdr['NFILTER'] = self.nfilter
        hdr['NSTATES'] = self.total_states

        hdu = fits.PrimaryHDU(header=hdr)
        hdul = fits.HDUList([hdu])

        # Save block matrices
        hdul.append(fits.ImageHDU(data=cpuArray(self.A), name='A'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.B), name='B'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.C), name='C'))
        hdul.append(fits.ImageHDU(data=cpuArray(self.D), name='D'))

        hdul.writeto(filename, overwrite=True)
        hdul.close()

    @staticmethod
    def restore(filename, target_device_idx=None):
        """Restore filter data from FITS file."""
        with fits.open(filename) as hdul:
            version = hdul[0].header.get('VERSION', 1)
            if version != 1:
                raise ValueError(f"Unsupported SSR filter data version: {version}")

            # New block-diagonal format
            A = hdul['A'].data
            B = hdul['B'].data
            C = hdul['C'].data
            D = hdul['D'].data

            return SsrFilterData(A, B, C, D,
                                    target_device_idx=target_device_idx)

    def get_fits_header(self):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def from_header(hdr):
        # TODO
        raise NotImplementedError()

    def get_value(self):
        # TODO
        raise NotImplementedError()

    def set_value(self, v):
        # TODO
        raise NotImplementedError()

    @staticmethod
    def from_gain(gain, target_device_idx=None):
        """Create a simple proportional controller: y[k] = gain * u[k].
        
        Parameters
        ----------
        gain : array_like
            Gains for each filter
            
        Returns
        -------
        SsrFilterData
            Pure gain (no state): y = gain * u
        """
        gain = np.atleast_1d(gain)
        n = len(gain)

        A_list = []
        B_list = []
        C_list = []
        D_list = []

        for i in range(n):
            # No internal state for pure gain (dummy 1x1 zero state)
            A_list.append(np.zeros((1, 1)))
            B_list.append(np.zeros((1, 1)))
            C_list.append(np.zeros((1, 1)))
            D_list.append(np.array([[gain[i]]]))

        # Pass lists directly - __init__ will handle block-diagonal construction
        return SsrFilterData(A_list, B_list, C_list, D_list,
                           target_device_idx=target_device_idx)

    @staticmethod
    def from_integrator(gain, ff=None, target_device_idx=None):
        """Create a discrete integrator with optional forgetting factor.
        
        Parameters
        ----------
        gain : array_like
            Integrator gains
        ff : array_like, optional
            Forgetting factors (leaky integrator). If None, uses 1.0 (pure integrator).
            
        Returns
        -------
        SsrFilterData
            State-space representation: 
            x[k+1] = ff*x[k] + gain*u[k]
            y[k] = x[k+1]
        """
        gain = np.atleast_1d(gain)
        n = len(gain)

        # Handle forgetting factor
        if ff is not None:
            ff = np.atleast_1d(ff)
            if len(ff) == 1:
                ff = np.full(n, ff[0])
            elif len(ff) != n:
                raise ValueError(f"ff length {len(ff)} doesn't match gain length {n}")
        else:
            ff = np.ones(n)  # Pure integrator (no forgetting)

        A_list = []
        B_list = []
        C_list = []
        D_list = []

        for i in range(n):
            # State equation: x[k+1] = ff*x[k] + gain*u[k]
            A_list.append(np.array([[ff[i]]]))
            B_list.append(np.array([[gain[i]]]))
            # Output equation: y[k] = x[k+1]
            C_list.append(np.array([[1.0]]))
            D_list.append(np.array([[0.0]]))

        # Pass lists directly - __init__ will handle block-diagonal construction
        return SsrFilterData(A_list, B_list, C_list, D_list,
                           target_device_idx=target_device_idx)

    def get_eigenvalues(self):
        """Get eigenvalues of A matrix for stability analysis."""
        return np.linalg.eigvals(cpuArray(self.A))

    def is_stable(self):
        """Check stability: all eigenvalues must be inside unit circle."""
        eigenvalues = self.get_eigenvalues()
        return bool(np.all(np.abs(eigenvalues) < 1.0))
