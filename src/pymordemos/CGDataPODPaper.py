import os
import random



from scipy import np_maxversion

from pymor.basic import *
from scipy.io import loadmat
from pymor.algorithms.error import *
from pymor.reductors.neural_network import NeuralNetworkReductor
from pymor.vectorarrays.interface import VectorArray

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.gridspec as gridspec

#fix seeds to get deterministic results
import torch
torch.manual_seed(42)
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
random.seed(42)
np.random.seed(42)


import pickle
from dataclasses import dataclass
import scienceplots


plt.style.use(['science', 'ieee', 'no-latex'])
plt.rc('text', usetex = True)
plt.rcParams.update({'figure.dpi': 300, 'font.size': 10})
# 1. read in data from MercuryCG and save it to pickle file
# 2. get steady state density fields from MercuryCG (last 2 rotations)
# 3. train_test_split the data into training and testing data (0.8 to 0.2) try different ratios!
# 4. flatten density fields to get them into a numpy vector array (see green book) (steady state = only last timestep).
# 5. apply POD to the numpy vector array.
# 6. get the reduced basis. Take the same basis for the testing dataset. Take the inner product of the testing data with the reduced POD basis (modes)

class SimData:
    def __init__(self, field_data, parameters, x=None, z=None):
        self.field_data = field_data
        self.parameters = parameters
        self.x = x
        self.z = z

        # filled by split()
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.train_params = None
        self.val_params = None
        self.test_params = None
        self.data_params = None
        self.data_permutation = None

        # parameter template; decides how many and which parameters have been used by the simulation
        self._param_template = Parameters({"volumeRatio": 1})

    @classmethod
    def from_files(cls, field_file, x_file="x_s.pickle", z_file="z_s.pickle",
                   param_range=(1.0, 2.0)):
        """Factory method to load grid, field data, and parameters. Creates a SimData object."""
        x = pickle.load(open(x_file, "rb"))
        z = pickle.load(open(z_file, "rb"))

        field_data = pickle.load(open(field_file, "rb"))
        field_data = field_data[:, :, :, -1]  # take last timestep

        params = np.linspace(param_range[0], param_range[1], field_data.shape[0])
        return cls(field_data=field_data, parameters=params, x=x, z=z)

    def split(self, test_size=0.1, val_size=0.1, random_state=42):
        """Split data and parameters into train, validation, and test sets."""
        n = self.field_data.shape[0]
        indices = np.arange(n)
        # compute exact integer sizes to avoid rounding errors using train_test_split
        n_test = int(round(test_size * n))
        n_val = int(round(val_size * n))

        train_val_idx, test_idx = train_test_split(indices, test_size=n_test, random_state=random_state)
        train_idx, val_idx = train_test_split(train_val_idx,
                                              test_size=n_val,
                                              random_state=random_state)

        self.train_data, self.val_data, self.test_data = \
            self.field_data[train_idx], self.field_data[val_idx], self.field_data[test_idx]
        self.train_params, self.val_params, self.test_params = \
            self.parameters[train_idx], self.parameters[val_idx], self.parameters[test_idx]

        # save indices for reproducibility
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx

    def split_fixed(self, seed: int = 42):
        """Shuffle the dataset once and keep the permutation fixed."""
        n_total = self.field_data.shape[0]
        rng = np.random.default_rng(seed)
        self.data_permutation = rng.permutation(n_total)

    def split_for_train_size(self, train_size: int):
        """Split into train/val/test deterministically based on a fixed permutation."""
        if train_size > len(self.data_permutation):
            raise ValueError(f"train_size={train_size} exceeds total samples={len(self._perm)}")

        train_idx = self.data_permutation[:train_size]
        remaining = self.data_permutation[train_size:]

        mid = len(remaining) // 2
        val_idx = remaining[:-10]
        test_idx = remaining[-10:]

        self.train_data, self.val_data, self.test_data = \
            self.field_data[train_idx], self.field_data[val_idx], self.field_data[test_idx]
        self.train_params, self.val_params, self.test_params = \
            self.parameters[train_idx], self.parameters[val_idx], self.parameters[test_idx]

        # save indices for reproducibility
        self.train_idx, self.val_idx, self.test_idx = train_idx, val_idx, test_idx

    # helper functions for easy access to pyMOR VectorSpaces and Parameters
    @property
    def grid_shape(self):
        """Return (nx, nz) spatial grid shape as tuple."""
        return self.x.shape[0], self.z.shape[0]

    def reshape_vector_array(self, V: VectorArray) -> np.ndarray:
        """Convert pyMOR VectorArray to (n_snapshots, nx, nz)."""
        nx, nz = self.grid_shape
        return V.to_numpy().T.reshape((len(V), nx, nz))

    # --- snapshots ---
    def all_snapshots(self):
        return NumpyVectorSpace.from_numpy(self.field_data.reshape((self.field_data.shape[0], -1)).T)

    def train_snapshots(self):
        return NumpyVectorSpace.from_numpy(self.train_data.reshape((self.train_data.shape[0], -1)).T)

    def val_snapshots(self):
        return NumpyVectorSpace.from_numpy(self.val_data.reshape((self.val_data.shape[0], -1)).T)

    def test_snapshots(self):
        return NumpyVectorSpace.from_numpy(self.test_data.reshape((self.test_data.shape[0], -1)).T)

    # --- parameters ---
    def parse_train_params(self):
        return [self._param_template.parse(p) for p in self.train_params]

    def parse_val_params(self):
        return [self._param_template.parse(p) for p in self.val_params]

    def parse_test_params(self):
        return [self._param_template.parse(p) for p in self.test_params]

@dataclass
class MORResult:
    epsilon: float
    abs_epsilon: float
    modes: VectorArray       # pyMOR VectorArray of modes
    singular_values: np.ndarray
    reductor: NeuralNetworkReductor | None = None
    rom: object | None = None
    # error metrics (relative l2-norm)
    pod_errors: np.ndarray | None = None              # shape (n_snapshots)
    pod_error_mean: float | None = None
    nn_errors: dict[str, np.ndarray] | None = None    # {"train": ..., "val": ..., "test": ...}
    nn_errors_mean: dict[str, float] | None = None

class PODNNPipeline:
    def __init__(self, sim_data: SimData):
        self.sim_data = sim_data
        self.results: list[MORResult] = []

    def run_pipeline(self, modes: int, train_size: int) -> MORResult:
        """
        Run one POD + NN experiment with given number of modes and training size.
        Validation and test sizes adapt automatically according to test = val = (N-train)/2.
        """
        # n_total = self.sim_data.field_data.shape[0]
        #
        # if (train_size > n_total):
        #     raise ValueError(f"train_size={train_size} must be smaller than total samples={n_total}")
        #
        # # split remaining samples equally into val and test
        # remaining = n_total - train_size
        # val_size = test_size = remaining // 2
        #
        # if remaining % 2 != 0:
        #     raise ValueError("Remaining samples after choosing train_size must be even.")
        #
        # val_percent = val_size / n_total
        # test_percent = test_size / n_total
        #
        # # Step 1: custom split
        # self.sim_data.split(test_percent, val_percent, random_state=42)

        # Step 2: compute POD basis with given modes
        #try:
            #pod = self.get_result(n_modes=modes)
        #except KeyError:
        pod = self.apply_pod(n_modes=modes)

        # Step 3: train NN with train subset
        self.reduce_with_nn(pod, ann_rel_l2_err=1e-1)

        # Step 5: compute errors once (train, val, test at once)
        self.compute_pod_error(pod)
        self.compute_nn_error(pod)

    # retrieve MORResult for a given epsilon from the results list
    def get_result(self, epsilon: float = None, n_modes: int = None,  tol=1e-12) -> MORResult:
        for r in self.results:
            if epsilon is not None and np.isclose(r.epsilon, epsilon, atol=tol, rtol=0):
                return r
            if n_modes is not None and np.isclose(len(r.modes), n_modes, atol=tol, rtol=0):
                return r
        raise KeyError(f"No MORResult for epsilon={epsilon} or n_modes={n_modes} found in results list.")

    def get_last_result(self):
        if len(self.results) == 0:
            raise RuntimeError("No MORResult computed yet. Call apply_pod first.")
        return self.results[-1]

    # apply POD and Neural Network Reduction and append reference to results list
    def apply_pod(self, epsilon=3e-3, n_modes=None) -> MORResult:
        """
        Compute POD for the given training data and relative L2 error.
        Returns the MORResult and stores it internally.
        """
        # Convert epsilon into absolute error and compute POD with this absolute error threshold
        abs_epsilon = epsilon * np.linalg.norm(self.sim_data.train_data)
        if n_modes is None:
            modes, singular_values = pod(self.sim_data.train_snapshots(), l2_err=abs_epsilon)
        else:
            modes, singular_values = pod(self.sim_data.train_snapshots(), modes=n_modes)
        print("Number of modes: ", len(modes))
        # Store the result
        result = MORResult(epsilon=epsilon, abs_epsilon=abs_epsilon, modes=modes, singular_values=singular_values)
        self.results.append(result)
        return result

    def reduce_with_nn(self, mor_result: MORResult, ann_rel_l2_err: float = 1e-2) -> MORResult:
        """
        Train a neural network reduced-order model for the given POD basis which was
        computed with a specific epsilon.

        Stores both the reductor and the ROM inside the MORResult dataclass.
        """
        if mor_result.singular_values.size == 0:
            raise RuntimeError("No MORResult provided or computed yet. Call apply_pod first.")

        # compute absolute MSE from relative L2 error
        ann_mse = ann_rel_l2_err ** 2 * np.linalg.norm(self.sim_data.field_data)

        # ann_mse="like_basis" would be an ann_mse of: (np.sum(Phi_s**2) - np.sum(singular_values**2)) / 100),
        # but with this high error threshold it does not find a neural network below the prescribed error.
        reductor = NeuralNetworkReductor(
            training_parameters=self.sim_data.parse_train_params(),
            training_snapshots=self.sim_data.train_snapshots(),
            validation_parameters=self.sim_data.parse_val_params(),
            validation_snapshots=self.sim_data.val_snapshots(),
            reduced_basis=mor_result.modes,
            ann_mse=ann_mse,
        )

        mor_result.reductor = reductor
        mor_result.rom = reductor.reduce(restarts=10, log_loss_frequency=10)

    # reconstruction and error computation by solving the ROM for given parameters
    # or by a projection onto the POD modes
    def solve_and_reconstruct(self, mor_result: MORResult, parameter) -> np.ndarray:
        """
        Reconstruct reduced solutions for a given parameter.
        """
        data_shape = self.sim_data.grid_shape
        return mor_result.reductor.reconstruct(mor_result.rom.solve(parameter)).to_numpy().reshape(data_shape)

    def solve_and_reconstruct_set(self, mor_result: MORResult, parameters, data_shape=None) -> np.ndarray:
        """
        Reconstruct all reduced solutions for a given parameter set.

        Parameters
        ----------
        mor_result : MORResult
            The MORResult containing reductor and ROM.
        parameters : list
            List of parameter values.
        data_shape : tuple, optional
            Shape of the spatial grid (x, z). If None, taken from sim_data.

        Returns
        -------
        np.ndarray
            Array of shape (len(mus), nx, nz) with reconstructed solutions.
        """
        if data_shape is None:
            data_shape = (self.sim_data.x.shape[0], self.sim_data.z.shape[0])

        preds = np.zeros((len(parameters), *data_shape))

        for i, mu in enumerate(parameters):
            snapshot = mor_result.rom.solve(mu)
            full_order = mor_result.reductor.reconstruct(snapshot)
            preds[i, :, :] = full_order.to_numpy().reshape(data_shape)

        return preds

    def project_and_reconstruct_set(self, mor_result: MORResult) -> np.ndarray:
        """
        Project snapshots onto reduced POD modes and reconstruct them.

        Parameters
        ----------
        mor_result : MORResult
            The MORResult containing the POD results.

        Returns
        -------
        np.ndarray
            Reconstructed snapshots with shape (n_snapshots, nx, nz).
        """
        # Projection coefficients (inner product)
        #TODO CHECK COEFFICIENTS! TO KNOW HOW THE MODES LOOK LIKE
        coefficients = self.sim_data.test_snapshots().inner(mor_result.modes)
        #print(np.sign(coefficients))
        # Reconstruct solutions from reduced basis by linear combination of modes and coefficients
        reconstruction = mor_result.modes.lincomb(coefficients.T)
        # Reshape to (n_snapshots, nx, nz) and return the result
        return self.sim_data.reshape_vector_array(reconstruction)

    # general purpose relative l2-error computation
    def compute_relative_errors(self, truth: np.ndarray, preds: np.ndarray, axis=(1, 2)):
        """
        Compute relative L2 errors between truth and predictions.
        Assumes that the 0-th axis holds the different snapshots and the remaining axes are spatial dimensions.

        Parameters
        ----------
        truth : np.ndarray
            Ground truth data of shape (n_snapshots, nx, nz).
        preds : np.ndarray
            Predicted data of the same shape.
        axis : tuple, optional
            Axes along which to compute the norm (default: spatial dims).

        Returns
        -------
        np.ndarray
            Relative L2 error for each sample, shape (n_snapshots,).
        """
        num = np.linalg.norm(truth - preds, axis=axis)
        den = np.linalg.norm(truth, axis=axis)
        return num / den
    # general purpose mean relative l2-error computation
    def compute_mean_relative_error(self, truth: np.ndarray, preds: np.ndarray):
        """
        Compute the mean relative L2 error between truth and predictions.

        Parameters
        ----------
        truth : np.ndarray
            Ground truth data of shape (n_snapshots, nx, nz).
        preds : np.ndarray
            Predicted data of the same shape.

        Returns
        -------
        np.ndarray
            Mean relative L2 error for each sample, shape (n_snapshots,).
        """
        num = np.linalg.norm(truth - preds)
        den = np.linalg.norm(truth)
        return num / den

    def compute_pod_error(self, mor_result: MORResult, truth: np.ndarray = None) -> MORResult:
        """
        Reconstruct solutions using POD for a set of parameters and compute errors.

        Parameters
        ----------
        mor_result : MORResult
            The MORResult containing modes and singular values.
        truth : np.ndarray
            Ground truth corresponding to mus, same shape as the reconstruction.
        """
        preds = self.project_and_reconstruct_set(mor_result)
        if truth is None:
            truth = self.sim_data.test_data
        # per-snapshot errors
        mor_result.pod_errors = self.compute_relative_errors(truth, preds)
        mor_result.pod_error_mean = self.compute_mean_relative_error(truth, preds)

    def compute_nn_error(self, mor_result: MORResult, data_shape=None):
        """
        Reconstruct solutions using the trained neural network reductor
        for a set of parameters and compute errors.

        Parameters
        ----------
        mor_result : MORResult
            The MORResult containing reductor and ROM.
        data_shape : tuple, optional
            Shape of the spatial grid (x, z). If None, taken from sim_data.
        """
        nn_errors = {}
        nn_errors_mean = {}

        for split_name, params, truth in [
            ("train", self.sim_data.train_params, self.sim_data.train_data),
            ("val", self.sim_data.val_params, self.sim_data.val_data),
            ("test", self.sim_data.test_params, self.sim_data.test_data),
        ]:
            preds = self.solve_and_reconstruct_set(mor_result, params, data_shape)
            nn_errors[split_name] = self.compute_relative_errors(truth, preds)
            nn_errors_mean[split_name] = self.compute_mean_relative_error(truth, preds)

        mor_result.nn_errors = nn_errors
        mor_result.nn_errors_mean = nn_errors_mean

    # plotting functions for figures
    def plot_pod_error_and_modes(self, mor_result: MORResult, num_modes: int = 4):
        """
        Plot singular values, cumulative tail error, and the first few normalized POD modes.

        Parameters
        ----------
        mor_result : MORResult
            The MORResult object containing epsilon, modes, and singular values.
        num_modes : int, optional
            Number of first POD modes to plot (default: 4).
        """
        train_data = self.sim_data.train_data
        x = self.sim_data.x
        z = self.sim_data.z
        epsilon = mor_result.epsilon
        modes = mor_result.modes
#TODO CHANGE POD MODE AND COEFFICIENT SIGN TO POSITIVE OF THE FIRST AND THIRD MODE!! SWITHC BOTH COEFF AND MODE SIGN
        # compute a full POD with all modes except the last one
        # used for plotting only, to intuitively show the mode selection procedure
        full_modes, full_singular_values = pod(self.sim_data.train_snapshots(),
                                               modes=len(self.sim_data.train_snapshots())-1)

        # Absolute mean L2 error threshold
        l2_err_mean_epsilon = epsilon * np.linalg.norm(train_data)

        # --- absolute l2 error ---
        abs_err = [np.sqrt(np.sum(full_singular_values[n:] ** 2)) for n in range(len(full_singular_values))]

        # --- Plot singular value decay ---
        plt.figure()
        plt.semilogy(full_singular_values, 'x-')
        plt.xlabel('$N$ [-]')
        plt.ylabel('$\sigma$ [-]')
        plt.savefig('singularValueDecay.eps', bbox_inches="tight")
        plt.savefig('singularValueDecay.png', bbox_inches="tight")
        plt.close()

        # --- Plot absolute l2 error over the number of selected modes ---
        plt.figure()
        plt.semilogy(abs_err, 'x-')
        plt.axhline(l2_err_mean_epsilon, color="r", linestyle="--", label=f'$\\varepsilon={epsilon}$')
        plt.axvline(len(modes), color="k", linestyle="--", label=f'$n={len(modes)}$')
        ax = plt.gca()
        ax.set_xlabel("$n$ [-]")
        ax.set_ylabel(r'$\displaystyle \sum_{i = n+1}^R \sigma_i$ [-]')
        ax.yaxis.set_label_coords(-0.08, 0.5)
        ax.legend()
        plt.tight_layout()
        plt.savefig('ModeSelection.eps', bbox_inches="tight", pad_inches=0.2)
        plt.savefig('ModeSelection.png', bbox_inches="tight")
        plt.close()

        # --- Plot the first num_modes normalized modes ---
        npModes = modes.to_numpy().T.reshape((len(modes), x.shape[0], z.shape[0]))
        npModes_norm = npModes[:num_modes] / np.max(np.abs(npModes[:num_modes]), axis=(1, 2), keepdims=True)

        rows = int(np.ceil(num_modes / 2))
        cols = min(2, num_modes)
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
        axs = np.array(axs).reshape(-1)  # flatten in case of single row/column
        vmin, vmax = -1, 1

        for i in range(num_modes):
            axs[i].contourf(x[:, :, -1], z[:, :, -1], npModes_norm[i], cmap='seismic', levels=100, vmin=vmin, vmax=vmax)
            axs[i].set_title(f'Mode {i + 1}')

        # create a continuous colorbar with a diverging colormap to see positive
        # and negative contributions of the modes
        sm = cm.ScalarMappable(cmap='seismic', norm=colors.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        fig.colorbar(sm, ax=axs[:num_modes], orientation='vertical', format='%1.2f', ticks=np.linspace(vmin, vmax, 5))

        fig.text(0.42, 0.03, '$x$ [m]', ha='center', va='center')
        fig.text(0.03, 0.5, '$y$ [m]', ha='center', va='center', rotation='vertical')
        fig.subplots_adjust(hspace=0.55, wspace=0.45, right=0.75, bottom=0.12)
        plt.savefig('Modes.eps', bbox_inches="tight")
        plt.savefig('Modes.png', bbox_inches="tight")
        plt.show()

    def plot_reconstruction(self, mor_result: MORResult, mu, levels_num=10, filename=None):
        """
        Plot reconstructed contour against the corresponding full-order solution.

        Parameters
        ----------
        mor_result : MORResult
            The POD/NN result containing reductor and ROM.
        mu : float
            Parameter value to reconstruct.
        levels_num : int
            Number of contour levels for contour plot. Default: 8.
        filename : str | None
            If given, save figure to this file. Otherwise, just show.
        """
        x, z = self.sim_data.x, self.sim_data.z

        # Find the corresponding snapshot for mu
        try:
            snapshot_index = np.where(np.isclose(self.sim_data.parameters, mu))[0][0]
        except IndexError:
            raise ValueError(f"Parameter mu={mu} not found in sim_data.parameters.")

        full_order = self.sim_data.field_data[snapshot_index]
        reconstructed = mor_result.reductor.reconstruct(mor_result.rom.solve(mu)).to_numpy().reshape(
            self.sim_data.grid_shape
        )

        # Define fixed contour levels (nice and simple)
        vmin, vmax = 0,1
        levels = np.linspace(vmin, vmax, levels_num)

        reconstructed_norm = reconstructed / np.max(np.abs(reconstructed), keepdims=True)
        full_order_norm = full_order / np.max(np.abs(full_order), keepdims=True)

        # Create plots
        fig, axes = plt.subplots(2, 1)
        data = [reconstructed_norm, full_order_norm]
        titles = [rf'$u_N(\mu)$', r'$u(\mu)$']

        for ax, dat, title in zip(axes, data, titles):
            contour = ax.contourf(x[:, :, -1], z[:, :, -1], dat, cmap='Greys', levels=levels)
            ax.set_xlim(x.min(), x.max())
            ax.set_ylim(z.min(), z.max())
            ax.set_aspect('equal')
            ax.set_title(title)

        # Shared colorbar
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        mappable = cm.ScalarMappable(norm=norm, cmap='Greys')
        fig.colorbar(mappable, ax=axes, label=r'$\Phi^{\mathrm{s}}$ [-]', format='%1.2f')

        # Axis labels
        fig.text(0.65, 0.04, '$x$ [m]', ha='center', va='center')
        fig.text(0.44, 0.5, '$y$ [m]', ha='center', va='center', rotation='vertical')
        fig.subplots_adjust(hspace=0.55, right=0.75, bottom=0.14)

        if filename:
            fig.savefig(filename, bbox_inches="tight", dpi=300)
        plt.show()

    def plot_field_data_samples(self, data_type='train', max_samples=16):
        """
        Plot all field data samples in a matrixed figure.

        Parameters
        ----------
        data_type : str
            Which data to plot: 'train', 'val', 'test', or 'all'.
        max_samples : int
            Maximum number of samples to display (arranged in square grid).
        """
        # Select data
        if data_type == 'train':
            data = self.sim_data.train_data
        elif data_type == 'val':
            data = self.sim_data.val_data
        elif data_type == 'test':
            data = self.sim_data.test_data
        elif data_type == 'all':
            data = self.sim_data.field_data
        else:
            raise ValueError("data_type must be 'train', 'val', 'test', or 'all'")

        n_samples = min(len(data), max_samples)
        # Determine grid size
        n_cols = int(np.ceil(np.sqrt(n_samples)))
        n_rows = int(np.ceil(n_samples / n_cols))

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows), squeeze=False)

        x, z = self.sim_data.x, self.sim_data.z

        for idx in range(n_samples):
            r, c = divmod(idx, n_cols)
            ax = axs[r, c]

            # Normalize sample for consistent color scale
            sample = data[idx]
            norm_sample = sample / np.max(np.abs(sample))

            im = ax.contourf(x[:, :, -1], z[:, :, -1], norm_sample, cmap='viridis', levels=50)
            ax.set_title(f'Sample {idx}')
            ax.axis('off')

        # Remove unused axes
        for idx in range(n_samples, n_rows*n_cols):
            r, c = divmod(idx, n_cols)
            axs[r, c].axis('off')

        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.02)
        cbar.set_label('Normalized field [-]')

        plt.show()

    def plot_pod_nn_errors(self, mor_result: MORResult, filename_prefix: str = "POD_NN_Errors"):
        """
        Plot POD and Neural Network reconstruction errors stored in a MORResult.

        Parameters
        ----------
        mor_result : MORResult
            MORResult object containing POD and NN errors.
        filename_prefix : str
            Prefix for saving figure files (default: 'POD_NN_Errors').
        """
        # POD errors
        if mor_result.pod_errors is None:
            self.compute_pod_error(mor_result)
        # NN errors
        if mor_result.nn_errors is None:
            self.compute_nn_error(mor_result)

        pod_error = mor_result.pod_errors
        nn_errors = mor_result.nn_errors

        size_ratios_train = np.arange(1, len(nn_errors["train"]) + 1)
        size_ratios_val = np.arange(1, len(nn_errors["val"]) + 1)
        size_ratios_test = np.arange(1, len(nn_errors["test"]) + 1)

        plt.figure()
        plt.semilogy(self.sim_data.parameters, pod_error, ".", label='$u_n$')
        plt.semilogy(self.sim_data.train_params, nn_errors["train"], ".", color="red", label=r'$\tilde{u}_n^{\mathrm{train}}$')
        plt.semilogy(self.sim_data.val_params, nn_errors["val"], "x", color="red", label=r'$\tilde{u}_n^{\mathrm{val}}$')
        plt.semilogy(self.sim_data.test_params, nn_errors["test"], "^", color="red", label=r'$\tilde{u}_n^{\mathrm{test}}$')
        plt.axhline(y=mor_result.epsilon, linestyle="-", color="black", label=r'$\varepsilon$')

        plt.xlabel(r'$s$ [-]')
        plt.ylabel(r'$E$ [-]')
        plt.xticks(np.arange(1, 3, 1))
        plt.ylim(1e-3, 1e-1)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

        plt.savefig(f'{filename_prefix}.eps', bbox_inches="tight")
        plt.savefig(f'{filename_prefix}.png', bbox_inches="tight")
        plt.show()

    def plot_pod_nn_errors_all_combinations(self, modes_list, train_sizes_list):
        """
        Perform a parameter study over modes and training sizes. Use run_pipeline to apply the POD,
        train a neural network with the coefficients and compute the respective errros for a given
        combination of modes and training size. Then compute mean errors for each combination and
        plot a contour of mean errors against parameters (modes and training size).

        Parameters
        ----------
        modes_list : list[int]
            List of POD mode numbers to test.
        train_sizes_list : list[int]
            List of training sample sizes for NN.
        """
        errors_nn = np.zeros((len(modes_list), len(train_sizes_list)))
        errors_pod = np.zeros((len(modes_list), len(train_sizes_list)))
        errors_gap = np.zeros((len(modes_list), len(train_sizes_list)))

        self.sim_data.split_fixed(seed=42)

        if "errors_gap_fix_layers2.npy" in os.listdir("."):
            print("Loading existing error data from 'errors_gap.np', 'errors_pod.np', 'errors_nn.np'")
            errors_gap = np.load("errors_gap_fix_layers2.npy")
            errors_pod = np.load("errors_pod_fix_layers2.npy")
            errors_nn = np.load("errors_nn_fix_layers2.npy")
        else:
            for j, train_size in enumerate(train_sizes_list):
                self.sim_data.split_for_train_size(train_size)
                for i, n_modes in enumerate(modes_list):
                    if n_modes > train_size:
                        # Option 1: skip invalid combination
                        errors_nn[i, j] = np.nan
                        errors_pod[i, j] = np.nan
                        errors_gap[i, j] = np.nan
                        continue
                    # Run POD + NN with given parameters
                    self.run_pipeline(modes=n_modes, train_size=train_size)

                    # Attach to pipeline-managed results (NOT NEEDED?)
                    #self.results.append(mor_result)

                    # compute errors
                    #err = mor_result.nn_errors["test"]
                    # compute mean error for this sample

                    # compute error metrics
                    nn_mean = self.get_last_result().nn_errors_mean["test"]
                    pod_mean = self.get_last_result().pod_error_mean
                    gap = nn_mean - pod_mean
                    # Store mean errors for this configuration
                    errors_nn[i, j] = nn_mean
                    errors_pod[i, j] = pod_mean
                    errors_gap[i, j] = gap

                    # free up memory after extracting values
                    self.results.clear()

            # save errors to file
            np.save("errors_pod_fix_layers2.npy", errors_pod)
            np.save("errors_nn_fix_layers2.npy", errors_nn)
            np.save("errors_gap_fix_layers2.npy", errors_gap)

        # Set common log scale limits
        vmin, vmax = 1e-2, 1
        levels = np.logspace(-2, 0, 50)
        ticks = np.logspace(-2, 0, 3)
        cmap = "viridis"

        X, Y = np.meshgrid(train_sizes_list, modes_list)

        # Normalize to [0, 1]
        max_abs = np.nanmax(np.abs(errors_nn))
        errors_norm = errors_nn / max_abs

        plt.figure()
        contour = plt.contourf(X, Y, errors_norm, levels=levels, cmap=cmap,
                               norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(contour, ticks=ticks, label=r"$E^{\prime}_{\mathrm{m,NN}}$ [-]")
        # Use LogFormatterSciNotation for scientific notation
        cbar.formatter = ticker.LogFormatterSciNotation(base=10.0)
        cbar.update_ticks()
        cbar.ax.minorticks_off()

        plt.xlabel(r"$n_{train}$")
        plt.ylabel(r"$n$")
        plt.tight_layout()
        plt.savefig("Errors_nn.eps", bbox_inches="tight")
        plt.savefig("Errors_nn.png", bbox_inches="tight")

        # Normalize to [0, 1]
        max_abs = np.nanmax(np.abs(errors_pod))
        errors_norm = errors_pod / max_abs

        plt.figure()
        contour = plt.contourf(X, Y, errors_norm, levels=levels, cmap=cmap,
                               norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(contour, ticks=ticks, label=r"$E^{\prime}_{\mathrm{m,POD}}$ [-]")
        cbar.formatter = ticker.LogFormatterSciNotation(base=10.0)
        cbar.update_ticks()
        cbar.ax.minorticks_off()

        plt.xlabel(r"$n_{train}$")
        plt.ylabel(r"$n$")
        plt.tight_layout()
        plt.savefig("Errors_pod.eps", bbox_inches="tight")
        plt.savefig("Errors_pod.png", bbox_inches="tight")

        # Normalize to [-1, 1]
        max_abs = np.nanmax(np.abs(errors_gap))
        errors_norm = errors_gap / max_abs

        plt.figure()
        contour = plt.contourf(X, Y, errors_norm, levels=levels, cmap=cmap,
                               norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(contour, ticks=ticks, label=r"$E^{\prime}_{\mathrm{m,gap}}$ [-]")
        cbar.formatter = ticker.LogFormatterSciNotation(base=10.0)
        cbar.update_ticks()
        cbar.ax.minorticks_off()

        plt.xlabel(r"$n_{train}$")
        plt.ylabel(r"$n$")
        plt.tight_layout()
        plt.savefig("Errors_gap.eps", bbox_inches="tight")
        plt.savefig("Errors_gap.png", bbox_inches="tight")
        plt.show()

    def plot_mixing_index(self, mor_result: MORResult, field_data_bulk_file):
        # bulk field data (Phi_g)
        field_data_bulk = pickle.load(open(field_data_bulk_file, "rb"))


# def plot_full_POD_mode_selection(x, z, Phi, l2_err_epsilon=3e-3):
#     # POD relative mean l2 error. relative to the bulk volume fraction magnitude:
#     # E = ||u_n - u||_2 / ||u||_2
#     # l2_err_rel_mean = 1e-4
#     # l2_err_mean = l2_err_rel_mean * np.linalg.norm(Phi)
#
#     l2_err_mean_epsilon = l2_err_epsilon * np.linalg.norm(Phi)
#
#     U = NumpyVectorSpace.from_numpy(Phi.reshape((Phi.shape[0], -1)).T)
#     modes, singular_values = pod(U, modes=len(Phi)-1)
#
#     print("Number of modes: ", len(modes))
#
#     abs_err = []
#     for n in range(1, len(singular_values) + 1):
#         tail_err = np.sum(singular_values[n:] ** 2)
#         abs_err.append(np.sqrt(tail_err))
#
#     plt.figure(1)
#     plt.semilogy(singular_values, 'x-')
#     plt.xlabel('$N$ [-]')
#     plt.ylabel('$\sigma$ [-]')
#     plt.legend()
#     plt.savefig('singularValueDecay.eps', bbox_inches="tight")
#     plt.savefig('singularValueDecay.png', bbox_inches="tight")
#
#     plt.figure(2)
#     plt.semilogy(abs_err, 'x-')
#     plt.axhline(l2_err_mean_epsilon, color="r", linestyle="--", label=r'$\varepsilon=3\times10^{-3}$')
#     plt.axvline(15, color="k", linestyle="--", label="$n=15$")
#     ax = plt.gca()  # get current axes
#     ax.set_xlabel("$n$ [-]")
#     # Move the y-label closer using label coordinates
#     ax.set_ylabel(r'$\displaystyle \sum_{i = n+1}^R \sigma_i$ [-]')
#     ax.yaxis.set_label_coords(-0.08,0.5)
#     ax.legend()
#     plt.tight_layout()
#     plt.savefig('ModeSelection.eps', bbox_inches="tight", pad_inches=0.2)
#     plt.savefig('ModeSelection.png', bbox_inches="tight")
#     plt.show()
#
#     npModes = modes.to_numpy()
#     npModes = npModes.T.reshape((len(modes), x.shape[0], z.shape[0]))
#
#     # Plot the first four modes in a single figure
#     # normalise to [-1,1] to keep the negative and positive contributions to the field intact
#     fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
#     vmin = -1
#     vmax = 1
#     npModes_norm = npModes[:4] / np.max(np.abs(npModes[:4]), axis=(1, 2), keepdims=True)
#     contours = []
#     for i, ax in enumerate(axs.flat):
#         contour = ax.contourf(x[:, :, -1], z[:, :, -1], npModes_norm[i],
#                               cmap='seismic', levels=100, vmin=vmin, vmax=vmax)
#         contours.append(contour)
#         ax.set_title(f'Mode {i + 1}')
#
#     # create a continuous colorbar with a diverging colormap to see positive and negative contributions of the modes
#     sm = cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=vmin, vmax=vmax))
#     sm.set_array([])  # required for colorbar
#     cbar = fig.colorbar(sm, ax=axs, orientation='vertical',
#                         format='%1.2f', ticks=np.linspace(vmin, vmax, 5))
#     cbar.set_label(r'$\mathit{\Phi}^{\mathrm{s}\prime}$ [-]')
#
#     # place text manually
#     fig.text(0.42, 0.03, '$x$ [m]', ha='center', va='center')
#     fig.text(0.03, 0.5, '$y$ [m]', ha='center', va='center', rotation='vertical')
#
#     # adjust plot size to make the text fit
#     fig.subplots_adjust(hspace=0.55, wspace=0.45, right=0.75, bottom=0.12)
#     plt.savefig('Modes.eps', bbox_inches="tight")
#     plt.savefig('Modes.png', bbox_inches="tight")
#     plt.show()

# def load_grid():
#     # load the pickle of the spatial evaluation grid data
#     x = pickle.load(open("x_s.pickle", "rb"))
#     z = pickle.load(open("z_s.pickle", "rb"))
#     return x, z
#
# def load_field_data(filename):
#     # load the pickle of the respective field data
#     field_data = pickle.load(open(filename, "rb"))
#     # take only the last timestep (if timeaveraged, reduce dimensions)
#     field_data = field_data[:, :, :, -1]
#     return field_data

# def assemble_NN_training_validation_test_data(field_data, parameter_range_start=1.0, parameter_range_end=2.0):
#     # assemble training, validation and test set
#     num_samples = field_data.shape[0]
#     indices = np.arange(num_samples)
#
#     # Split indices
#     train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
#     train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11111111, random_state=42)
#
#     # Use indices to split data and parameters
#     field_data_train = field_data[train_idx]
#     field_data_val = field_data[val_idx]
#     field_data_test = field_data[test_idx]
#
#     parameters = np.linspace(1.0, 2.0, num_samples)
#     parameters_train = parameters[train_idx]
#     parameters_val = parameters[val_idx]
#     parameters_test = parameters[test_idx]

def main():
    # POD relative mean l2 error. relative to the bulk volume fraction magnitude:
    # E = ||u_n - u||_2 / ||u||_2
  #  l2_err_rel_mean = 3e-3
  #  l2_err_mean = l2_err_rel_mean * np.linalg.norm(Phi_s)
    # ANN relative mean square error. relative to the bulk volume fraction magnitude:
    # E^2 = ||u_n - u||_2^2 / ||u||_2^2
  #  l2_err_ann_rel_mean = 7e-3
  #  ann_mse = l2_err_ann_rel_mean ** 2 * np.linalg.norm(Phi_s)
    # 1. Load data
    sim_data = SimData.from_files(
        "simulationStudy_s.pickle", "x_s.pickle", "z_s.pickle"
    )
    sim_data.split(test_size=0.5, val_size=0.3)

    # 2. Create pipeline
    pipeline = PODNNPipeline(sim_data)

    modes_list = np.linspace(1,100, 100).astype(int)
    train_size_list = np.linspace(1,98,49).astype(int)

    pipeline.plot_pod_nn_errors_all_combinations(modes_list, train_size_list)
#TODO REWRITE TEST TO USE THE INDEPENDENT DATA SET WHEN IT IS READY
    # pipeline.plot_field_data_samples("test", max_samples=10)

    # 3. Compute PODs
    #pod3 = pipeline.apply_pod(3e-3)

    # l2_err_epsilon = 3e-3 results in 15 modes
    #pipeline.reduce_with_nn(pod3,1e-2)

    #plot reconstruction of a single parameter to see the match
    #pipeline.plot_reconstruction(pod3, mu=2.0, filename="reconstructionNN2.png")

    #pipeline.plot_pod_error_and_modes(pod3, num_modes=4)

    #pipeline.compute_nn_error(pod3)
    #pipeline.compute_pod_error(pod3)

    #pipeline.plot_pod_nn_errors(pod3)
if __name__=='__main__':
     main()