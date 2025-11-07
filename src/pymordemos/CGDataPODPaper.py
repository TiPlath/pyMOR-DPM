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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
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
    def __init__(self, field_data, testset_data, testset_parameters, parameters, x=None, z=None):
        self.field_data = field_data
        self.testset_data = testset_data
        self.parameters = parameters
        self.testset_parameters = testset_parameters
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
    def from_files(cls, field_file, testset_file, testset_params, x_file="x_s.pickle", z_file="z_s.pickle",
                   param_range=(1.0, 2.0)):
        """Factory method to load grid, field data, and parameters. Creates a SimData object."""
        x = pickle.load(open(x_file, "rb"))
        z = pickle.load(open(z_file, "rb"))

        field_data = pickle.load(open(field_file, "rb"))
        if field_data.ndim == 5: # n_sample,n_windows,nx,nz,nt
            field_data = field_data[:, :, :, :, -1]  # take last timestep
            params = np.linspace(param_range[0], param_range[1], field_data.shape[0]*field_data.shape[1])
            n_samples, n_w = field_data.shape[:2]
            # flatten the first two dimensions n_samples*n_windows
            field_data = field_data.reshape(-1, *field_data.shape[2:])
            params = np.repeat(params[::n_w], n_w)
        elif field_data.ndim == 4: # n_sample,nx,nz,nt
            field_data = field_data[:, :, :, -1]  # take last timestep
            params = np.linspace(param_range[0], param_range[1], field_data.shape[0])

        # independent test set data, same shape as field_data
        testset_data = pickle.load(open(testset_file, "rb"))
        testset_data = testset_data[:, :, :, -1]  # take last timestep

        testset_params = testset_params
        return cls(field_data=field_data, testset_data=testset_data,
                   testset_parameters=testset_params, parameters=params, x=x, z=z)

    def split(self, val_size=0.1, random_state=42):
        """Split data and parameters into train, validation, and test sets."""
        n = self.field_data.shape[0]
        n_test = self.testset_data.shape[0]
        indices = np.arange(n)

        n_val = int(round(val_size * n))

        train_idx, val_idx = train_test_split(indices, test_size=n_val, random_state=random_state)
        test_idx = np.arange(n_test)

        self.train_data, self.val_data, self.test_data = \
            self.field_data[train_idx], self.field_data[val_idx], self.testset_data[test_idx]
        self.train_params, self.val_params, self.test_params = \
            self.parameters[train_idx], self.parameters[val_idx], self.testset_parameters[test_idx]

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
            raise ValueError(f"train_size={train_size} exceeds total samples={len(self.data_permutation)}")

        train_idx = self.data_permutation[:train_size]
        val_idx = self.data_permutation[train_size:]
        test_idx = np.arange(self.testset_data.shape[0])

        self.train_data, self.val_data, self.test_data = \
            self.field_data[train_idx], self.field_data[val_idx], self.testset_data[test_idx]
        self.train_params, self.val_params, self.test_params = \
            self.parameters[train_idx], self.parameters[val_idx], self.testset_parameters[test_idx]

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
    coefficients: VectorArray | None = None  # pyMOR VectorArray of POD coefficients
    pca_mean: float | None = None
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

        self.compute_pod_with_training_data = False

    def run_pipeline(self, modes: int, train_size: int, compute_pod_with_training_data = False) -> MORResult:
        """
        Run one POD + NN experiment with given number of modes and training size.
        Validation and test sizes adapt automatically according to test = val = (N-train)/2.
        """

        self.sim_data.split_for_train_size(train_size)
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
        pod = self.apply_pod(n_modes=modes, compute_pod_with_training_data)

        # Step 3: train NN with train subset
        self.reduce_with_nn(pod, ann_rel_l2_err=1.0)

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

    def apply_pca(self, epsilon=3e-3, n_modes=None) -> MORResult:
        U = self.sim_data.train_snapshots()
        mean = U.lincomb(np.full(len(U), 1 / len(U)))
        V = (U - mean)
        # Convert epsilon into absolute error and compute POD with this absolute error threshold
        abs_epsilon = epsilon * np.linalg.norm(V.to_numpy())
        if n_modes is None:
            modes, singular_values = pod(V, l2_err=abs_epsilon)
        else:
            modes, singular_values = pod(V, modes=n_modes)
        print("Number of modes: ", len(modes))
        # Store the result
        result = MORResult(pca_mean=mean, epsilon=epsilon,  abs_epsilon=abs_epsilon, modes=modes, singular_values=singular_values)
        self.results.append(result)
        return result

    # apply POD and Neural Network Reduction and append reference to results list
    def apply_pod(self, epsilon=3e-3, n_modes=None, compute_pod_with_training_data = False) -> MORResult:
        """
        Compute POD for the given training data and relative L2 error.
        Returns the MORResult and stores it internally.
        """
        # Convert epsilon into absolute error and compute POD with this absolute error threshold
        abs_epsilon = epsilon * np.linalg.norm(self.sim_data.field_data)
        if compute_pod_with_training_data:
            if n_modes is None:
                modes, singular_values, coefficients = pod(self.sim_data.train_snapshots(), l2_err=abs_epsilon,
                                                           return_reduced_coefficients=True)
            else:
                modes, singular_values, coefficients = pod(self.sim_data.train_snapshots(), modes=n_modes,
                                                           return_reduced_coefficients=True)
        else:
            if n_modes is None:
                modes, singular_values, coefficients = pod(self.sim_data.all_snapshots(), l2_err=abs_epsilon,
                                             return_reduced_coefficients=True)
            else:
                modes, singular_values, coefficients = pod(self.sim_data.all_snapshots(), modes=n_modes,
                                             return_reduced_coefficients=True)
        print("Number of modes: ", len(modes))
        # Store the result
        result = MORResult(coefficients=coefficients, epsilon=epsilon, abs_epsilon=abs_epsilon,
                           modes=modes, singular_values=singular_values)
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
            ann_mse=ann_mse)
        # no lr_scheduler due to LBFGS, learning_rate = 1
        mor_result.reductor = reductor
        mor_result.rom = reductor.reduce(restarts=10, log_loss_frequency=10, lr_scheduler=None)
#TODO TEST PCA
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

    def plot_pod_nn_errors_all_combinations(self, modes_list, train_sizes_list, compute_pod_with_training_data=False):
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
        self.sim_data.split_fixed(seed=42)

        if "errors_gap.npy" in os.listdir("."):
            print("Loading existing error data from 'errors_gap.np', 'errors_pod.np', 'errors_nn.np'")
            errors_gap = np.load("errors_gap.npy")
            errors_pod = np.load("errors_pod.npy")
            errors_nn = np.load("errors_nn.npy")

            n_modes_loaded, n_train_loaded = errors_nn.shape

            # Reconstruct parameter lists to match the loaded arrays.
            modes_list = np.arange(1, n_modes_loaded + 1, dtype=int)
            train_sizes_list = np.arange(1, n_train_loaded + 1, dtype=int)

        else:
            errors_nn = np.zeros((len(modes_list), len(train_sizes_list)))
            errors_pod = np.zeros((len(modes_list), len(train_sizes_list)))
            errors_gap = np.zeros((len(modes_list), len(train_sizes_list)))

            for j, train_size in enumerate(train_sizes_list):
                self.sim_data.split_for_train_size(train_size)
                for i, n_modes in enumerate(modes_list):
                    if compute_pod_with_training_data:
                        if n_modes > train_size:
                            # skip invalid combination
                            errors_nn[i, j] = np.nan
                            errors_pod[i, j] = np.nan
                            errors_gap[i, j] = np.nan
                            continue
                    # Run POD + NN with given parameters
                    self.run_pipeline(modes=n_modes, train_size=train_size, compute_pod_with_training_data)

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
            np.save("errors_pod.npy", errors_pod)
            np.save("errors_nn.npy", errors_nn)
            np.save("errors_gap.npy", errors_gap)

        # Set common log scale limits
        vmin, vmax = 1e-3, 1
        levels = np.logspace(-3, 0, 11)
        ticks = np.logspace(-3, 0, 4)
        cmap = "viridis"

        X, Y = np.meshgrid(train_sizes_list, modes_list)

        # Normalize to [0, 1]
        #max_abs = np.nanmax(np.abs(errors_nn))
        #errors_norm = errors_nn / max_abs

        fig, ax = plt.subplots()

        contour = ax.contourf(X, Y, errors_nn, levels=levels, cmap=cmap,
                               norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = fig.colorbar(contour, ax=ax, ticks=ticks, label=r"$E^{\mathrm{NN}}_{\mathrm{m}}$ [-]")
        # Use LogFormatterSciNotation for scientific notation
        cbar.formatter = ticker.LogFormatterSciNotation(base=10.0)
        cbar.update_ticks()
        cbar.ax.minorticks_off()

        ax.set_xlabel(r"$n_{\mathrm{train}}$")
        ax.set_ylabel(r"$n$")
        ax.set_xlim(1, 80)
        ax.set_ylim(1, 80)
        ax.set_xticks([20, 40, 60, 80])
        ax.set_yticks([20, 40, 60, 80])
        ax.tick_params(top=False, right=False)

        # Create inset axes [width, height, location]
        axins = inset_axes(ax,width="40%", height="40%", loc="upper left",
                           bbox_to_anchor=(0.14, -0.07, 1, 1),
                           bbox_transform=ax.transAxes, borderpad=0)
        # Plot the same data into the inset
        contour_inset = axins.contourf(X, Y, errors_nn, levels=levels, cmap=cmap,
                                       norm=colors.LogNorm(vmin=vmin, vmax=vmax))

        # Set zoomed region
        axins.set_xlim(1, 12)  # zoom-in x range
        axins.set_ylim(1, 12)  # zoom-in y range
        axins.set_xticks([1, 6, 12])
        axins.set_yticks([1, 6, 12])

        # Draw lines connecting inset and main plot
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        plt.tight_layout()
        plt.savefig("Errors_nn.eps", bbox_inches="tight")
        plt.savefig("Errors_nn.png", bbox_inches="tight")

        # Normalize to [0, 1]
        #max_abs = np.nanmax(np.abs(errors_pod))
        #errors_norm = errors_pod / max_abs

        plt.figure()
        contour = plt.contourf(X, Y, errors_pod, levels=levels, cmap=cmap,
                               norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(contour, ticks=ticks, label=r"$E^{\mathrm{POD}}_{\mathrm{m}}$ [-]")
        cbar.formatter = ticker.LogFormatterSciNotation(base=10.0)
        cbar.update_ticks()
        cbar.ax.minorticks_off()

        plt.xlabel(r"$n_{train}$")
        plt.ylabel(r"$n$")
        plt.tight_layout()
        plt.savefig("Errors_pod.eps", bbox_inches="tight")
        plt.savefig("Errors_pod.png", bbox_inches="tight")

        # Normalize to [-1, 1]
        #max_abs = np.nanmax(np.abs(errors_gap))
        #errors_norm = errors_gap / max_abs
        plt.figure()
        plt.semilogy(modes_list, errors_pod[:,-1])

        plt.ylabel(r"$E^{\mathrm{POD}}_{\mathrm{m}}$ [-]")
        plt.xlabel(r"$n$")
        plt.xticks([0, 20, 40, 60, 80, 100])
        plt.tight_layout()
        plt.savefig("Errors_pod_line.eps", bbox_inches="tight")
        plt.savefig("Errors_pod_line.png", bbox_inches="tight")

        plt.figure()
        contour = plt.contourf(X, Y, errors_gap, levels=levels, cmap=cmap,
                               norm=colors.LogNorm(vmin=vmin, vmax=vmax))
        cbar = plt.colorbar(contour, ticks=ticks, label=r"$E^{\mathrm{gap}}_{\mathrm{m}}$ [-]")
        cbar.formatter = ticker.LogFormatterSciNotation(base=10.0)
        cbar.update_ticks()
        cbar.ax.minorticks_off()

        plt.xlabel(r"$n_{train}$")
        plt.ylabel(r"$n$")
        plt.tight_layout()
        plt.savefig("Errors_gap.eps", bbox_inches="tight")
        plt.savefig("Errors_gap.png", bbox_inches="tight")
        plt.show()

    def plot_mixing_index_over_parameters(self, field_data_bulk_file):
        # Load bulk field data (Phi_g)
        field_data_bulk = pickle.load(open(field_data_bulk_file, "rb"))

        Phi_g = field_data_bulk[:, :, :, -1]  # shape: (n_sim, nx, nz)
        Phi_s_NN = self.solve_and_reconstruct_set(self.get_last_result(), self.sim_data.parameters)
        Phi_s = self.sim_data.field_data  # shape: (n_sim, nx, nz)
        params = self.sim_data.parameters  # shape: (n_sim,)

        eps = 1e-12
        n_sim = Phi_g.shape[0]
        dS_true = []
        dS_nn = []

        for i in range(n_sim):
            mask = Phi_g[i] > 0.1

            # True field
            phi_s = np.zeros_like(Phi_s[i], dtype=float)
            phi_s[mask] = Phi_s[i][mask] / Phi_g[i][mask]
            phi_l = 1.0 - phi_s
            phi_s = np.clip(phi_s, eps, 1.0)
            phi_l = np.clip(phi_l, eps, 1.0)
            s_local = np.zeros_like(phi_s)
            s_local[mask] = phi_s[mask] * np.log(phi_s[mask]) + phi_l[mask] * np.log(phi_l[mask])
            phi_total = phi_s + phi_l
            S = np.mean(s_local[mask] * phi_total[mask])
            S_mix = (-np.log(2) * phi_total[mask]).mean()
            dS_true.append(S / S_mix)

            # NN-predicted field
            phi_s_nn = np.zeros_like(Phi_s_NN[i], dtype=float)
            phi_s_nn[mask] = Phi_s_NN[i][mask] / Phi_g[i][mask]
            phi_l_nn = 1.0 - phi_s_nn
            phi_s_nn = np.clip(phi_s_nn, eps, 1.0)
            phi_l_nn = np.clip(phi_l_nn, eps, 1.0)
            s_local_nn = np.zeros_like(phi_s_nn)
            s_local_nn[mask] = phi_s_nn[mask] * np.log(phi_s_nn[mask]) + phi_l_nn[mask] * np.log(phi_l_nn[mask])
            phi_total_nn = phi_s_nn + phi_l_nn
            S_nn = np.mean(s_local_nn[mask] * phi_total_nn[mask])
            S_mix_nn = (-np.log(2) * phi_total_nn[mask]).mean()
            dS_nn.append(S_nn / S_mix_nn)

        dS_true = np.array(dS_true)
        dS_nn = np.array(dS_nn)

        plt.figure()
        plt.plot(params, dS_true, label=r'$\Delta S$')
        plt.plot(params, dS_nn, '--', label=r'$\Delta S^{\mathrm{NN}}$')
        plt.xlabel(r'$s$ [-]')
        plt.ylabel(r'$\Delta S$ [-]')
        plt.legend()
        plt.tight_layout()
        plt.savefig("mixing_index.png", bbox_inches="tight")
        plt.savefig("mixing_index.eps", bbox_inches="tight")
        plt.show()


    def compute_floor_noise(self):
        """
        Compute the floor noise of the simulation data as the minimum non-zero value
        across K time-windowed snapshots with W time windows.

        Returns
        -------
        float
            The computed floor noise value.
        """

        V = SimData.from_files(
            field_file="simulationStudy_timewindows.pickle", testset_file="simulationStudy_test.pickle",
            testset_params=self.sim_data.test_params, x_file="x_s.pickle", z_file="z_s.pickle")
        V.split_fixed(seed=42)
        V.split_for_train_size(train_size=80)

        # Compute mean and standard deviation across time windows
        nWindows = 10  # number of time windows per simulation
        nFiles = V.field_data.shape[0] // nWindows
        nSamples = nWindows*nFiles
        nx, ny = V.field_data.shape[1:]
        space_dim = nx * ny

        # -------------------------------------------------------
        # 1) Reshape field data back to (nFiles, nWindows, nx, ny)
        # -------------------------------------------------------
        V_field = V.field_data.reshape(nFiles, nWindows, nx, ny)

        # ---------------------------------------------------------
        # 2) Compute per-simulation mean and fluctuations
        # ---------------------------------------------------------
        V_mean = np.mean(V_field, axis=1)  # (nFiles, nx, ny)
        V_std = V_field - V_mean[:, None, :, :]  # (nFiles, nWindows, nx, ny)

        # ---------------------------------------------------------
        # 3) Create pyMOR VectorArrays
        # ---------------------------------------------------------
        # Flatten spatial dimensions for pyMOR
        V_mean_vec = NumpyVectorSpace.from_numpy(V_mean.reshape(nFiles, space_dim).T)
        V_field_vec = NumpyVectorSpace.from_numpy(V_field.reshape(nFiles * nWindows, space_dim).T)

        # ---------------------------------------------------------
        # 4) POD on mean fields (get spatial modes)
        # ---------------------------------------------------------
        modes_mean, svals_mean, coeff_mean = pod(V_mean_vec, return_reduced_coefficients=True)
        n = coeff_mean.shape[0]
        print(f"Computed {n} POD modes from mean fields")

        # ---------------------------------------------------------
        # 5) Project all windows onto the POD modes to get coefficients
        # ---------------------------------------------------------
        # Center the full data for consistency
        X_flat = V_field.reshape(nFiles * nWindows, space_dim)

        # Projection using pyMOR inner products
        # coeffs is the same as coeff_mean when we rehape accordingly and take the mean over the windows
        # and rescale coeff_mean by svals_mean
        basis_mat = modes_mean.to_numpy().T  # (M, space_dim)
        coeffs = X_flat @ basis_mat.T  # (nSamples, M)
        coeffs = coeffs.reshape(nFiles, nWindows, n)

        # ---------------------------------------------------------
        # 6) Compute per-mode homogenization noise
        # ---------------------------------------------------------
        coeff_var_per_sim = np.var(coeffs, axis=1, ddof=1)  # (nFiles, M)
        noise_var_per_mode = np.mean(coeff_var_per_sim, axis=0) / nWindows  # (M,)
        noise_std_per_mode = np.sqrt(noise_var_per_mode)

        # ---------------------------------------------------------
        # 7) Relative noise per mode (L2 normalized)
        # ---------------------------------------------------------
        # mean_coeff_per_sim = np.mean(coeffs, axis=1)  # (nFiles, M)
        # rms_mean_coeff_per_mode_centered = np.sqrt(np.mean(mean_coeff_per_sim ** 2, axis=0))  # (M,)
        # relative_noise_per_mode_centered = noise_std_per_mode / (rms_mean_coeff_per_mode_centered)
        # relative_noise_std_per_mode_centered = noise_std_per_mode / rms_mean_coeff_per_mode_centered

        # ---------------------------------------------------------
        # 8) Total field-level relative noise
        # ---------------------------------------------------------
        V_mean_flat = V_mean.reshape(nFiles, space_dim)
        V_std_flat = V_std.reshape(nFiles, nWindows, space_dim)

        std_field_per_sim = np.sqrt(np.mean(np.sum(V_std_flat ** 2, axis=2), axis=1))
        norm_mean_field_per_sim = np.sqrt(np.sum(V_mean_flat ** 2, axis=1))

        relative_field_noise_per_sim = std_field_per_sim / (norm_mean_field_per_sim)
        relative_field_noise_mean = np.mean(relative_field_noise_per_sim)
        relative_field_noise_std = np.std(relative_field_noise_per_sim)

        print(f"Field-level relative noise: {relative_field_noise_mean:.3e} ± {relative_field_noise_std:.3e}")

        rng = np.random.default_rng(123)
        indices = np.arange(nFiles)
        rng.shuffle(indices)
        n_train = int(0.8 * nFiles)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        params_unique = np.unique(V.parameters)
        train_params = [params_unique[i] for i in train_idx]  # list or array matching pyMOR expected format
        val_params = [params_unique[i] for i in val_idx]
        train_params = [Parameters({"VolumeRatio": 1}).parse(p) for p in train_params]
        val_params = [Parameters({"VolumeRatio": 1}).parse(p) for p in val_params]
        # prepare VectorArrays for training/validation snapshots
        train_snap_mat = V_mean.reshape(nFiles, space_dim)[train_idx].T  # (space_dim, n_train)
        val_snap_mat = V_mean.reshape(nFiles, space_dim)[val_idx].T  # (space_dim, n_val)
        train_snapshots = NumpyVectorSpace.from_numpy(train_snap_mat)
        val_snapshots = NumpyVectorSpace.from_numpy(val_snap_mat)
        # ---------------------------------------------------------
        # 9) Neural network reduction & prediction
        # ---------------------------------------------------------
        # Build a reductor using pyMOR neural network machinery
        reductor = NeuralNetworkReductor(
            training_parameters=train_params,
            training_snapshots=train_snapshots,
            validation_parameters=val_params,
            validation_snapshots=val_snapshots,
            reduced_basis=modes_mean,
            ann_mse=1e-1
        )

        rom = reductor.reduce(restarts=10, log_loss_frequency=10, lr_scheduler=None)

        # Predict mean coefficients for all samples
        a_pred_mean = np.zeros((nFiles, n))
        for i, mu in enumerate(params_unique):
            a_pred_mean[i, :] = rom.solve(mu).to_numpy()[:,-1]

        # POD modes already computed from V_mean_vec (uncentered)
        # coeff_mean has shape (M, nFiles) or (nFiles, M) depending on pod implementation
        # rescale by svals_mean from POD to get actual coefficients
        a_true_mean = (np.diag(svals_mean) @ coeff_mean).T  # shape (nFiles, M)
        # TODO Variance correction by stephan, L**2 instead of L??
        # L = a_true_mean.shape[0]
        # rms_mean_coeff_per_mode = np.sqrt(np.sum(a_true_mean ** 2, axis=0) /L)
        rms_mean_coeff_per_mode = np.sqrt(np.mean(a_true_mean ** 2, axis=0))
        # Simulation noise (centered across windows)
        # delta_coeffs = coeffs - np.mean(coeffs, axis=1)[:, None, :]  # (nFiles, nWindows, M)
        noise_std_per_mode = np.sqrt(np.mean(np.var(coeffs, axis=1, ddof=1) / nWindows, axis=0))
        relative_noise_per_mode = noise_std_per_mode / rms_mean_coeff_per_mode
        relative_noise_std_per_mode = noise_std_per_mode / rms_mean_coeff_per_mode

        rmse_per_mode = np.sqrt(np.mean((a_pred_mean - a_true_mean) ** 2, axis=0))
        relative_rmse_per_mode = rmse_per_mode / rms_mean_coeff_per_mode
        # Standard deviation across simulations for each mode
        rmse_std_per_mode = np.std(a_pred_mean - a_true_mean, axis=0, ddof=1)  # (M,)
        relative_rmse_std_per_mode = rmse_std_per_mode / rms_mean_coeff_per_mode

        # -------------------------
        # 10) PLOTTING: relative noise vs mode, and total field relative noise
        # -------------------------
        modes = np.arange(1, n + 1)
        plt.figure()
        plt.semilogy(modes, relative_noise_per_mode, "o-", label=r'$\mathcal{E}_i^\sigma$')
        plt.semilogy(modes, relative_rmse_per_mode, "x--", label=r'$\mathcal{E}_i^{\mathrm{NN}}$')
        plt.xlabel(r'$i$')
        plt.ylabel(r'$\mathcal{E}_i$')
        #plt.title(r"\sigma^\prime = {relative_field_noise_mean:.3e} ± {relative_field_noise_std:.3e}")
        plt.legend()
        plt.tight_layout()
        plt.savefig('FloorNoise_perMode.eps', bbox_inches="tight")
        plt.savefig('FloorNoise_perMode.png', bbox_inches="tight")
        plt.show()

        print("Field-level relative noise across simulations (mean ± std):",
              f"{relative_field_noise_mean:.3e} ± {relative_field_noise_std:.3e}")
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
    # Halton sequence parameters for test set
    testset_params =np.array([1.5, 1.25, 1.75, 1.125, 1.625, 1.375, 1.875, 1.0625, 1.5625, 1.3125])
    # 1. Load data
    sim_data = SimData.from_files(field_file="simulationStudy_s.pickle", testset_file="simulationStudy_test.pickle",
                                  testset_params = testset_params, x_file="x_s.pickle", z_file="z_s.pickle")
    sim_data.split_fixed(seed=42)
    # sim_data.split_for_train_size(60)

    # 2. Create pipeline
    pipeline = PODNNPipeline(sim_data)
    #pipeline.run_pipeline(30,50)
    # pipeline.plot_mixing_index_over_parameters("simulationStudyBulk.pickle")
    # pipeline.apply_pca(n_modes=20)
    #pipeline.compute_floor_noise()
    # pipeline.run_pipeline(89,60)
    # pipeline.plot_pod_error_and_modes(pipeline.get_last_result(), num_modes=4)
    modes_list = np.linspace(1,100, 100).astype(int)
    train_size_list = np.linspace(1,99,99).astype(int)

    pipeline.plot_pod_nn_errors_all_combinations(modes_list, train_size_list)
    # pipeline.plot_pod_nn_errors_all_combinations(modes_list, train_size_list, compute_pod_with_training_data=True)
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