from jeepney.low_level import padding

from pymor.basic import *
from scipy.io import loadmat
from pymor.algorithms.error import *
from pymor.reductors.neural_network import NeuralNetworkReductor

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec

import pickle

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


def plot_full_POD_mode_selection(x, z, Phi, l2_err_epsilon=3e-3):
    # POD relative mean l2 error. relative to the bulk volume fraction magnitude:
    # E = ||u_n - u||_2 / ||u||_2
    # l2_err_rel_mean = 1e-4
    # l2_err_mean = l2_err_rel_mean * np.linalg.norm(Phi)

    l2_err_mean_epsilon = l2_err_epsilon * np.linalg.norm(Phi)

    U = NumpyVectorSpace.from_numpy(Phi.reshape((Phi.shape[0], -1)).T)
    modes, singular_values = pod(U, modes=len(Phi)-1)

    print("Number of modes: ", len(modes))

    abs_err = []
    for n in range(1, len(singular_values) + 1):
        tail_err = np.sum(singular_values[n:] ** 2)
        abs_err.append(np.sqrt(tail_err))

    plt.figure(1)
    plt.semilogy(singular_values, 'x-')
    plt.xlabel('$N$ [-]')
    plt.ylabel('$\sigma$ [-]')
    plt.legend()
    plt.savefig('singularValueDecay.eps', bbox_inches="tight")
    plt.savefig('singularValueDecay.png', bbox_inches="tight")

    plt.figure(2)
    plt.semilogy(abs_err, 'x-')
    plt.axhline(l2_err_mean_epsilon, color="r", linestyle="--", label=r'$\varepsilon=3\times10^{-3}$')
    plt.axvline(15, color="k", linestyle="--", label="$n=15$")
    ax = plt.gca()  # get current axes
    ax.set_xlabel("$n$ [-]")
    # Move the y-label closer using label coordinates
    ax.set_ylabel(r'$\displaystyle \sum_{i = n+1}^R \sigma_i$ [-]')
    ax.yaxis.set_label_coords(-0.08,0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('ModeSelection.eps', bbox_inches="tight", pad_inches=0.2)
    plt.savefig('ModeSelection.png', bbox_inches="tight")
    plt.show()

    npModes = modes.to_numpy()
    npModes = npModes.T.reshape((len(modes), x.shape[0], z.shape[0]))

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # normalise to [-1,1] to keep the negative and positive contributions to the field intact
    vmin = -1
    vmax = 1
    npModes_norm = npModes[:4] / np.max(np.abs(npModes[:4]), axis=(1, 2), keepdims=True)
    contours = []
    for i, ax in enumerate(axs.flat):
        contour = ax.contourf(x[:, :, -1], z[:, :, -1], npModes_norm[i],
                              cmap='seismic', levels=100, vmin=vmin, vmax=vmax)
        contours.append(contour)
        ax.set_title(f'Mode {i + 1}')
    # contour0 = axs[0, 0].contourf(x[:,:,-1], z[:,:,-1], npModes[0,:,:], cmap='viridis', levels=10)
    # fig.colorbar(contour0, ax=axs[0, 0], format='%1.2f',ticks=np.linspace(np.min(npModes[0,:,:]),np.max(npModes[0,:,:]),5))
    # contour1 = axs[0, 1].contourf(x[:,:,-1], z[:,:,-1], npModes[1,:,:], cmap='viridis', levels=10)
    # fig.colorbar(contour1, ax=axs[0, 1], label=r'$\Phi^{\mathrm{s}}$ [-]', format='%1.2f', ticks=np.linspace(np.min(npModes[1,:,:]),np.max(npModes[1,:,:]),5))
    # contour2 = axs[1, 0].contourf(x[:,:,-1], z[:,:,-1], npModes[2,:,:], cmap='viridis', levels=10)
    # fig.colorbar(contour2, ax=axs[1, 0], format='%1.2f', ticks=np.linspace(np.min(npModes[2,:,:]),np.max(npModes[2,:,:]),5))
    # contour3 = axs[1, 1].contourf(x[:,:,-1], z[:,:,-1], npModes[3,:,:], cmap='viridis', levels=10)
    # fig.colorbar(contour3, ax=axs[1, 1], label=r'$\Phi^{\mathrm{s}}$ [-]', format='%1.2f', ticks=np.linspace(np.min(npModes[3,:,:]),np.max(npModes[3,:,:]),5))
    sm = cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])  # required for colorbar
    cbar = fig.colorbar(sm, ax=axs, orientation='vertical',
                        format='%1.2f', ticks=np.linspace(vmin, vmax, 5))
    cbar.set_label(r'$\mathit{\Phi}^{\mathrm{s}\prime}$ [-]')

    fig.text(0.42, 0.03, '$x$ [m]', ha='center', va='center')
    fig.text(0.03, 0.5, '$y$ [m]', ha='center', va='center', rotation='vertical')

    fig.subplots_adjust(hspace=0.55, wspace=0.45, right=0.75, bottom=0.12)
    # plt.tight_layout()
    plt.savefig('Modes.eps', bbox_inches="tight")
    plt.savefig('Modes.png', bbox_inches="tight")
    plt.show()

def load_grid():
    # load the pickle of the spatial evaluation grid data
    x = pickle.load(open("x_s.pickle", "rb"))
    z = pickle.load(open("z_s.pickle", "rb"))
    return x, z

def load_field_data(filename):
    # load the pickle of the respective field data
    field_data = pickle.load(open(filename, "rb"))
    # take only the last timestep (if timeaveraged, reduce dimensions)
    field_data = field_data[:, :, :, -1]
    return field_data

def main():

    x,z = load_grid()
    Phi_s = load_field_data("simulationStudy_s.pickle")
    Phi_g = load_field_data("simulationStudyBulk.pickle")

    # l2_err_epsilon = 3e-3 results in 15 modes
    plot_full_POD_mode_selection(x, z, Phi_s, l2_err_epsilon=3e-3)

    # assemble training, validation and test set
    num_samples = Phi_s.shape[0]
    indices = np.arange(num_samples)

    # Split indices
    train_val_idx, test_idx = train_test_split(indices, test_size=0.1, random_state=42)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=0.11111111, random_state=42)

    # Use indices to split data and parameters
    Phi_s_train = Phi_s[train_idx]
    Phi_s_val = Phi_s[val_idx]
    Phi_s_test = Phi_s[test_idx]

    size_ratios = np.linspace(1.0, 2.0, num_samples)
    size_ratios_train = size_ratios[train_idx]
    size_ratios_val = size_ratios[val_idx]
    size_ratios_test = size_ratios[test_idx]

    # POD relative mean l2 error. relative to the bulk volume fraction magnitude:
    # E = ||u_n - u||_2 / ||u||_2
    l2_err_rel_mean = 3e-3
    l2_err_mean = l2_err_rel_mean * np.linalg.norm(Phi_s)
    # ANN relative mean square error. relative to the bulk volume fraction magnitude:
    # E^2 = ||u_n - u||_2^2 / ||u||_2^2
    l2_err_ann_rel_mean = 7e-3
    ann_mse = l2_err_ann_rel_mean**2 * np.linalg.norm(Phi_s)

    # Plot the contour plot
    plt.figure(1)
    contour = plt.contourf(x[:,:,-1], z[:,:,-1], Phi_s[-1,:,:], cmap='Greys', levels=200)
    plt.ylim(-40,40)
    plt.colorbar(contour, label='Density', format='%1.2f')  # Add a colorbar with label and formatting
    plt.axis("off")
    #Add labels and title
    #plt.xlabel('X-axis')
    #plt.ylabel('Z-axis')
    #plt.title('Contour Plot of Density')

    # x = x.reshape((x.shape[0]*x.shape[1],-1)).T
    # z = z.reshape((z.shape[0]*z.shape[1],-1)).T
    # density = density.reshape((density.shape[0]*density.shape[1],-1)).T
    # Add a grid for better readability
    # plt.grid(True, linestyle='--', alpha=0.7)

    # plt.show(block=False)

    U = NumpyVectorSpace.from_numpy(Phi_s.reshape((Phi_s.shape[0],-1)).T)
    modes, singular_values = pod(U, atol=0, rtol=0, l2_err=l2_err_mean)

    print("Number of modes: ", len(modes))

    # Initialize the list to store the L2 norms
    l2_norm_vector = []
    sigma_square_vector = []
    # create reduced order model
    V = U
    # to create a figure for the error run this loop
    for i in range(1, len(modes) + 1):
        # Select the current number of modes
        reduced_modes = modes[:i]

        # Calculate the approximated Phi_s
        coefficients = V.inner(reduced_modes)
        # V_approx is the linear combination of the reduced modes with the coefficients (coefficients.shape= (N,M)
        V_approx = reduced_modes.lincomb(coefficients[:i].T)
        np_V_approx = V_approx.to_numpy().T.reshape((-1,x.shape[0],z.shape[0]))

        # Calculate the relative L2 norm
        l2_norm = np.linalg.norm(Phi_s[:i,:,:] - np_V_approx)/(np.linalg.norm(Phi_s[:i,:,:]))
        sigma_square = np.sum(singular_values[i:]**2)
        # Append the L2 norm to the list
        l2_norm_vector.append(l2_norm)
        sigma_square_vector.append(sigma_square)


    # error as criteria for modes selection
    # error*numberOfNodes = 1e-6
    # select number of modes
    reduced_modes = modes[:]

    # I know the solution that is why U=V
    coefficients = V.inner(reduced_modes)
    V_approx = reduced_modes.lincomb(coefficients.T)
    np_V_approx = V_approx.to_numpy().T.reshape((Phi_s.shape[0],x.shape[0],z.shape[0]))

    npModes = modes.to_numpy()
    npModes = npModes.T.reshape((len(modes),x.shape[0],z.shape[0]))

    # Calculate the L2 norm of the difference between the Phi_s and the approximated Phi_s
    l2_norm = np.linalg.norm(Phi_s - np_V_approx)/(np.linalg.norm(Phi_s))

    # plt.figure(2)
    # plt.semilogy(singular_values, 'x-')
    # plt.axhline(y=(3e-3 * np.linalg.norm(Phi_s)), linestyle="-", label=r'$\varepsilon$')
    # plt.xlabel('$N$')
    # plt.ylabel('$\sigma$ [-]')
    # plt.legend()
    # plt.savefig('myfile.png', bbox_inches="tight")
    #
    # plt.figure(3)
    # plt.semilogy(l2_norm_vector, 'x-')
    # plt.xlabel('$n$')
    # plt.ylabel(r'$\frac{| \Phi-\Phi_n |_2}{| \Phi |_2}$ [-]')
    # plt.tight_layout()
    #
    # sigma_square_vector.sort()
    # sigma_square_vector[-1] = l2_err_mean ** 2
    # sigma_square_vector.sort()
    # plt.figure(4)
    # plt.semilogy(sigma_square_vector, 'x-')
    # plt.axhline(y=(3e-3 * np.linalg.norm(Phi_s))**2, linestyle="-", label=r'$\varepsilon$')
    # plt.axvline(x=15, linestyle="--")
    # plt.xlabel('$n$')
    # plt.ylabel(r'$\sum_{i}^R \sigma_i^2$ [-]')
    # plt.tight_layout()
    # plt.show()
    #
    # abs_err = []
    # for n in range(1, len(singular_values)+1):
    #     tail_energy = np.sum(singular_values[n:]**2)
    #     abs_err.append(np.sqrt(tail_energy))
    #
    # plt.figure()
    # plt.semilogy(abs_err, 'x-')
    # plt.axhline(3e-3 * np.linalg.norm(Phi_s), color="r", linestyle="--", label="Tolerance")
    # plt.axvline(15, color="k", linestyle="--", label="Chosen modes")
    # plt.xlabel("Number of modes")
    # plt.ylabel(r'$\sum_{i = n+1}^R \sigma_i^2$ [-]')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    # # normalise to [-1,1] to keep the negative and positive contributions to the field intact
    # vmin = -1
    # vmax = 1
    # npModes_norm = npModes[:4] / np.max(np.abs(npModes[:4]), axis=(1, 2), keepdims=True)
    # contours = []
    # for i, ax in enumerate(axs.flat):
    #     contour = ax.contourf(x[:, :, -1], z[:, :, -1], npModes_norm[i],
    #                           cmap='seismic', levels=100, vmin=vmin, vmax=vmax)
    #     contours.append(contour)
    #     ax.set_title(f'Mode {i + 1}')
    # # contour0 = axs[0, 0].contourf(x[:,:,-1], z[:,:,-1], npModes[0,:,:], cmap='viridis', levels=10)
    # # fig.colorbar(contour0, ax=axs[0, 0], format='%1.2f',ticks=np.linspace(np.min(npModes[0,:,:]),np.max(npModes[0,:,:]),5))
    # # contour1 = axs[0, 1].contourf(x[:,:,-1], z[:,:,-1], npModes[1,:,:], cmap='viridis', levels=10)
    # # fig.colorbar(contour1, ax=axs[0, 1], label=r'$\Phi^{\mathrm{s}}$ [-]', format='%1.2f', ticks=np.linspace(np.min(npModes[1,:,:]),np.max(npModes[1,:,:]),5))
    # # contour2 = axs[1, 0].contourf(x[:,:,-1], z[:,:,-1], npModes[2,:,:], cmap='viridis', levels=10)
    # # fig.colorbar(contour2, ax=axs[1, 0], format='%1.2f', ticks=np.linspace(np.min(npModes[2,:,:]),np.max(npModes[2,:,:]),5))
    # # contour3 = axs[1, 1].contourf(x[:,:,-1], z[:,:,-1], npModes[3,:,:], cmap='viridis', levels=10)
    # # fig.colorbar(contour3, ax=axs[1, 1], label=r'$\Phi^{\mathrm{s}}$ [-]', format='%1.2f', ticks=np.linspace(np.min(npModes[3,:,:]),np.max(npModes[3,:,:]),5))
    # sm = cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    # sm.set_array([])  # required for colorbar
    # cbar = fig.colorbar(sm, ax=axs, orientation='vertical',
    #                     format='%1.2f', ticks=np.linspace(vmin, vmax, 5))
    # cbar.set_label(r'$\mathit{\Phi}^{\mathrm{s}\prime}$ [-]')
    #
    # # fig.subplots_adjust(hspace=0.35)
    #
    # fig.text(0.42, 0.03, '$x$ [m]', ha='center', va='center')
    # fig.text(0.03, 0.5, '$y$ [m]', ha='center', va='center', rotation='vertical')
    #
    # fig.subplots_adjust(hspace=0.55, wspace=0.45, right=0.75, bottom=0.12)
    # # plt.tight_layout()
    # plt.savefig('Modes.eps', bbox_inches="tight")
    # plt.savefig('Modes.png', bbox_inches="tight")

    # plt.show(block=True)
    # Define contour levels, excluding zero
    levels = np.linspace(np.min(Phi_s[Phi_s > 0]), np.max(Phi_s), num=8)
    additional_levels = [0.01, 0.1, 0.5]

    levels = np.concatenate([levels[1:], additional_levels])
    levels = np.sort(levels)

    fig, axes = plt.subplots(2, 1)
    titles = [r'$u_N(\mu)$', r'$u(\mu)$']
    data = [np_V_approx[-1, :, :], Phi_s[-1, :, :]]

    contours = []
    for ax, dat, title in zip(axes, data, titles):
        contour = ax.contourf(x[:, :, -1], z[:, :, -1], dat, cmap='viridis', levels=levels)
        ax.set_xlim(-33, 33)
        ax.set_ylim(-33, 33)
        ax.set_aspect('equal')
        ax.set_title(title)
        contours.append(contour)

    norm = colors.Normalize(vmin=np.min(Phi_s), vmax=np.max(Phi_s))
    mappable = cm.ScalarMappable(norm=norm, cmap='viridis')
    cbar = fig.colorbar(
        mappable, ax=axes, label=r'$\Phi^{\mathrm{s}}$ [-]', format='%1.2f',
        ticks=np.linspace(np.min(Phi_s[Phi_s > 0]), np.max(Phi_s), 5)
    )

    fig.text(0.65, 0.04, '$x$ [m]', ha='center', va='center')
    fig.text(0.44, 0.5, '$y$ [m]', ha='center', va='center', rotation='vertical')
    fig.subplots_adjust(hspace=0.55, right=0.75, bottom=0.14)
    fig.savefig('myfile3.png', bbox_inches="tight", dpi=300)
    plt.show(block=False)

    # Principle of POD, Eckhard Schmidt theorem (see Wikipedia)
    # Snapshots = U \sum(V.T) = \sum_i=1^N \sigma_i u_i v_i^T
    # Snapshots \approx \sum_i=1^k \sigma_i u_i v_i^T, where k << N

    print("L2 norm: ", (U-V_approx).norm()/U.norm())


    parameters = Parameters({"volumeRatio": 1})
    training_set = [parameters.parse(p) for p in size_ratios_train]
    training_snapshots = NumpyVectorSpace.from_numpy(Phi_s_train.reshape((Phi_s_train.shape[0], -1)).T)
    validation_set = [parameters.parse(p) for p in size_ratios_val]
    validation_snapshots = NumpyVectorSpace.from_numpy(Phi_s_val.reshape((Phi_s_val.shape[0], -1)).T)
    test_set = [parameters.parse(p) for p in size_ratios_test]
    test_snapshots = NumpyVectorSpace.from_numpy(Phi_s_test.reshape((Phi_s_test.shape[0], -1)).T)

    # like_basis would be an ann_mse of: (np.sum(Phi_s**2) - np.sum(singular_values**2)) / 100),
    # but with this high error threshold it does not find a neural network below the prescribed error.
    reductor_bulk = NeuralNetworkReductor(training_parameters= training_set,
                                          training_snapshots=training_snapshots,
                                          validation_parameters=validation_set,
                                          validation_snapshots=validation_snapshots,
                                          reduced_basis=reduced_modes,
                                          ann_mse=ann_mse)
    rom_bulk = reductor_bulk.reduce(restarts=10, log_loss_frequency=10)


    # U = NumpyVectorSpace.from_numpy(Phi_s.reshape((Phi_s.shape[0],-1)).T)
    # modes, singular_values = pod(U, l2_err=l2_err)

    rom_solutions = np.zeros((num_samples, 100, 100))
    nn_predictions_train = np.zeros((80, 100, 100))
    nn_predictions_val = np.zeros((10, 100, 100))
    nn_predictions_test = np.zeros((10, 100, 100))

    rom_solutions = np_V_approx
    for i, mu in enumerate(training_set):
        nn_predictions_train[i, :, :] = reductor_bulk.reconstruct(rom_bulk.solve(mu)).to_numpy().reshape((x.shape[0], z.shape[0]))
    for i, mu in enumerate(validation_set):
        nn_predictions_val[i,:,:] = reductor_bulk.reconstruct(rom_bulk.solve(mu)).to_numpy().reshape((x.shape[0], z.shape[0]))
    for i, mu in enumerate(test_set):
        nn_predictions_test[i,:,:] = reductor_bulk.reconstruct(rom_bulk.solve(mu)).to_numpy().reshape((x.shape[0], z.shape[0]))

    # Discuss, replace with singular values?
    pod_error = np.linalg.norm(Phi_s - rom_solutions, axis=(1, 2)) / np.linalg.norm(Phi_s, axis=(1,2))
    nn_error_train = np.linalg.norm(Phi_s_train - nn_predictions_train, axis=(1, 2)) / np.linalg.norm(Phi_s_train, axis=(1,2))
    nn_error_val = np.linalg.norm(Phi_s_val - nn_predictions_val, axis=(1, 2)) / np.linalg.norm(Phi_s_val, axis=(1,2))
    nn_error_test = np.linalg.norm(Phi_s_test - nn_predictions_test, axis=(1, 2)) / np.linalg.norm(Phi_s_test, axis=(1,2))

    # sizeRatios = np.linspace(1.00,2.00,100)
    # TODO make epsilon relative and project according to relation
    # |x|_{l2}=sqrt(sum x_i^2)
    # |x|_{MSE}=(1/N) sum x_i^2
    #
    # |x|_{MSE} = 1/N |x|_{l2}^2
    plt.figure(11)
    plt.semilogy(size_ratios, pod_error, ".", label='$u_n$')
    plt.semilogy(size_ratios_train, nn_error_train, ".", color="red", label=r'$\tilde{u}_n^{\mathrm{train}}$')
    plt.semilogy(size_ratios_val, nn_error_val, "x", color="red", label=r'$\tilde{u}_n^{\mathrm{val}}$')
    plt.semilogy(size_ratios_test, nn_error_test, "^", color="red", label=r'$\tilde{u}_n^{\mathrm{test}}$')
    plt.axhline(y=l2_err_rel_mean, linestyle="-", color="black", label=r'$\varepsilon$')
    # plt.axhline(y=l2_err_ann_rel_mean, linestyle="-", color="red", label=r'$\tilde{\varepsilon}$')
    plt.xlabel(r'$s$ [-]')
    plt.ylabel(r'$E$ [-]')
    plt.xticks(np.arange(1, 3, 1))
    plt.ylim(1e-3, 1e-1)
    # plt.title('Comparison of Reconstruction Errors')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    plt.savefig('POD_NN_Errors.eps', bbox_inches="tight")
    plt.savefig('POD_NN_Errors.png', bbox_inches="tight")
    plt.show()

if __name__=='__main__':
     main()