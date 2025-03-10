"""
Data loader for the matter power spectrum
"""

import os, json
from typing import List, Optional
import re

import numpy as np
import h5py

from .gpemulator_singlebin import _map_params_to_unit_cube as input_normalize
from .data_loader_dgmgp import interpolate

# put the folder name on top, easier to find and modify
def folder_name(
    num1: int,
    res1: int,
    box1: int,
    num2: int,
    res2: int,
    box2: int,
    z: float,
    selected_ind,
):
    return "Matterpower_{}_res{}box{}_{}_res{}box{}_z{}_ind-{}".format(
        num1,
        res1,
        box1,
        num2,
        res2,
        box2,
        "{:.2g}".format(z).replace(".", "_"),
        "-".join(map(str, selected_ind)),
    )


def convert_h5_to_txt(
    lf_filename: str = "data/dmo_60_res128box256/cc_emulator_powerspecs.hdf5",
    hf_filename: str = "data/dmo_24_res512box256/cc_emulator_powerspecs.hdf5",
    test_filename: str = "data/dmo_10_res512box256/cc_emulator_powerspecs.hdf5",
    lf_json: str = "data/dmo_60_res128box256/emulator_params.json",
    hf_json: str = "data/dmo_24_res512box256/emulator_params.json",
    test_json: str = "data/dmo_10_res512box256/emulator_params.json",
    hf_selected_ind: Optional[List[int]] = [
        0,
        1,
        2,
    ],  ## these should be the index in the "LF" LHD
    num_lf: int = 60, # number of Low-fidelity design points
):
    """
    Convert the h5 files Martin gave me to txt files to be read by the dataloader.

    Keys:
    ----
    bounds : the bounds of the parameter prior 
    powerspecs : Matter power spectrum (rebinned)
    kfmpc : k bins in Mpc/h unit
    params : un-normalized input parameters
    zout : redshifts
    scale_factors : 1 / (zout + 1)

    Parameters:
    ----
    hf_selected_ind : the selected subset for hf training set within the lf LHD.
    num_lf : number of LF design points, assuming to be params[:num_lf].

    Note: save each redshift as separate folders
    """

    f_lf = h5py.File(lf_filename, "r")
    f_hf = h5py.File(hf_filename, "r")
    f_test = h5py.File(test_filename, "r")

    with open(lf_json, "r") as param:
        param_lf = json.load(param)
    with open(hf_json, "r") as param:
        param_hf = json.load(param)
    with open(test_json, "r") as param:
        param_test = json.load(param)

    param_limits = np.array(param_lf["bounds"])
    assert np.all(param_limits == np.array(param_hf["bounds"]))
    assert np.all(param_limits == np.array(param_test["bounds"]))

    # make sure all keys are in the file
    keys = ["powerspecs", "kfmpc", "params", "zout"]
    for key in keys:
        assert key in f_lf.keys()
        assert key in f_hf.keys()
        assert key in f_test.keys()

    print("Low-fidelity file:")
    print("----")
    print("Resolution:", param_lf["npart"])
    print("Box (Mpc/h):", param_lf["box"])
    print("Shape of redshfits", f_lf["zout"].shape)
    print("Shape of params", f_lf["params"].shape)
    print("Shape of kfmpc", f_lf["kfmpc"].shape)
    print("Shape of powerspecs", f_lf["powerspecs"].shape)
    print("\n")

    # use kfmpc so all redshifts use the same k bins
    kfmpc_lf = f_lf["kfmpc"][()]
    # Different fidelities have different k bins
    # assert np.all(np.abs(kfmpc - f_hf["kfmpc"][()]) < 1e-10)

    zout = f_lf["zout"][()]
    assert np.all((zout - f_hf["zout"][()]) < 1e-10)

    # power spectra, all redshifts
    powerspecs_lf = f_lf["powerspecs"][()]
    # Select a subset of lf points for training.
    assert num_lf <= len(powerspecs_lf)
    powerspecs_lf = powerspecs_lf[:num_lf, :, :]
    print("=> Shape of powerspecs", powerspecs_lf.shape)

    # input parameters
    x_train_lf = f_lf["params"][()]
    # Select a subset of lf points for training.
    assert num_lf <= len(x_train_lf)
    x_train_lf = x_train_lf[:num_lf, :]

    # read power spectrum at ith redshift
    # power spectrum | z = ?
    def get_powerspec_at_z(i: int, powerspecs: np.ndarray) -> np.ndarray:
        return powerspecs[:, i, :]

    # some checking
    last_powerspec = get_powerspec_at_z(len(zout) - 1, powerspecs_lf)
    assert len(last_powerspec) == x_train_lf.shape[0]
    assert last_powerspec.shape[1] == len(kfmpc_lf)

    first_powerspec = get_powerspec_at_z(0, powerspecs_lf)
    assert len(first_powerspec) == x_train_lf.shape[0]
    assert first_powerspec.shape[1] == len(kfmpc_lf)

    print("High-fidelity file:")
    print("----")
    print("Resolution:", param_hf["npart"])
    print("Box (Mpc/h):", param_hf["box"])
    print("Shape of redshfits", f_hf["zout"].shape)
    print("Shape of params", f_hf["params"].shape)
    print("Shape of kfmpc", f_hf["kfmpc"].shape)
    print("Shape of powerspecs", f_hf["powerspecs"].shape)
    print("Selected indices:", f_hf["selected_ind"][()])
    print("\n")

    # We only need to save a subset we want to have in the HF training
    selected_ind = f_hf["selected_ind"][()]
    if hf_selected_ind is not None:
        ind_hf_sims = np.isin(selected_ind, hf_selected_ind,)
        print("Check: selected inds are,", selected_ind[ind_hf_sims])
        assert np.all(selected_ind[ind_hf_sims] == hf_selected_ind)
    else:
        ind_hf_sims = np.isin(selected_ind, np.arange(len(selected_ind)),)
        print("Check: selected inds are,", selected_ind[ind_hf_sims])
        assert selected_ind[ind_hf_sims] == hf_selected_ind

    # power spectra, all redshifts
    powerspecs_hf = f_hf["powerspecs"][()]

    # input parameters
    x_train_hf = f_hf["params"][()]

    powerspecs_hf = powerspecs_hf[ind_hf_sims, :, :]
    x_train_hf = x_train_hf[ind_hf_sims, :]
    print("-> Shape of powerspecs", powerspecs_hf.shape)
    print("-> Selected indices:", selected_ind[ind_hf_sims][()])

    kfmpc_hf = f_hf["kfmpc"][()]
    assert np.all(np.abs(kfmpc_hf - f_test["kfmpc"][()]) < 1e-10)

    # some checking
    last_powerspec = get_powerspec_at_z(len(zout) - 1, powerspecs_hf)
    assert len(last_powerspec) == f_hf["params"][()][ind_hf_sims, :].shape[0]
    assert last_powerspec.shape[1] == len(kfmpc_hf)

    first_powerspec = get_powerspec_at_z(0, powerspecs_hf)
    assert len(first_powerspec) == f_hf["params"][()][ind_hf_sims, :].shape[0]
    assert first_powerspec.shape[1] == len(kfmpc_hf)

    # test files: same resolution as high-fidelity
    print("Test file:")
    print("----")
    print("Resolution:", param_test["npart"])
    print("Box (Mpc/h):", param_test["box"])
    print("Shape of redshfits", f_test["zout"].shape)
    print("Shape of params", f_test["params"].shape)
    print("Shape of kfmpc", f_test["kfmpc"].shape)
    print("Shape of powerspecs", f_test["powerspecs"].shape)
    print("\n")

    # power spectra, all redshifts
    powerspecs_test = f_test["powerspecs"][()]

    # input parameters
    x_train_test = f_test["params"][()]

    # some checking
    last_powerspec = get_powerspec_at_z(len(zout) - 1, powerspecs_test)
    assert len(last_powerspec) == f_test["params"].shape[0]
    assert last_powerspec.shape[1] == len(kfmpc_hf)

    first_powerspec = get_powerspec_at_z(0, powerspecs_test)
    assert len(first_powerspec) == f_test["params"].shape[0]
    assert first_powerspec.shape[1] == len(kfmpc_hf)

    # output training files, one redshift per folder
    for i, z in enumerate(zout):
        print("Preparing training files in {:.3g}".format(z))

        powerspec_lf = get_powerspec_at_z(i, powerspecs_lf)
        powerspec_hf = get_powerspec_at_z(i, powerspecs_hf)
        powerspec_test = get_powerspec_at_z(i, powerspecs_test)

        outdir = folder_name(
            len(x_train_lf),
            param_lf["npart"],
            param_lf["box"],
            len(x_train_hf),
            param_hf["npart"],
            param_hf["box"],
            z,
            selected_ind=hf_selected_ind,
        )

        this_outdir = os.path.join("data", "processed", outdir,)
        os.makedirs(
            this_outdir, exist_ok=True,
        )

        # Interpolate the LF,
        # and limit the HF kmax to the maximum of LF
        # Min k bins LF <= k bins HF <= Max k bins LF
        ind_min = (np.log10(kfmpc_lf).min() <= np.log10(kfmpc_hf)) & (
            np.log10(kfmpc_hf) <= np.log10(kfmpc_lf).max()
        )

        # interpolate: interp(log10_k, Y_lf)(log10_k[ind_min])
        # I do want to interpolate in loglog scale.
        # TODO: Think about if our smooth prior is on linear or log scale?
        powerspec_lf_new = interpolate(
            np.log10(kfmpc_lf), np.log10(powerspec_lf), np.log10(kfmpc_hf)[ind_min]
        )
        powerspec_lf_new = 10**powerspec_lf_new
        powerspec_hf_new = powerspec_hf[:, ind_min]
        powerspec_test_new = powerspec_test[:, ind_min]
        kfmpc_new = kfmpc_hf[ind_min]

        # new k bins: same k bin over different fidelities
        print("New k bins:")
        print("----")
        print("Shape of kfmpc", kfmpc_new.shape)
        print("Shape of LF powerspecs", powerspec_lf_new.shape)
        print("Shape of HF powerspecs", powerspec_hf_new.shape)
        print("Shape of Test powerspecs", powerspec_test_new.shape)
        print("\n")

        # only power spec needs a loop
        np.savetxt(
            os.path.join(this_outdir, "train_output_fidelity_0.txt"), np.log10(powerspec_lf_new)
        )
        np.savetxt(
            os.path.join(this_outdir, "train_output_fidelity_1.txt"), np.log10(powerspec_hf_new)
        )
        np.savetxt(os.path.join(this_outdir, "test_output.txt"), np.log10(powerspec_test_new))

        np.savetxt(os.path.join(this_outdir, "train_input_fidelity_0.txt"), x_train_lf)
        np.savetxt(os.path.join(this_outdir, "train_input_fidelity_1.txt"), x_train_hf)
        np.savetxt(os.path.join(this_outdir, "test_input.txt"), x_train_test)

        np.savetxt(os.path.join(this_outdir, "input_limits.txt"), param_limits)
        np.savetxt(os.path.join(this_outdir, "kf.txt"), np.log10(kfmpc_new))


class PowerSpecs:
    """
    A data loader to load multi-fidelity training and test data

    Assume two fidelities.
    """

    def __init__(self, folder: str = "data/50_LR_3_HR/", n_fidelities: int = 2):
        self.n_fidelities = n_fidelities

    def read_from_txt(self, folder: str = "data/50_LR_3_HR/",):
        """
        Read the multi-fidelity training set from txt files (they have a specific data structure)

        See here https://github.com/jibanCat/matter_multi_fidelity_emu/tree/main/data/50_LR_3_HR
        """
        # training data
        self.X_train = []
        self.Y_train = []
    
        for i in range(self.n_fidelities):
            x_train = np.loadtxt(
                os.path.join(folder, "train_input_fidelity_{}.txt".format(i))
            )
            y_train = np.loadtxt(
                os.path.join(folder, "train_output_fidelity_{}.txt".format(i))
            )

            self.X_train.append(x_train)
            self.Y_train.append(y_train)

        # parameter limits for normalization
        self.parameter_limits = np.loadtxt(os.path.join(folder, "input_limits.txt"))

        # testing data
        self.X_test = []
        self.Y_test = []
        self.X_test.append(np.loadtxt(os.path.join(folder, "test_input.txt")))
        self.Y_test.append(np.loadtxt(os.path.join(folder, "test_output.txt")))

        # load k bins (in log)
        self.kf = np.loadtxt(os.path.join(folder, "kf.txt"))

        assert len(self.kf) == self.Y_test[0].shape[1]
        assert len(self.kf) == self.Y_train[0].shape[1]

    @property
    def X_train_norm(self):
        """
        Normalized input parameters
        """
        x_train_norm = []
        for x_train in self.X_train:
            x_train_norm.append(input_normalize(x_train, self.parameter_limits))

        return x_train_norm

    @property
    def X_test_norm(self):
        """
        Normalized input parameters
        """
        x_test_norm = []
        for x_test in self.X_test:
            x_test_norm.append(input_normalize(x_test, self.parameter_limits))

        return x_test_norm

    @property
    def Y_train_norm(self):
        """
        Normalized training output. Subtract the low-fidelity data with
        their sample mean. Don't change high-fidelity data.
        """
        y_train_norm = []
        for y_train in self.Y_train[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train[-1])

        return y_train_norm


class PowerSpecsMedianNorm:
    """
    A data loader to load multi-fidelity training and test data

    Assume two fidelities, conditioning on a single redshift.

    Normalizing using median power, keep the scales be linear (instead of log).
    """

    def __init__(self,  n_fidelities: int = 2):
        self.n_fidelities = n_fidelities

    def read_from_array(self, kf: np.ndarray, X_train_list: List[np.ndarray], Y_train_list: List[np.ndarray],
            X_test: List[np.ndarray], Y_test: List[np.ndarray], parameter_limits: np.ndarray):
        """
        Assign training/test set without using the txt files.
        Note that specific data structure needs to be fullfilled.

        See here https://github.com/jibanCat/matter_multi_fidelity_emu/tree/main/data/50_LR_3_HR
        """
        # training data
        self.X_train = X_train_list
        self.Y_train = Y_train_list

        assert self.n_fidelities == len(self.X_train)

        # test : in List, List[array], but only one element
        self.X_test = X_test
        self.Y_test = Y_test

        assert len(X_test) == 1
        assert len(Y_test) == 1
        
        self.parameter_limits = parameter_limits
        self.kf = kf

        assert len(self.kf) == self.Y_test[0].shape[1]
        assert len(self.kf) == self.Y_train[0].shape[1]


    def read_from_txt(self, folder: str = "data/50_LR_3_HR/",):
        """
        Read the multi-fidelity training set from txt files (they have a specific data structure)

        See here https://github.com/jibanCat/matter_multi_fidelity_emu/tree/main/data/50_LR_3_HR
        """
        # training data
        self.X_train = []
        self.Y_train = []
        for i in range(self.n_fidelities):
            x_train = np.loadtxt(
                os.path.join(folder, "train_input_fidelity_{}.txt".format(i))
            )
            y_train = np.loadtxt(
                os.path.join(folder, "train_output_fidelity_{}.txt".format(i))
            )

            self.X_train.append(x_train)
            self.Y_train.append(y_train)

        # parameter limits for normalization
        self.parameter_limits = np.loadtxt(os.path.join(folder, "input_limits.txt"))

        # testing data
        self.X_test = []
        self.Y_test = []
        self.X_test.append(np.loadtxt(os.path.join(folder, "test_input.txt")))
        self.Y_test.append(np.loadtxt(os.path.join(folder, "test_output.txt")))

        # load k bins (in log)
        self.kf = np.loadtxt(os.path.join(folder, "kf.txt"))

        assert len(self.kf) == self.Y_test[0].shape[1]
        assert len(self.kf) == self.Y_train[0].shape[1]

    @property
    def X_train_norm(self):
        """
        Normalized input parameters
        """
        x_train_norm = []
        for x_train in self.X_train:
            x_train_norm.append(input_normalize(x_train, self.parameter_limits))

        return x_train_norm

    @property
    def X_test_norm(self):
        """
        Normalized input parameters
        """
        x_test_norm = []
        for x_test in self.X_test:
            x_test_norm.append(input_normalize(x_test, self.parameter_limits))

        return x_test_norm

    @property
    def Y_train_norm(self):
        """
        Normalized training output. Subtract the low-fidelity data with
        their sample mean. Don't change high-fidelity data.
        """
        y_train_norm = []

        #Normalise the flux vectors by the median power spectrum.
        #This ensures that the GP prior (a zero-mean input) is close to true.
        medind = np.argsort(np.mean(self.Y_train[0], axis=1))[np.shape(self.Y_train[0])[0]//2]

        self.scalefactors = self.Y_train[0][medind,:]

        y_normspectra = []
        for y_train in self.Y_train:
            normspectra = y_train / self.scalefactors - 1.

            y_normspectra.append(normspectra)

        return y_normspectra
    



class StellarMassFunctions(PowerSpecs):
    """
    A data loader for stellar mass functions from CAMELS.
    """
    # redefine the init with different loading methods:
    def __init__(self, folder: str = "data/illustris_smfs/1000_LR_3_HR_test0", n_fidelities: int = 2):
        self.n_fidelities = n_fidelities

        # training data
        self.X_train = []
        self.Y_train = []
        for i in range(n_fidelities):
            x_train = np.loadtxt(
                os.path.join(folder, "train_input_fidelity_{}.txt".format(i))
            )
            y_train = np.loadtxt(
                os.path.join(folder, "train_output_fidelity_{}.txt".format(i))
            )

            self.X_train.append(x_train)
            self.Y_train.append(y_train)

        # parameter limits for normalization
        self.parameter_limits = np.loadtxt(os.path.join(folder, "input_limits.txt"))

        # testing data
        self.X_test = []
        self.Y_test = []
        self.X_test.append(np.loadtxt(os.path.join(folder, "test_input.txt")))
        self.Y_test.append(np.loadtxt(os.path.join(folder, "test_output.txt")))

        if len(self.X_test[0].shape) == 1:
            self.X_test[0] = self.X_test[0][None, :]

        if len(self.Y_test[0].shape) == 1:
            self.Y_test[0] = self.Y_test[0][None, :]

        # Currently no bins for SMFs (TODO: Ask Yongseok)
        # self.kf = np.loadtxt(os.path.join(folder, "kf.txt"))

        # assert len(self.kf) == self.Y_test[0].shape[1]
        # assert len(self.kf) == self.Y_train[0].shape[1]


    @property
    def Y_train_log10(self):
        """
        Transform from log1p to log10, and filling up the zero bins with minimum.
        """
        y_train_log10 = []

        for y_train in self.Y_train:
            # Assume the log1p in data
            y_train_linear = np.expm1(y_train)
            # check if there are zeros
            ind = y_train_linear == 0
            if np.any(ind):
                # fill the zeros with minimum
                min_nonzero = np.min(y_train_linear[y_train_linear > 0])
                print("Warning: There are zeros in the data. filling them with minimum. ", min_nonzero)
                y_train_linear[ind] = min_nonzero
                self.min_nonzero = min_nonzero

            y_train_log10.append(np.log10(y_train_linear))

        return y_train_log10

    @property
    def Y_test_log10(self):
        """
        Transform from log1p to log10, and filling up the zero bins with minimum.
        """
        y_test_log10 = []

        for y_test in self.Y_test:
            # Assume the log1p in data
            y_test_linear = np.expm1(y_test)
            # check if there are zeros
            ind = y_test_linear == 0
            if np.any(ind):
                # fill the zeros with minimum
                min_nonzero = np.min(y_test_linear[y_test_linear > 0])
                print("Warning: There are zeros in the data. filling them with minimum. ", min_nonzero)
                y_test_linear[ind] = min_nonzero
                self.min_nonzero = min_nonzero

            y_test_log10.append(np.log10(y_test_linear))

        return y_test_log10

    @property
    def Y_train_norm_log10(self):
        """
        Normalized training output. Subtract the low-fidelity data with
        their sample mean. Don't change high-fidelity data.
        """
        y_train_norm = []
        for y_train in self.Y_train_log10[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train_log10[-1])

        return y_train_norm
    
    @property
    def Y_test_norm_log10(self):
        """
        Normalized test output. Subtract the low-fidelity data with
        their sample mean. Don't change high-fidelity data.
        """
        y_test_norm = []
        for y_test in self.Y_test_log10:
            mean = y_test.mean(axis=0)
            y_test_norm.append(y_test - mean)
        
        return y_test_norm


    @property
    def Y_train_norm(self):
        """
        Normalized training output. Subtract the low-fidelity data with
        their sample mean.
        """
        y_train_norm = []
        for y_train in self.Y_train[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train[-1])

        return y_train_norm

    @property
    def Y_test_norm(self):
        """
        Normalized test output. Subtract the low-fidelity data with
        their sample mean.
        """
        y_train_norm = []
        for y_train in self.Y_train[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train[-1])

        return y_train_norm

class SMFDataLoader:
    def __init__(
        self,
        param_file,
        lh_smf_file,
        low_smf_file,
        high_smf_file,
        additional_low_smf_files=None,
        additional_high_smf_files=None,
        param_names=None,
    ):
        self.param_file = param_file
        self.lh_smf_file = lh_smf_file
        self.low_smf_file = low_smf_file
        self.high_smf_file = high_smf_file
        self.additional_low_smf_files = additional_low_smf_files or []
        self.additional_high_smf_files = additional_high_smf_files or []
        self.param_names = param_names or [
            "Omega_m",
            "sigma_8",
            "A_SN1",
            "A_AGN1",
            "A_SN2",
            "A_AGN2",
        ]

        self.lh_smfs = None
        self.low_smfs = None
        self.high_smfs = None
        self.params_values = None
        self.pair_param_values = None
        self.parameter_limits = np.array(
            [
                (0.1, 0.5),  # Omega_8
                (0.6, 1.0),  # sigma_8
                (0.25, 4.0),  # A_SN1
                (0.25, 4.0),  # A_AGN1
                (0.50, 2.0),  # A_SN2
                (0.50, 2.0),  # A_AGN2
            ]
        )

        self.log10_mass_bins = np.array(
            [8.15, 8.45, 8.75, 9.05, 9.35, 9.65, 9.95, 10.25, 10.55, 10.85]
        )
        self.mass_bins = 10**self.log10_mass_bins
        self.load_data()

    def load_data(self):
        # Read parameters
        size = 1000
        n_params = 6
        self.params_values = np.full((size, n_params), fill_value=np.nan)

        with open(self.param_file, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "").split(" ")
                line = list(filter(lambda l: l != "", line))

                if "LH" in line[0]:
                    re_find = re.findall("LH_([0-9]+)", line[0])
                    num = int(re_find[0])
                    values = np.array(list(map(lambda l: float(l), line[1:-1])))
                    self.params_values[num, :] = values

        # Read SMF data
        self.lh_smfs = np.load(self.lh_smf_file)
        self.low_smfs = np.load(self.low_smf_file)
        self.high_smfs = np.load(self.high_smf_file)

        for lf_file in self.additional_low_smf_files:
            self.low_smfs = np.concatenate([self.low_smfs, np.load(lf_file)])

        for hf_file in self.additional_high_smf_files:
            self.high_smfs = np.concatenate([self.high_smfs, np.load(hf_file)])

        # Concatenate the LF-HF paired simulations
        optimal_index = np.array(
            [596, 623, 329, 188]
        )  # Change this to your specific indices
        self.pair_param_values = np.array(
            [
                [
                    0.23470767,
                    0.68477296,
                    0.29091749,
                    1.37444453,
                    0.52004166,
                    1.38033561,
                ],
                [
                    0.25516079,
                    0.97785488,
                    0.47781017,
                    0.41655329,
                    0.53097153,
                    1.11179847,
                ],
                [
                    0.39536673,
                    0.85751988,
                    2.09561002,
                    0.27822487,
                    0.51237612,
                    1.27292889,
                ],
                [
                    0.40583122,
                    0.66674317,
                    0.82548774,
                    3.53390616,
                    0.84607057,
                    0.75517335,
                ],
            ]
        )
        self.pair_param_values = np.concatenate(
            [self.pair_param_values, self.params_values[optimal_index]]
        )

        # assign optimal index :
        self.optimal_index = np.full(
            (len(self.pair_param_values),), fill_value=-1
        ).astype(int)
        self.optimal_index[len(self.pair_param_values) - len(optimal_index) :] = (
            optimal_index
        )
        # index indicating the elements of the optimal parameter set in the pair_param_values array
        self.order_optimal_index = np.arange(len(self.pair_param_values))[
            len(self.pair_param_values) - len(optimal_index) :
        ]

        # remove the optimal parameter set from the parameter values
        self.params_values = np.delete(
            self.params_values,
            self.optimal_index[self.optimal_index != -1],
            axis=0,
        )

        # Add the pair parameter values to the parameter values
        self.params_values = np.concatenate(
            [self.params_values, self.pair_param_values]
        )

        # For the SMF, also remove the optimal index SMFs
        self._Y_lf = np.delete(
            self.lh_smfs, self.optimal_index[self.optimal_index != -1], axis=0
        )
        self._Y_lf = np.concatenate([self._Y_lf, self.low_smfs])
        # This is simple, as there is on LH set for the high fidelity SMF
        self._Y_hf = self.high_smfs

    @property
    def Y_lf(self):
        """Returns the low fidelity SMF data, this is LF + additional LF."""
        return self._Y_lf

    @property
    def Y_hf(self):
        """Returns the high fidelity SMF data."""
        return self._Y_hf

    @property
    def empirical_smf_uncertainty(self):
        """This is estimated by the difference between the optimal and the LH set SMFs."""
        diff = (
            self.low_smfs[self.order_optimal_index]
            / self.lh_smfs[self.optimal_index[self.optimal_index != -1]]
            - 1
        )
        return np.abs(diff).mean(axis=0)

    def plot_smf(self, idx, ax):
        """Plots the low and high fidelity SMFs for a given parameter set index, with dashed lines for LH set."""
        # Plot the low and high SMFs
        ax.plot(self.log10_mass_bins, self.low_smfs[idx, :], label="Low")
        ax.plot(self.log10_mass_bins, self.high_smfs[idx, :], label="High")

        # Plot dashed line for LH set Low SMF
        if idx in self.order_optimal_index:
            # This indicate the potential inherent numerical error in calculating SMF
            ax.plot(
                self.log10_mass_bins,
                self.lh_smfs[self.optimal_index[idx], :],
                ls="--",
                color="C0",
                label="LH set Low SMF",
            )

        # Plot the empirical uncertainty as 1 sigma error bars
        ax.fill_between(
            self.log10_mass_bins,
            self.low_smfs[idx, :] * (1 + self.empirical_smf_uncertainty),
            self.low_smfs[idx, :] * (1 - self.empirical_smf_uncertainty),
            alpha=0.3,
            color="C0",
        )

        ax.set_xlabel("log10 (M) [$M_\odot$]")
        ax.set_ylabel("Stellar Mass Function")
        ax.set_title(
            "|".join(
                [
                    f"{pn}: {value:.2g}"
                    for pn, value in zip(self.param_names, self.pair_param_values[idx])
                ]
            )
        )
        ax.legend()

    def plot_all_smf(self):
        """Creates subplots for all SMF data."""
        fig, ax = plt.subplots(4, 2, figsize=(20, 30))
        for i in range(len(self.pair_param_values)):
            row, col = divmod(i, 2)
            self.plot_smf(i, ax[row, col])
        plt.tight_layout()