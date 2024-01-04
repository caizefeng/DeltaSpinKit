#!/usr/bin/env python3
# @File    : spherical_sample.py
# @Time    : 7/31/2023 10:31 AM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import copy
import os
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import dskit.configuration as config
from dskit.spin_parallel import SpinParallel
from dskit.spin_static_calculation import SpinStaticCalculation
from dskit.utils.math_helper import sphe2cart, jacobi
from dskit.utils.plot_helper import force_aspect, fit_centralize_heatmap, modify_if_single_tick
from dskit.utils.sampling import spherical_uniform


class ThetaDiffSample(SpinParallel):

    def __init__(self, delta: float = 0.01, name="test"):
        super().__init__(name)
        self.thetas = None
        self.phis = None
        self.delta = delta
        self.energy_matrix = None
        self.rms_matrix = None
        self.ef_matrix = None
        self.ef_matrix_spherical_one_atom = None

        self.energy_matrix_backward = None
        self.energy_matrix_forward = None
        self.ef_matrix_theta_grad = None
        self.ef_matrix_theta_constrain = None

    def sample(self, grid: Tuple, end_point):
        if len(self.atom_rotate) != 1:
            raise ValueError("Differentiating effective field calculation only supports one-atom rotation.")
        self.thetas, self.phis, self.energy_matrix = spherical_uniform(grid, end_point)
        self.energy_matrix_backward = np.zeros_like(self.energy_matrix)
        self.energy_matrix_forward = np.zeros_like(self.energy_matrix)
        self.rms_matrix = np.zeros_like(self.energy_matrix)

        self.ef_matrix = np.zeros((self.thetas.shape[0], self.phis.shape[0], self.num_atom, 3))
        self.ef_matrix_spherical_one_atom = np.zeros((self.thetas.shape[0], self.phis.shape[0], 3))
        self.ef_matrix_theta_grad = np.zeros_like(self.energy_matrix)
        self.ef_matrix_theta_constrain = np.zeros_like(self.energy_matrix)

        for i, theta in enumerate(self.thetas):
            for j, phi in enumerate(self.phis):
                for diff_label in ["c", "b", "f"]:
                    task = SpinStaticCalculation()
                    spin = copy.deepcopy(self.spin_ground)
                    theta_exact = theta
                    if diff_label == "b":
                        theta_exact = theta - self.delta
                    elif diff_label == "f":
                        theta_exact = theta + self.delta
                    spin[self.atom_rotate] = sphe2cart(self.rotate_spin_norm, theta_exact, phi)
                    task.set_label({"angle": (theta, phi), "position": (i, j), "diff": diff_label})
                    task.set_spin(spin)
                    self.sample_spin.append(task)

    def set_name_keys(self):
        self.name_keys = ["angle", "diff"]

    def compute(self):
        for task in self.sample_spin:
            task.get_result()
            if task.label["diff"] == "c":
                self.energy_matrix[task.label["position"]] = task.energy
                self.rms_matrix[task.label["position"]] = task.rms
                self.ef_matrix[task.label["position"]] = task.ef

                r = self.rotate_spin_norm
                theta = task.label["angle"][0]
                phi = task.label["angle"][1]
                jacobi_now = jacobi(r, theta, phi)
                # [dr dtheta dphi]
                self.ef_matrix_spherical_one_atom[task.label["position"]] = task.ef[self.atom_rotate][0] @ jacobi_now

            elif task.label["diff"] == "b":
                self.energy_matrix_backward[task.label["position"]] = task.energy

            elif task.label["diff"] == "f":
                self.energy_matrix_forward[task.label["position"]] = task.energy

        self.ef_matrix_theta_grad = -(self.energy_matrix_forward - self.energy_matrix_backward) / (2 * self.delta)
        self.ef_matrix_theta_constrain = self.ef_matrix_spherical_one_atom[:, :, 1]
        np.save(os.path.join(self.save_dir, "theta.npy"), self.thetas)
        np.save(os.path.join(self.save_dir, "phi.npy"), self.phis)
        np.save(os.path.join(self.save_dir, "energy.npy"), self.energy_matrix)
        np.save(os.path.join(self.save_dir, "rms.npy"), self.rms_matrix)
        np.save(os.path.join(self.save_dir, "ef_grad.npy"), self.ef_matrix_theta_grad)
        np.save(os.path.join(self.save_dir, "ef_constrain.npy"), self.ef_matrix_theta_constrain)

    def draw(self, metric="compare", phi_index="0"):

        import matplotlib as mpl
        mpl.rcParams['xtick.top'] = True
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.right'] = True
        mpl.rcParams['ytick.direction'] = 'in'

        os.chdir(self.save_dir)
        if metric == "compare":
            fig = plt.figure()  # type: Figure
            ax = fig.add_subplot(111)  # type: Axes
            theta = np.rad2deg(np.load("theta.npy"))
            ef_grad = np.load("ef_grad.npy")
            ef_constrain = np.load("ef_constrain.npy")

            ax.scatter(theta, ef_grad[:, phi_index], c='b',
                       label=r'$|M| B^{\mathrm{eff}}_{\theta} = -\frac{\partial E}{\partial \theta}$')
            ax.scatter(theta, ef_constrain[:, phi_index], c='r', label=r'$-|M|B^{\mathrm{con}}_{\theta}$')
            ax.legend()
            ax.set_xlabel(r'$\theta\,(^\circ)$')
            ax.set_xlim(np.min(theta), np.max(theta))
            ax.set_ylabel(r'$B$ (eV)')
            ax.axhline(c='k', ls="--", lw=0.5)
            if ("no_title" in config.task_dict) and (not config.task_dict["no_title"]):
                ax.set_title("Comparison of the effective field", fontsize='10')
            xticks = ax.get_xticks()
            # yticks = ax.get_yticks()

            if "inset_location" in config.task_dict:
                ax2 = fig.add_axes(config.task_dict["inset_location"])  # type: Axes
            else:
                ax2 = fig.add_axes([0.25, 0.25, 0.25, 0.25])  # type: Axes
            ax2.scatter(theta, ef_grad[:, phi_index] - ef_constrain[:, phi_index], c='k', s=3)
            ax2.set_xlim(np.min(theta), np.max(theta))
            ax2.set_xticks(xticks[:-1:2])
            # ax2.set_yticks(yticks[::2])
            ax2.set_ylabel(r'$\Delta$ (eV)')
            ax2.axhline(c='k', ls="--", lw=0.4)
            fig.savefig("compare_grad_constrain.png", dpi=1000)


class AllSphereSample(SpinParallel):
    def __init__(self, name="test"):
        super().__init__(name)
        self.thetas = None
        self.phis = None
        self.thetas_carte = None
        self.phis_carte = None

        self.energy_matrix = None
        self.rms_matrix = None

    def sample(self, grid: Tuple, end_point):
        self.thetas, self.phis, self.energy_matrix = spherical_uniform(grid, end_point)
        self.rms_matrix = np.zeros_like(self.energy_matrix)
        for i, theta in enumerate(self.thetas):
            for j, phi in enumerate(self.phis):
                task = SpinStaticCalculation()
                spin = copy.deepcopy(self.spin_ground)
                spin[self.atom_rotate] = sphe2cart(self.rotate_spin_norm, theta, phi)
                task.set_spin(spin)
                task.set_label({"angle": (theta, phi), "position": (i, j)})
                self.sample_spin.append(task)

    def set_name_keys(self):
        self.name_keys = ["angle"]

    def compute(self):
        np.save(os.path.join(self.save_dir, "theta.npy"), self.thetas)
        np.save(os.path.join(self.save_dir, "phi.npy"), self.phis)
        for task in self.sample_spin:
            task.get_result()
            self.energy_matrix[task.label["position"]] = task.energy
            self.rms_matrix[task.label["position"]] = task.rms
        np.save(os.path.join(self.save_dir, "energy.npy"), self.energy_matrix)
        np.save(os.path.join(self.save_dir, "rms.npy"), self.rms_matrix)

    def draw(self, metric, is_degree=False, mark_ground=False):
        import matplotlib as mpl
        mpl.rcParams['xtick.top'] = True
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.right'] = True
        mpl.rcParams['ytick.direction'] = 'in'

        os.chdir(self.save_dir)
        if not is_degree:
            thetas = np.rad2deg(np.load("theta.npy"))
            phis = np.rad2deg(np.load("phi.npy"))
        else:
            thetas = np.load("theta.npy")
            phis = np.load("phi.npy")

        if metric == "energy":
            energy_matrix = np.load("energy.npy")

            if config.task_dict["relative_energy"]:
                energy_matrix = energy_matrix - np.min(energy_matrix)

            fig = plt.figure()  # type: Figure
            ax = fig.add_subplot(111)  # type: Axes
            # fit 2D in heatmap
            extent_array = fit_centralize_heatmap(phis, thetas)
            im = ax.imshow(energy_matrix, cmap='coolwarm',
                           extent=extent_array)
            if mark_ground:
                min_index = np.unravel_index(np.argmin(energy_matrix, axis=None), energy_matrix.shape)
                ax.scatter([phis[min_index[1]], ], [thetas[min_index[0]], ], c='r', s=30, marker='x')

            force_aspect(ax, aspect=1)

            modify_if_single_tick(ax, phis, thetas)

            ax.set_xlabel(r'$\phi\,(^\circ)$')
            ax.set_ylabel(r'$\theta\,(^\circ)$')

            if ("no_title" in config.task_dict) and (not config.task_dict["no_title"]):
                ax.set_title("Magnetic Anisotropy for ONE Atom from DeltaSpin", fontsize='10')

            ticks = np.linspace(np.min(energy_matrix), np.max(energy_matrix), 8, endpoint=True)
            cb = fig.colorbar(ax=ax, mappable=im, ticks=ticks)
            cb.ax.set_yticklabels(["{:6.3f}".format(i) for i in ticks], fontsize='7')
            fig.savefig("MAE.png", dpi=1000)

        elif metric == "rms":
            rms_matrix = np.load("rms.npy")
            fig = plt.figure()  # type: Figure
            ax = fig.add_subplot(111)  # type: Axes

            extent_array = fit_centralize_heatmap(phis, thetas)
            im = ax.imshow(rms_matrix, cmap='viridis',
                           extent=extent_array)
            force_aspect(ax, aspect=1)
            modify_if_single_tick(ax, phis, thetas)

            ax.set_xlabel(r'$\phi\,(^\circ)$')
            ax.set_ylabel(r'$\theta\,(^\circ)$')
            ax.set_title("Root Mean Square of Spin Constraints from DeltaSpin", fontsize='10')

            ticks = np.linspace(np.min(rms_matrix), np.max(rms_matrix), 8, endpoint=True)
            cb = fig.colorbar(ax=ax, mappable=im, ticks=ticks)
            cb.ax.set_yticklabels(["{:5.2e}".format(i) for i in ticks], fontsize='7')
            fig.savefig("MAR.png", dpi=1000)
