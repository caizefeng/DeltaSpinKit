from dskit.spin_parallel import SpinParallel


class SpinMomentOptimize(SpinParallel):

    def __init__(self, name="test"):
        super().__init__(name)

    # def sample(self, grid: Tuple, end_point):
    #     self.thetas, self.phis, self.energy_matrix = spherical_uniform(grid, end_point)
    #     self.energy_matrix_backward = np.zeros_like(self.energy_matrix)
    #     self.energy_matrix_forward = np.zeros_like(self.energy_matrix)
    #     self.rms_matrix = np.zeros_like(self.energy_matrix)
    #
    #     self.ef_matrix = np.zeros((self.thetas.shape[0], self.phis.shape[0], self.num_atom, 3))
    #     self.ef_matrix_spherical_one_atom = np.zeros((self.thetas.shape[0], self.phis.shape[0], 3))
    #     self.ef_matrix_theta_grad = np.zeros_like(self.energy_matrix)
    #     self.ef_matrix_theta_constrain = np.zeros_like(self.energy_matrix)
    #
    #     for i, theta in enumerate(self.thetas):
    #         for j, phi in enumerate(self.phis):
    #             for diff_label in ["c", "b", "f"]:
    #                 task = SpinStaticCalculation()
    #                 spin = copy.deepcopy(self.spin_ground)
    #                 theta_exact = theta
    #                 if diff_label == "b":
    #                     theta_exact = theta - self.delta
    #                 elif diff_label == "f":
    #                     theta_exact = theta + self.delta
    #                 spin[self.atom_rotate] = sphe2cart(self.spin_norm, theta_exact, phi)
    #
    #                 task.set_label({"q_value": q, "q_direction": q_direction, "moment": m_moment})
    #                 task.set_spin(spin)
    #                 self.sample_spin.append(task)
    #
    # def set_name_keys(self):
    #     self.name_keys = ["moment"]
    #
    # def compute(self):
    #     os.chdir(self.save_dir)
    #     for task in self.sample_spin:
    #         task.get_result()
    #         if task.label["diff"] == "c":
    #             self.energy_matrix[task.label["position"]] = task.energy
    #             self.rms_matrix[task.label["position"]] = task.rms
    #             self.ef_matrix[task.label["position"]] = task.ef
    #
    #             r = self.spin_norm
    #             theta = task.label["angle"][0]
    #             phi = task.label["angle"][1]
    #             jacobi_now = jacobi(r, theta, phi)
    #             # [dr dtheta dphi]
    #             self.ef_matrix_spherical_one_atom[task.label["position"]] = task.ef[self.atom_rotate] @ jacobi_now
    #
    #         elif task.label["diff"] == "b":
    #             self.energy_matrix_backward[task.label["position"]] = task.energy
    #
    #         elif task.label["diff"] == "f":
    #             self.energy_matrix_forward[task.label["position"]] = task.energy
    #
    #     self.ef_matrix_theta_grad = -(self.energy_matrix_forward - self.energy_matrix_backward) / (2 * self.delta)
    #     self.ef_matrix_theta_constrain = self.ef_matrix_spherical_one_atom[:, :, 1]
    #     np.save("theta.npy", self.thetas)
    #     np.save("phi.npy", self.phis)
    #     np.save("energy.npy", self.energy_matrix)
    #     np.save("rms.npy", self.rms_matrix)
    #     np.save("ef_grad.npy", self.ef_matrix_theta_grad)
    #     np.save("ef_constrain.npy", self.ef_matrix_theta_constrain)
    #
    # def draw(self, metric="compare", phi_index="0"):
    #     os.chdir(self.save_dir)
    #     if metric == "compare":
    #         fig = plt.figure()  # type: Figure
    #         ax = fig.add_subplot(111)  # type: Axes
    #         theta = np.load("theta.npy")
    #         ef_grad = np.load("ef_grad.npy")
    #         ef_constrain = np.load("ef_constrain.npy")
    #
    #         ax.scatter(theta, ef_grad[:, phi_index], c='b',
    #                    label=r'$-B^{\mathrm{grad}} = -\frac{\partial E}{\partial \theta}$')
    #         ax.scatter(theta, ef_constrain[:, phi_index], c='r', label=r'$-B^{\mathrm{con}}$')
    #         ax.legend()
    #         ax.set_xlabel(r'$\theta$ (rad)')
    #         ax.set_xlim(np.min(theta), np.max(theta))
    #         ax.set_ylabel(r'$g\nu_BB$ (eV)')
    #         ax.axhline(c='k', ls="--", lw=0.5)
    #         ax.set_title(
    #             "Comparison of the effective field",
    #             fontsize='10')
    #         xticks = ax.get_xticks()
    #         # yticks = ax.get_yticks()
    #
    #         ax2 = fig.add_axes([0.26, 0.42, 0.22, 0.22])  # type: Axes
    #         ax2.scatter(theta, ef_grad[:, phi_index] - ef_constrain[:, phi_index], c='k', s=3)
    #         ax2.set_xlim(np.min(theta), np.max(theta))
    #         ax2.set_xticks(xticks[:-1:2])
    #         # ax2.set_yticks(yticks[::2])
    #         ax2.set_ylabel(r'$\Delta$')
    #         ax2.axhline(c='k', ls="--", lw=0.4)
    #         fig.savefig("compare_grad_constrain.png")