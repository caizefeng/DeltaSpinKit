#!/usr/bin/env python3
# @File    : SpinStaticTask.py
# @Time    : 9/5/2021 10:20 AM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import glob
import os
import shutil
import subprocess
from typing import Dict

import numpy as np

import dskit.configuration as config
from dskit.utils.file_helper import fill_templates


class SpinStaticCalculation:  # need external calls to construct and run
    spin: np.ndarray
    spin_incar: Dict[str, str]

    def __init__(self):
        self.spin = None
        self.spin_incar = {}
        self.dir = None
        self.label = None
        self.workload_label = None
        self.energy = None
        self.rms = None
        self.ef = None

    def set_spin(self, spin, project_real_ratio=1.0):
        self.spin = spin
        magmom_string = np.array2string(spin.reshape(-1) * project_real_ratio, max_line_width=np.inf,
                                        formatter={'float_kind': lambda x: "%12.8f" % x})[1:-1]
        m_constr_string = np.array2string(spin.reshape(-1), max_line_width=np.inf,
                                          formatter={'float_kind': lambda x: "%12.8f" % x})[1:-1]
        # spin_string = " ".join([str(i) for i in self.spin.reshape(-1).tolist()])
        self.spin_incar["MAGMOM"] = magmom_string
        self.spin_incar["M_CONSTR"] = m_constr_string
        self.spin_incar["LAMBDA"] = f"{len(spin) * 3}*0"
        try:
            self.spin_incar["CONSTRL"] = config.task_dict["DeltaSpin"]["CONSTRL"]
        except KeyError:
            self.spin_incar["CONSTRL"] = f"{len(spin) * 3}*1"

    def set_dir(self, task_dir):
        self.dir = task_dir

    def set_label(self, label: Dict):
        self.label = label

    def write_incar(self, template_dir):
        fill_templates(os.path.join(self.dir, "INCAR"), [os.path.join(template_dir, "INCAR")],
                       self.spin_incar)

    def write_other(self, template_dir, working_dir, modified_files=None, extra_files=None):

        if modified_files is None:
            modified_files = []
        if extra_files is None:
            extra_files = []

        other_list = ["POSCAR", "KPOINTS", "POTCAR", config.submit_pattern]
        other_list.extend(extra_files)
        for file_name in other_list:
            if file_name not in modified_files:
                file_path = glob.glob(os.path.join(template_dir, file_name))[0]
                shutil.copy(file_path, self.dir, follow_symlinks=True)  # allow override
            else:
                file_path = glob.glob(os.path.join(working_dir, file_name))[0]
                try:
                    shutil.move(file_path, self.dir)  # 1. create 2. move to the destination
                except shutil.Error:
                    shutil.copy(file_path, self.dir, follow_symlinks=True)
                    os.remove(file_path)

    def run(self):
        sub = subprocess.Popen(config.submit_command, shell=True, cwd=self.dir, stdout=subprocess.PIPE)
        self.workload_label = config.get_task_id(sub.stdout.read())
        sub.kill()

    def get_result(self):
        log_name = config.get_log_name(self.workload_label)
        get_result_shell_path = os.path.join(os.path.dirname(__file__), "utils/energy_force.sh")
        exit_code = subprocess.call(f'sh {get_result_shell_path} >> {log_name}', shell=True, cwd=self.dir)
        error_message = f"No valid output in {self.dir}, " \
                        f"the calculation probably stopped early for some reason or simply didn't use DeltaSpin."
        if exit_code != 0:
            raise ValueError(error_message)
        energy_cmd = f"grep -A 1 'Energy (eV):' {log_name} | tail -1"
        try:
            sub_energy = subprocess.Popen(energy_cmd, shell=True, cwd=self.dir, stdout=subprocess.PIPE)
            self.energy = float(str(sub_energy.stdout.read(), 'utf-8'))
            sub_energy.kill()
            rms_cmd = f"grep -A 1 'RMS (uB):' {log_name} | tail -1"
            sub_rms = subprocess.Popen(rms_cmd, shell=True, cwd=self.dir, stdout=subprocess.PIPE)
            self.rms = float(str(sub_rms.stdout.read(), 'utf-8'))
            sub_rms.kill()
            ef_cmd = f"grep -A 1 'Magnetic Force (eV/uB):' {log_name} | tail -1"
            sub_ef = subprocess.Popen(ef_cmd, shell=True, cwd=self.dir, stdout=subprocess.PIPE)
            self.ef = np.array([float(i) for i in str(sub_ef.stdout.read(), 'utf-8').split()]).reshape(-1, 3)
        except ValueError:
            print(error_message)
            self.energy = 0.0
            self.rms = 0.0
            self.ef = np.zeros_like(self.spin)
