#!/usr/bin/env python3
# @File    : SpinSample.py
# @Time    : 9/5/2021 11:00 AM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import os
import subprocess
from datetime import datetime
from time import sleep
from typing import List

import numpy as np

import dskit.configuration as config
from dskit.spin_static_calculation import SpinStaticCalculation
from dskit.utils.file_helper import mkdir_without_override
from dskit.utils.math_helper import normalize_to_norm


class SpinParallel:
    sample_spin: List[SpinStaticCalculation]

    def __init__(self, name="test"):
        self.spin_ground = None
        self.atom_rotate = None
        self.rotate_spin_norm = None
        self.sample_spin = []
        self.template_dir = None
        self.batch_size = None
        self.num_atom = None
        self.task_name = name
        self.save_dir = None
        self.name_keys = []
        self.modified_files = []

    def set_spin_ground(self, spin, atom_rotate: List[int]):
        spin = spin.astype("float64")
        self.atom_rotate = atom_rotate
        if config.task_dict["norm_spins"]["is_norm"]:
            self.rotate_spin_norm = config.task_dict["norm_spins"]["moment"]
            self.spin_ground = normalize_to_norm(spin, self.rotate_spin_norm)
        else:
            self.spin_ground = spin
            self.rotate_spin_norm = np.linalg.norm(self.spin_ground[self.atom_rotate][0])
        self.num_atom = self.spin_ground.shape[0]

    def set_template(self, template_dir):
        self.template_dir = template_dir
        # TODO: inspect templates are complete or not

    def set_name_keys(self):
        pass

    def construct_dir(self, parent_dir):
        self.save_dir = os.path.join(parent_dir, self.task_name)
        mkdir_without_override(self.save_dir)
        mkdir_without_override(os.path.join(self.save_dir, "samples"))

    def construct_all(self):
        self.set_name_keys()
        for task in self.sample_spin:
            name_string_list = []
            for key in self.name_keys:
                if isinstance(task.label[key], tuple):
                    name_string_list.extend([str(i) for i in task.label[key]])
                else:
                    name_string_list.extend([str(task.label[key])])
            task_dir = os.path.join(self.save_dir, "samples", "_".join(name_string_list))
            mkdir_without_override(task_dir)
            task.set_dir(task_dir)
            task.write_incar(self.template_dir)
            if "extra_files" in config.task_dict:
                task.write_other(self.template_dir, self.save_dir, self.modified_files, config.task_dict["extra_files"])
            else:
                task.write_other(self.template_dir, self.save_dir, self.modified_files)

    def run(self, batch_size):
        count = 0
        for task in self.sample_spin:
            if (not task.workload_label) and (count <= batch_size):
                task.run()
                count += 1

    def wait_until_all_done(self, batch_index):
        while True:
            all_job_done = True
            sub_poll = subprocess.Popen(config.check_command, shell=True, stdout=subprocess.PIPE)
            all_job = str(sub_poll.stdout.read(), 'utf-8')
            for task in self.sample_spin:
                if (task.workload_label and (task.workload_label in all_job)) or config.polling_error(all_job):
                    all_job_done = False
                    break
            if all_job_done:
                # sub_poll.kill()
                print(f"Batch {batch_index + 1} done.")
                print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                break
            sleep(30)

    def run_and_wait(self, batch_size=100):
        print("Starting...")
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
        self.batch_size = batch_size
        num_batch = len(self.sample_spin) // self.batch_size + 1
        for batch_index in range(num_batch):
            self.run(self.batch_size)
            self.wait_until_all_done(batch_index)
        print("All jobs done.")
        print(datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
