#!/usr/bin/env python3
# @File    : SpinStaticCall.py
# @Time    : 11/1/2021 11:32 AM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import os
import subprocess
from time import sleep

from dskit.utils.file_helper import mkdir_without_override
from dskit.spin_static_calculation import SpinStaticCalculation
import dskit.configuration as config

class SpinSerial:

    def __init__(self, name="optimize"):
        self.task_name = name
        self.task = None
        self.template_dir = None
        self.parent_dir = None
        self.sample_dir = None
        self.save_dir = None
        self.name_keys = []
        self.count = 0

    def set_template(self, template_dir):
        self.template_dir = template_dir

    def set_name_keys(self):
        self.name_keys = "count"

    def construct(self, parent_dir):
        self.parent_dir = parent_dir
        self.set_name_keys()
        self.save_dir = os.path.join(self.parent_dir, self.task_name)
        self.sample_dir = os.path.join(self.save_dir, "steps")
        mkdir_without_override(self.save_dir)
        mkdir_without_override(self.sample_dir)

    def __call__(self, x, *args, **kwargs):
        self.task = SpinStaticCalculation()
        self.task.set_label({"count": str(self.count)})
        self.task.set_spin(x)
        self.construct_single()
        self.run_and_wait_until_done()
        self.task.get_result()
        self.count += 1
        return self.task.energy, -self.task.ef

    def construct_single(self):
        name_string_list = []
        for key in self.name_keys:
            if isinstance(self.task.label[key], tuple):
                name_string_list.extend([str(i) for i in self.task.label[key]])
            else:
                name_string_list.extend([str(self.task.label[key])])
            task_dir = os.path.join(self.sample_dir, "_".join(name_string_list))
            mkdir_without_override(task_dir)
            self.task.set_dir(task_dir)
            self.task.write_incar(self.template_dir)
            self.task.write_other(self.template_dir)

    def run_and_wait_until_done(self):
        self.task.run()
        while True:
            sub_qstat = subprocess.Popen(config.check_command, shell=True, stdout=subprocess.PIPE)
            all_job = str(sub_qstat.stdout.read(), 'utf-8')
            job_done = not (
                    (self.task.workload_label and (self.task.workload_label in all_job))
                    or config.polling_error(all_job)
            )
            if job_done:
                sub_qstat.kill()
                print("Jobs done.")
                os.system("date")
                break
            sleep(30)
