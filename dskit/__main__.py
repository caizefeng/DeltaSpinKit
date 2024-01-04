#!/usr/bin/env python3
# @File    : __main__.py
# @Time    : 9/5/2021 3:35 PM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
from dskit.utils.build_spiral import build_spiral

import dskit.configuration as config
from dskit.task import pes_sampling, plot, optimize
from dskit.utils.incar_supercell import create_incar_supercell


def main():
    config.init()
    config.init_workload()
    if config.task_dict["task"] == "PES":
        pes_sampling(config.task_dict)
    elif config.task_dict["task"] == "plot":
        plot(config.task_dict)
    elif config.task_dict["task"] == "opt":
        optimize(config.task_dict)
    elif config.task_dict["task"] == "spiral":
        build_spiral(config.task_dict, config.task_dict["spiral_initial"])
    elif config.task_dict["task"] == "supercell":
        create_incar_supercell(config.task_dict)
    else:
        raise ValueError("Unsupported DeltaSpinKit task \"{}\"".format(config.task_dict["task"]))


if __name__ == '__main__':
    main()

