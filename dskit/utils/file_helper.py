#!/usr/bin/env python3
# @File    : file_helper.py
# @Time    : 9/5/2021 11:37 AM
# @Author  : Zavier Cai
# @Email   : caizefeng18@gmail.com
import os
import subprocess
from string import Template


def copy_with_status(from_dir, to_dir, file_name):
    from_path = os.path.join(from_dir, file_name)
    to_path = to_dir
    cmd = f"cp -r {from_path} {to_path}"
    status = subprocess.call(cmd, shell=True)
    if status != 0:
        if status < 0:
            print(f"Copy {file_name} from {from_dir} to {to_dir} killed by signal {status}")
        else:
            print(f"Copy {file_name} from {from_dir} to {to_dir} failed with return code - {status}")


def fill_templates(out_path, proto_path_list, all_dict):
    for i, proto_path in enumerate(proto_path_list):
        with open(proto_path, 'r') as f_in:
            temp = Template(f_in.read())
            out = temp.substitute(all_dict)  # `all_dict` could be larger than what `temp` requires
            if i == 0:
                f_out = open(out_path, 'w')
            else:
                f_out = open(out_path, 'a')
            f_out.write(out)
            f_out.close()


def mkdir_without_override(path):
    if not os.path.exists(path):
        os.makedirs(path)
