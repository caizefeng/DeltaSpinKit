import os
from string import Template

from dskit.utils.config_helper import json_entry

task_dict = {}
workload_dict = {}

submit_pattern = ""
submit_command = ""
check_command = ""
get_task_id = None
get_log_name = None
polling_error = None


def init():
    global task_dict, workload_dict
    description = "Toolkit for manipulating and analyzing DeltaSpin data"
    example_config_path = os.path.join(os.path.dirname(__file__), "config/PES_sphere.json")
    default_workload_path = os.path.join(os.path.dirname(__file__), "config/workload_pbs.json")
    task_dict, workload_dict = json_entry(description, example_config_path, default_workload_path)


def init_workload():
    global task_dict, workload_dict, submit_pattern, submit_command, check_command
    global get_task_id, get_log_name, polling_error
    submit_pattern = workload_dict["submit_file_pattern"]

    if workload_dict["workload_manager"] == "SLURM":
        submit_binary = "sbatch"
        check_command = "squeue -u `whoami`"

        def get_task_label_func(return_string):
            return str(return_string, 'utf-8').strip().split()[-1]

    elif workload_dict["workload_manager"] == "PBS":
        submit_binary = "qsub"
        check_command = "qstat -u `whoami`"

        def get_task_label_func(return_string):
            return str(return_string, 'utf-8').strip()

    else:
        raise ValueError("Unsupported workload manager \"{}\"".format(workload_dict["workload_manager"]))

    if not workload_dict["use_default_log_pattern"]:
        def get_log_name_func(task_id):
            return Template(workload_dict["log_file_pattern"]).substitute(id=task_id)
    else:
        if workload_dict["workload_manager"] == "SLURM":
            def get_log_name_func(task_id):
                return "slurm-{}.out".format(task_id)
        elif workload_dict["workload_manager"] == "PBS":
            raise ValueError(
                "No default naming pattern for PBS logging files. You have to specify one in \"log_file_pattern\".")
        else:
            raise ValueError("Unsupported workload manager \"{}\"".format(workload_dict["workload_manager"]))

    if "error_string" in workload_dict:
        def polling_error_func(polling_return):
            return workload_dict["error_string"] in polling_return
    else:
        def polling_error_func(_):
            return False

    submit_command = "{} {}".format(submit_binary, submit_pattern)
    get_task_id = get_task_label_func
    get_log_name = get_log_name_func
    polling_error = polling_error_func
