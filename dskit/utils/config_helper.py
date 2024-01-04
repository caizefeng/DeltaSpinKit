import argparse
import json
import os
import sys

from dskit import __version__


def json_entry(description, example_config_path, default_workload_path):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-c", nargs='?', help="JSON settings")
    parser.add_argument("config", nargs='?', help="JSON settings")
    parser.add_argument("-w", "--workload", nargs='?', help="Workload manager settings")
    parser.add_argument('-V', '--version', action='version', version='%(prog)s ' + __version__)
    parser.add_argument("--example", action='store_true', help="view example configuration JSON at {}".format(
        os.path.abspath(example_config_path)))
    args = parser.parse_args()

    if args.example:
        editor = os.environ.get('EDITOR', 'vim')  # use `vim` as default
        os.system("{} {}".format(editor, example_config_path))
        sys.exit(0)

    if args.c:
        config_path = args.c
    elif args.config:
        config_path = args.config
    else:
        raise ValueError('No configuration file provided, use `--example` to see a example JSON for configuration!')

    if args.workload:
        workload_path = args.workload
    else:
        workload_path = default_workload_path

    with open(config_path, "r") as f:
        config_dict = json.load(f)
    with open(workload_path, "r") as f:
        workload_dict = json.load(f)

    return config_dict, workload_dict
