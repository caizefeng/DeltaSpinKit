from collections import OrderedDict
from typing import Dict, List, Iterable


def gen_name(run_dict: Dict, run_extra: List):
    """function to generate names for each run in TensorBoard"""
    name_dict = OrderedDict()
    name = ""
    for key, value in run_dict.items():
        if isinstance(value, str):
            value_in_name = value
            name = '_'.join((name, value_in_name))
            name_dict[key] = 1
        elif isinstance(value, Iterable):
            value_in_name = [str(i) for i in value]
            name = '_'.join((name, *value_in_name))
            name_dict[key] = len(value)
        else:
            value_in_name = str(value)
            name = '_'.join((name, value_in_name))
            name_dict[key] = 1
    name = '_'.join((name, *run_extra))
    name_dict["extra"] = len(run_extra)
    return name, name_dict
