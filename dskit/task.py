import numpy as np

from dskit.spherical_sample import AllSphereSample, ThetaDiffSample
from dskit.spin_serial import SpinSerial


def pes_sampling(config_dict):  # dskit_scan

    if config_dict["type"] == "spherical_distribution":
        batch = AllSphereSample(config_dict["name"])
    elif config_dict["type"] == "effective_comparison":
        batch = ThetaDiffSample(config_dict["diff_delta"], config_dict["name"])
    else:
        raise ValueError("Unsupported PES calculating type \"{}\"".format(config_dict["type"]))
    spin_ground = np.array(config_dict["ground_state"]).reshape(-1, 3)
    batch.set_spin_ground(spin_ground, config_dict["atom_index"])
    batch.set_template(config_dict["template_dir"])  # template directory
    batch.construct_dir(config_dict["working_dir"])
    batch.sample(tuple(config_dict["num_sampling"]), end_point=config_dict["end_point"])
    batch.construct_all()  # parent directory
    if not config_dict["construct_only"]:
        batch.run_and_wait(config_dict["num_batch"])
        batch.compute()
        if config_dict["type"] == "spherical_distribution":
            batch.draw(config_dict["plot_type"])
        elif config_dict["type"] == "effective_comparison":
            batch.draw(config_dict["plot_type"], config_dict["plot_azimuthal_index"])  # unique


def plot(config_dict):  # dskit_plot

    if config_dict["type"] == "spherical_distribution":
        batch = AllSphereSample()
        batch.save_dir = config_dict["data_dir"]
        batch.draw(config_dict["plot_type"], config_dict["is_degree"], config_dict["mark_ground"])
    elif config_dict["type"] == "effective_comparison":
        batch = ThetaDiffSample()
        batch.save_dir = config_dict["data_dir"]
        batch.draw(config_dict["plot_type"], config_dict["plot_azimuthal_index"])  # unique

def optimize(config_dict):  # dskit_opt

    func = SpinSerial(config_dict["name"])
    func.set_template(config_dict["template_dir"])
    func.construct(config_dict["working_dir"])
    raise NotImplementedError("DeltaSpin-based optimization hasn't been implemented yet.")
    # result = basinhopping(func, x0)

