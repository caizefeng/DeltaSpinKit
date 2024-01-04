import numpy as np

from dskit.utils.incar_helper import extend_array, reduce_array, trim_incar, merge_backslash


def tile_supercell(tag, trim_result, n_supercell, omit_num=3):
    array_before = trim_result[0]
    comment = trim_result[1]
    array_before = np.array(extend_array(array_before), dtype=str)
    array_after = np.tile(array_before.reshape(-1, 3), n_supercell).ravel()
    array_after = reduce_array(array_after.tolist(), omit_num=omit_num)
    new_config_string = "  ".join(array_after)
    new_line = '{} = {}              {}\n'.format(tag, new_config_string, comment)
    return new_line


def create_incar_supercell(config_dict):
    incar_in = config_dict["INCAR_orginal"]
    incar_out = config_dict["INCAR_supercell"]
    num_supercell = np.array(config_dict["num_supercell"], dtype=int).cumprod()[-1]

    no_backslash_lines = merge_backslash(incar_in)

    new_lines = []
    is_constrained = False
    for line in no_backslash_lines:
        if trim_result := trim_incar(line, "SCTYPE"):  # must have SCTYPE ahead of LAMBDA in INCAR
            sctype = trim_result[0][0]
            if sctype == "1":
                is_constrained = True
            else:
                is_constrained = False

        if trim_result := trim_incar(line, "MAGMOM"):
            new_lines.append(tile_supercell("MAGMOM", trim_result, num_supercell))
        elif trim_result := trim_incar(line, "M_CONSTR"):
            new_lines.append(tile_supercell("M_CONSTR", trim_result, num_supercell))
        elif trim_result := trim_incar(line, "CONSTRL"):
            new_lines.append(tile_supercell("CONSTRL", trim_result, num_supercell))
        elif (trim_result := trim_incar(line, "LAMBDA")) and is_constrained:
            new_lines.append(tile_supercell("LAMBDA", trim_result, num_supercell))

        else:
            new_lines.append(line)

    f_out = open(incar_out, "w")

    for line in new_lines:
        f_out.write(line)

    f_out.close()
