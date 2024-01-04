import os.path
from fractions import Fraction

import numpy as np
from pymatgen.core.lattice import get_integer_index
from pymatgen.core.structure import Structure, Lattice
from pymatgen.transformations.standard_transformations import SupercellTransformation

from dskit.utils.math_helper import almost_zero, normalize, almost_collinear, cart2sphe, \
    general_spiral_parametric_equation, included_angle


def build_spiral(config_dict, spiral_dict):
    # Accurate for only cubic systems

    # structure = Structure.from_file("/tmp/pycharm_project_887/BFO.vasp")
    if "structure_file" in config_dict:
        structure_path = config_dict["structure_file"]
    else:
        structure_path = os.path.join(config_dict["template_dir"], "POSCAR")

    structure = Structure.from_file(structure_path)
    spin_geo = spiral_dict["spin_geo"]  # spiral or cycloid
    spiral = False
    cycloid_sum = False
    cycloid = False

    if spin_geo == "spiral":
        spiral = True
    elif spin_geo == "cycloid_sum":
        cycloid_sum = True
    elif spin_geo == "cycloid":
        cycloid = True
    else:
        raise ValueError("Unsupported spin geometry.")
    q_str = spiral_dict["q"]
    q_value = eval(q_str)
    q_direction = spiral_dict["q_direction"]

    if isinstance(spiral_dict["original_spin"], str):
        original_spin = np.loadtxt(spiral_dict["original_spin"], skiprows=1).reshape(-1, 3)
    elif isinstance(spiral_dict["original_spin"], list):
        original_spin = spiral_dict["original_spin"]  # Cartesian coordinates
    else:
        raise ValueError("Unsupported spin form.")
    # original_spin = np.vstack((np.loadtxt("MAGMOM.txt"), np.array(original_spin)))
    magnetic_moment = spiral_dict["m_moment"]

    force_spiral = False
    fixed_theta = None
    fixed_phi = None

    force_cycloid_plane = False
    cycloid_plane = None

    if spiral:
        # Forcing original angles is only available for spiral geometry.
        force_spiral = spiral_dict["force_spiral"]
        if force_spiral:
            fixed_theta = np.radians(spiral_dict["forced_original_theta"])
            fixed_phi = np.radians(spiral_dict["forced_original_phi"])

    elif cycloid_sum:  # cycloid_sum
        # must be True when original spin is parallel to q direction, and it's cycloid geometry
        force_cycloid_plane = spiral_dict["force_cycloid_plane"]
        try:
            cycloid_plane = spiral_dict["cycloid_plane"]
        except KeyError:
            cycloid_plane = None

    elif cycloid:
        try:
            cycloid_plane = spiral_dict["cycloid_plane"]  # Direct coordinates
        except KeyError:
            raise ValueError("Cycloidal (not cycloid of summation) structure needs chiral plane specified")
        print(f"Propagation vector q {q_direction}")
        print(f"Cycloid plane (chiral vector) c {cycloid_plane}")
        vector_c = structure.lattice.get_cartesian_coords(cycloid_plane)
        vector_q = structure.lattice.get_cartesian_coords(q_direction)
        if not np.allclose(included_angle(vector_c, vector_q), np.pi / 2, rtol=0, atol=1e-5):
            raise ValueError("The propagation vector (q) must be perpendicular to the chiral vector (cycloid_plane)")

    # q_max is the maximum absolute value of q relative to that of ferromagnetic fcc Fe
    (a, b, c) = q_direction
    magnetic_elements = list(magnetic_moment.keys())

    if not force_spiral:
        # m_moments = np.linalg.norm(original_spin, axis=1)
        for i, site in enumerate(structure):
            if site.species_string in magnetic_elements:
                site.properties["original_magmom_unit"] = normalize(np.array(original_spin[i]))
            else:
                site.properties["original_magmom_unit"] = [0, 0, 0]

    # properties must be available for all sites, out-of-index error otherwise
    primitive_structure = structure.get_primitive_structure(use_site_props=False)
    unique_id = 0
    for site in primitive_structure:
        if site.species_string in magnetic_elements:
            site.properties["primitive_id"] = unique_id
            unique_id += 1
        else:
            site.properties["primitive_id"] = -1

    # get orthgonal miller indices in CONVENTIONAL cells
    zero_index_list = []
    for i, q_component in enumerate([a, b, c]):
        if q_component == 0:
            zero_index_list.append(i)
    num_zero = len(zero_index_list)

    if num_zero == 2:
        redefine_matrix_convention = np.zeros((3, 3))
        for i, index in enumerate(zero_index_list):
            redefine_matrix_convention[i, index] = 1
        redefine_matrix_convention[2, :] = q_direction
    elif num_zero == 1:
        if zero_index_list[0] != 0:  # a != 0
            redefine_matrix_convention = [[b, -a, 0],
                                          [c, 0, -a],
                                          [a, b, c]]
        else:
            redefine_matrix_convention = [[0, c, -b],
                                          [c, 0, -a],
                                          [a, b, c]]
    else:
        redefine_matrix_convention = [[b, -a, 0],
                                      [c, 0, -a],
                                      [a, b, c]]

    # have to make sure det(SE3) > 0 while c axis remains (a, b, c) (unnecessary, E3 is good)
    # if np.linalg.det(redefine_convention) < 0:
    #     redefine_convention[0] = [-x for x in redefine_convention[0]]

    print(f'''In the generated orthogonal cell of spin spiral
{redefine_matrix_convention[0]} -> Axis a  
{redefine_matrix_convention[1]} -> Axis b
{redefine_matrix_convention[2]} -> Axis c''')

    # convert transformation matrix into primitive cell form and redefine lattice
    redefine_matrix_primitive_real = primitive_structure.lattice.get_fractional_coords(
        structure.lattice.get_cartesian_coords(redefine_matrix_convention))

    # get hkl indices with simpliest integers
    # get_integer_index will flip symbols, preserve it in advance.
    matrix_abs = np.empty_like(redefine_matrix_primitive_real)
    matrix_symbol = np.where(redefine_matrix_primitive_real > 0, 1, -1)
    for i, coords in enumerate(redefine_matrix_primitive_real):
        matrix_abs[i, :] = np.abs(get_integer_index(coords))
    redefine_matrix_int = matrix_abs * matrix_symbol

    trans = SupercellTransformation(redefine_matrix_int)
    structure_correct_orient = trans.apply_transformation(primitive_structure)

    # reorient c axis to z direction
    lattice = structure_correct_orient.lattice
    lattice_before = structure_correct_orient.lattice.matrix
    structure_correct_orient.lattice = Lattice.from_parameters(
        lattice.a,
        lattice.b,
        lattice.c,
        lattice.alpha,
        lattice.beta,
        lattice.gamma,
    )
    lattice_after = structure_correct_orient.lattice.matrix

    # E(3): rotate to c along z
    e3_c_along_z = lattice_after.T @ np.linalg.inv(lattice_before.T)
    c = structure_correct_orient.lattice.c
    q_unit_vector = np.array([0, 0, 1])
    q_vector = q_value * q_unit_vector * (2 * np.pi / c)

    # transform spin structure as well
    spiral_supercell_base = structure_correct_orient.get_sorted_structure(key=lambda atom: atom.z)
    primitive_dict = {}
    if not force_spiral:
        for i, site in enumerate(spiral_supercell_base):
            if site.species_string in magnetic_elements:
                unit_magmom_before = site.properties["original_magmom_unit"]
                site.properties["original_magmom_unit"] = e3_c_along_z @ unit_magmom_before
                if not str(site.properties["primitive_id"]) in primitive_dict:
                    primitive_dict[str(site.properties["primitive_id"])] = site

    # print(get_integer_index(structure.lattice.get_fractional_coords(
    #     np.linalg.inv(e3_c_along_z) @ np.fromstring(' -0.891428    0.029771   -0.452184', sep=" "))))

    # get E(3): rotate to cycloid spherical coordinates
    # M cross x(q) = z
    # z cross x(q) = y
    if cycloid_sum:
        # x
        x_cycloid = q_unit_vector

        # z
        unit_magmom_list = []
        for _, site in primitive_dict.items():
            unit_magmom_list.append(site.properties["original_magmom_unit"])
        net_mag = np.sum(unit_magmom_list, axis=0)
        net_mag_unit = normalize(net_mag)

        if almost_zero(net_mag, atol=1e-5):
            if cycloid_plane:
                # net_mag = e3_c_along_z @ structure.lattice.get_cartesian_coords(fixed_M_direction)
                z_cycloid = normalize(e3_c_along_z @ structure.lattice.get_cartesian_coords(cycloid_plane))
            else:
                raise ValueError(
                    "Gotta specify normal vector of the cycloidal plane if the net magnetization equals almost zero.")
        elif almost_collinear(net_mag, q_unit_vector, atol=1e-4):
            if cycloid_plane:
                z_cycloid = normalize(e3_c_along_z @ structure.lattice.get_cartesian_coords(cycloid_plane))
            else:
                raise ValueError(
                    "Gotta specify normal vector of the cycloidal plane if the net magnetization is almost parallel "
                    "to vector q.")
        elif force_cycloid_plane:
            z_cycloid = normalize(e3_c_along_z @ structure.lattice.get_cartesian_coords(cycloid_plane))
        else:
            z_cycloid = normalize(np.cross(net_mag_unit, q_unit_vector))

        # y
        y_cycloid = np.cross(z_cycloid, x_cycloid)

        e3_cycloid = np.linalg.inv(np.array([x_cycloid,
                                             y_cycloid,
                                             z_cycloid]).T)

        e3_cycloid_inv = np.array([x_cycloid,
                                   y_cycloid,
                                   z_cycloid]).T

    # get E(3): rotate to tilted (canting) plane of rotation
    # treat as cubic
    elif cycloid:
        chiral_unit = normalize(e3_c_along_z @ structure.lattice.get_cartesian_coords(cycloid_plane))
        azimuth_direction = np.cross(chiral_unit, q_unit_vector)
        # chiral_unit = normalize(e3_c_along_z @ cycloid_plane)
        # azimuth_direction = np.cross(chiral_unit, normalize(e3_c_along_z @ q_direction))

    # determine theta and phi in parametric equation
    qr_inner_product_single_cell = spiral_supercell_base.cart_coords @ q_vector
    for i, site in enumerate(spiral_supercell_base):
        if site.species_string in magnetic_elements:
            if force_spiral:
                site.properties["original_angles"] = [fixed_theta, fixed_phi]
            else:
                if site == primitive_dict[str(site.properties["primitive_id"])]:
                    if spiral:
                        _, theta_current, phi_current = cart2sphe(*site.properties["original_magmom_unit"])
                    elif cycloid_sum:
                        original_magmom_unit_cycloid = e3_cycloid @ site.properties["original_magmom_unit"]
                        # site.properties["original_magmom_unit_cycloid"] = original_magmom_unit_cycloid
                        _, theta_current, phi_current = cart2sphe(*original_magmom_unit_cycloid)
                    elif cycloid:
                        s = site.properties["original_magmom_unit"]
                        rotate_sign = q_unit_vector @ site.properties["original_magmom_unit"]
                        if np.allclose(rotate_sign, 0, rtol=0, atol=1e-4):
                            rotate_z = chiral_unit
                            print("Cycloid structure without cants will be generated.")
                        else:
                            # chiral c ==> c dot (plane of rotation) > 0
                            # x(azimuth reference) = c cross q ==> q = x cross c
                            # (plane of rotation) = s cross x <==> c dot (s cross x) > 0
                            # <==> s dot (x cross c) > 0 <==> s dot q > 0
                            if rotate_sign > 0:
                                rotate_z = np.cross(s, azimuth_direction)
                            else:
                                rotate_z = np.cross(azimuth_direction, s)
                        # print(site.properties["primitive_id"])
                        rotate_y = np.cross(rotate_z, azimuth_direction)
                        rotate_e3 = np.linalg.inv(np.array([azimuth_direction,
                                                            rotate_y,
                                                            rotate_z]).T)
                        rotate_e3_inv = np.array([azimuth_direction,
                                                  rotate_y,
                                                  rotate_z]).T

                        _, _, phi_current = cart2sphe(*(rotate_e3 @ s))
                        theta_current = np.pi / 2
                        site.properties["rotate_e3_inv"] = rotate_e3_inv

                    else:
                        raise NotImplementedError("Only spiral or cycloid geometry is available now.")
                    theta_bottom = theta_current
                    phi_bottom = phi_current - qr_inner_product_single_cell[i]
                    site.properties["original_angles"] = [theta_bottom, phi_bottom]
                else:
                    if cycloid:
                        site.properties["rotate_e3_inv"] = \
                            primitive_dict[str(site.properties["primitive_id"])].properties["rotate_e3_inv"]

                    site.properties["original_angles"] = \
                        primitive_dict[str(site.properties["primitive_id"])].properties["original_angles"]
        else:
            if cycloid:
                site.properties["rotate_e3_inv"] = np.zeros((3, 3))
            site.properties["original_angles"] = [0, 0]

    # print(spiral_supercell_base)

    # build up spin configurations
    if np.allclose(q_value, 0, rtol=0, atol=1e-8):
        spiral_period = 1
    else:
        spiral_period = Fraction(
            q_str).limit_denominator().denominator  # Fraction(0).limit_denominator().denominator == 0

    supercell = spiral_supercell_base * (1, 1, spiral_period)
    qr_inner_product_supercell = supercell.cart_coords @ q_vector

    parametric_equation = general_spiral_parametric_equation
    for i, site in enumerate(supercell):
        if site.species_string in magnetic_elements:
            theta, phi = site.properties["original_angles"]
            site.properties["magmom"] = [magnetic_moment[site.species_string] * component for component in
                                         parametric_equation(qr_inner_product_supercell[i], theta, phi)]
            if cycloid_sum:
                site.properties["magmom"] = e3_cycloid_inv @ site.properties["magmom"]

            elif cycloid:
                site.properties["magmom"] = site.properties["rotate_e3_inv"] @ site.properties["magmom"]

        else:
            site.properties["magmom"] = [0, 0, 0]

    # local ferromagnetic moment
    primitive_id_list = []
    sum_s_list = []
    sum_s = np.zeros(3)
    sublattice_group_count = 0
    for i, site in enumerate(supercell.get_sorted_structure(key=lambda atom: atom.z)):
        if site.species_string in magnetic_elements:
            sublattice_group_count += 1

            if not site.properties["primitive_id"] in primitive_id_list:
                primitive_id_list.append(site.properties["primitive_id"])

            if site.properties["primitive_id"] == primitive_id_list[0] and sublattice_group_count != 1:
                sum_s_list.append(sum_s)
                sum_s = np.zeros(3)
            sum_s += site.properties["magmom"]
    sum_s_list.append(sum_s)  # append the last local moment
    sum_s_array = np.array(sum_s_list)

    supercell_more = supercell * tuple(config_dict["build_supercell"])
    num_supercell = config_dict['build_supercell'][0] * config_dict['build_supercell'][1] * \
                    config_dict['build_supercell'][2]
    supercell_specie_sorted = supercell_more.get_sorted_structure(key=lambda atom: atom.species_string)
    os.chdir(config_dict["working_dir"])
    mcif_name = config_dict["output_name_mcif"]
    supercell_specie_sorted.to(fmt="mcif", filename=mcif_name)
    poscar_name = config_dict["output_name_vasp"]
    supercell_specie_sorted.to(fmt="poscar", filename=poscar_name)
    magmom_matrix = np.array([site.properties["magmom"] for site in supercell_specie_sorted])

    np.savetxt("MAGMOM_generated.txt", magmom_matrix,
               header=f"Generated by DeltaSpinKit. {spiral_period} unit cells in the final structure.")
    np.savetxt("local_moment_generated.txt", sum_s_array,
               header=f"Generated by DeltaSpinKit. Sorted by z components of local areas.")
    print(f"Finished. {spiral_period * num_supercell} unit cells in the final structure.")
    return magmom_matrix, spiral_period, supercell
