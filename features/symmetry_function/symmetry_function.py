from ase import io


def feature_generator(structure_list):
    for item in structure_list:
        calculate_feature(item)
    return 0


def calculate_feature(structure):
    io.read(structure)
    return 0
