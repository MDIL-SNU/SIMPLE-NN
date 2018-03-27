from mpi4py import MPI
from six.moves import cPickle as pickle
from ase import io


def feature_generator(structure_list):
    for item in structure_list:
        feature_dict = calculate_feature(item)
        pickle.dump(feature_dict, _name_, pickle.HIGHST_PROTOCOL)  # TODO: directory setting?
    return 0


def calculate_feature(structure):
    comm = MPI.COMM_WORLD
    atoms = io.read(structure)  # TODO: add structure file information

    feature_dict = dict()
    feature_dict['sf'], feature_dict['dsf'] = _c_extension_for_calculating_sf_(atoms, comm)
    # TODO: make c extension

    return feature_dict
