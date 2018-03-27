# Code for generating feature vector from atomic coordinates.
# Currently, atomic symmetry function(reference) can available.
# TODO: make atomic symmetry function feature generator(use C)

import sys


def generate_feature(structure_list, feature_generator):
    feature_generator(structure_list)
    return 0


def main(argv=argv):
    generate_feature(argv[0], argv[1])

if __name__ == "__main__":
    main(sys.argv)