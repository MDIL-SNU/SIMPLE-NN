# Code for generating feature vector from atomic coordinates.
# Currently, atomic symmetry function(reference) can available.

import sys
import symmetry_function


def generate_feature(structure_list, feature_type):
    if feature_type == 'symmetry_function':
        symmetry_function.feature_generator(structure_list)
    else:
        raise NotImplementedError
    return 0


def main(argv=argv):
    generate_feature(argv[0], argv[1])

if __name__ == "__main__":
    main(sys.argv)