from __future__ import print_function
import six
from six.moves import cPickle as pickle
import numpy as np
from cffi import FFI
import os, sys, psutil
import types
import re
from collections import OrderedDict
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, control_flow_ops, tensor_array_ops

def _gen_2Darray_for_ffi(arr, ffi, cdata="double"):
    # Function to generate 2D pointer for cffi  
    shape = arr.shape
    arr_p = ffi.new(cdata + " *[%d]" % shape[0])
    for i in range(shape[0]):
        arr_p[i] = ffi.cast(cdata + " *", arr[i].ctypes.data)
    return arr_p


def pickle_load(filename):
    with open(filename, 'rb') as fil:
        if six.PY2:
            return pickle.load(fil)
        elif six.PY3:
            return pickle.load(fil, encoding='latin1')


def _make_data_list(filename):
    return sum(_make_str_data_list(filename), [])


def _make_str_data_list(filename):
    """
    read pickle_list file to make group of list of pickle files.
    group id is denoted in front of the line (optional).
    if group id is not denoted, group id of -1 is assigned.

    example:
    0:file1.pickle
    0:file2.pickle
    14:file3.pickle
    """
    h = re.compile("([0-9]+):(.*)")
    data_list = OrderedDict()
    with open(filename, 'r') as fil:
        for line in fil:
            m = h.match(line.strip())
            if m:
                group_id, file_name = m.group(1), m.group(2)
            else:
                group_id, file_name = -1, line.strip()
            if group_id not in data_list:
                data_list[group_id] = []
            data_list[group_id].append(file_name)
    return data_list.values()


def _make_full_featurelist(filelist, atom_types, feature_tag):
    data_list = _make_data_list(filelist)

    feature_list = dict()
    idx_list = dict()

    for item in atom_types:
        feature_list[item] = list()
        idx_list[item] = list()

    for i,item in enumerate(data_list):
        tmp_data = pickle_load(item)
        for jtem in atom_types:
            if jtem in tmp_data[feature_tag]:
                feature_list[jtem].append(tmp_data[feature_tag][jtem])
                idx_list[jtem].append([i]*tmp_data['N'][jtem])
                
    for item in atom_types:
        if len(feature_list[item]) > 0:
            feature_list[item] = np.concatenate(feature_list[item], axis=0)
            idx_list[item] = np.concatenate(idx_list[item], axis=0)

    return feature_list, idx_list


def _generate_scale_file(feature_list, atom_types):
    scale = dict()
    for item in atom_types:
        inp_size = feature_list[item].shape[1]
        scale[item] = np.zeros([2, inp_size])

        if len(feature_list[item]) > 0:
            scale[item][0,:] = 0.5*(np.amax(feature_list[item], axis=0) + np.amin(feature_list[item], axis=0))
            scale[item][1,:] = 0.5*(np.amax(feature_list[item], axis=0) - np.amin(feature_list[item], axis=0))

            #scale[item][0,:] = np.mean(feature_list[item], axis=0)
            #scale[item][1,:] = np.std(feature_list[item], axis=0)

            scale[item][1, scale[item][1,:] < 1e-15] = 1.
        else:
            scale[item][1,:] = 1.

    with open('scale_factor', 'wb') as fil:
        pickle.dump(scale, fil, pickle.HIGHEST_PROTOCOL)

    return scale


def _generate_gdf_file(feature_list, scale, atom_types, idx_list, sigma=0.02, modifier=None):
    ffi = FFI()
    ffi.cdef("""void calculate_gdf(double **, int, int, double, double *);""")
    lib = ffi.dlopen(os.path.join(os.path.dirname(os.path.realpath(__file__)) + "/libgdf.so"))

    gdf = dict()
    for item in atom_types:
        if len(feature_list[item]) > 0:
            scaled_feature = feature_list[item] - scale[item][0:1,:]
            scaled_feature /= scale[item][1:2,:]
            scaled_feature_p = _gen_2Darray_for_ffi(scaled_feature, ffi)

            temp_gdf = np.zeros([scaled_feature.shape[0]], dtype=np.float64, order='C')
            temp_gdf_p = ffi.cast("double *", temp_gdf.ctypes.data)

            lib.calculate_gdf(scaled_feature_p, scaled_feature.shape[0], scaled_feature.shape[1], sigma, temp_gdf_p)
            gdf[item] = np.squeeze(np.dstack(([temp_gdf, idx_list[item]])))
            if callable(modifier):
                gdf[item] = modifier(gdf[item])
            gdf[item][:,0] /= np.mean(gdf[item][:,0])

    with open('atomic_weights', 'wb') as fil:
        pickle.dump(gdf, fil, pickle.HIGHEST_PROTOCOL)

    return gdf


def compress_outcar(filename):
    """
    Compress VASP OUTCAR file for fast file-reading in ASE.
    Compressed file (tmp_comp_OUTCAR) is temporarily created in the current directory.

    :param str filename: filename of OUTCAR

    supported properties:

    - atom types
    - lattice vector(cell)
    - free energy
    - force
    """
    comp_name = './tmp_comp_OUTCAR'

    with open(filename, 'r') as fil, open(comp_name, 'w') as res:
        minus_tag = 0
        line_tag = 0
        for line in fil:
            if 'POTCAR:' in line:
                res.write(line)
            elif 'ions per type' in line:
                res.write(line)
            elif 'direct lattice vectors' in line:
                res.write(line)
                minus_tag = 3
            elif 'FREE ENERGIE OF THE ION-ELECTRON SYSTEM' in line:
                res.write(line)
                minus_tag = 4
            elif 'POSITION          ' in line:
                res.write(line)
                line_tag = 3
            elif minus_tag > 0:
                res.write(line)
                minus_tag -= 1
            elif line_tag > 0:
                res.write(line)
                if '-------------------' in line:
                    line_tag -= 1

    return comp_name


def modified_sigmoid(gdf, b=150.0, c=1.0):
    """
    modified sigmoid function for GDF calculation.

    :param gdf: numpy array, calculated gdf value
    :param b: float or double, coefficient for modified sigmoid
    :param c: float or double, coefficient for modified sigmoid
    """
    gdf[:,0] = gdf[:,0] / (1.0 + np.exp(-b * gdf[:,0] + c))
    return gdf


def memory():
    pid = os.getpid()
    py = psutil.Process(pid)
    memory_use = py.memory_info()[0]
    print('memory_use:', memory_use)


def repeat(x, counts):
    """
    repeat x repeated by counts (elementwise)
    counts must be integer tensor.

    example:
      x = [3.0, 4.0, 5.0, 6.0]
      counts = [3, 1, 0, 2]
      repeat(x, counts)
      >> [3.0, 3.0, 3.0, 4.0, 6.0, 6.0]
    """
    def cond(_, i):
        return i < size

    def body(output, i):
        value = array_ops.fill(counts[i:i+1], x[i])
        return (output.write(i, value), i + 1)

    size = array_ops.shape(counts)[0]
    init_output_array = tensor_array_ops.TensorArray(
        dtype=x.dtype, size=size, infer_shape=False)
    output_array, num_writes = control_flow_ops.while_loop(
        cond, body, loop_vars=[init_output_array, 0])

    return control_flow_ops.cond(
        num_writes > 0,
        output_array.concat,
        lambda: array_ops.zeros(shape=[0], dtype=x.dtype))
