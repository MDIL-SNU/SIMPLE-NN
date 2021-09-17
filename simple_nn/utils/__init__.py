from __future__ import print_function
import six
from six.moves import cPickle as pickle
import numpy as np
from ._libgdf import lib, ffi
import os, sys, psutil, shutil
import types
import re
import collections
from collections import OrderedDict
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops, control_flow_ops, tensor_array_ops
from .mpiclass import DummyMPI, MPI4PY
from scipy.integrate import nquad

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


def _make_full_featurelist(filelist, feature_tag, atom_types=None, use_idx=False):
    """
    atom_types
    - None: atom type is not considered
    - list: use atom_types list
    """
    data_list = _make_data_list(filelist)

    if atom_types == None:
        feature_list = list()
        idx_list = list()

        for i,item in enumerate(data_list):
            tmp_data = pickle_load(item)
            feature_list.append(tmp_data[feature_tag])

        feature_list = np.concatenate(feature_list, axis=0)

    else:
        if use_idx:
            feature_list = dict()
            idx_list = dict()

            tmp_feature_list = list()
            tmp_atom_idx_list = list()
            for i,item in enumerate(data_list):
                tmp_data = pickle_load(item)
                tmp_feature_list.append(tmp_data[feature_tag])
                tmp_atom_idx_list.append(tmp_data['atom_idx'])

            tmp_feature_list = np.concatenate(tmp_feature_list, axis=0)
            tmp_atom_idx_list = np.concatenate(tmp_atom_idx_list, axis=0)

            for i,item in enumerate(atom_types):
                feature_list[item] = tmp_feature_list[tmp_atom_idx_list == i+1]

        else:
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


def _generate_scale_file(feature_list, atom_types, filename='scale_factor', scale_type='minmax', scale_scale=1.0, comm=DummyMPI(), scale_rho=None, params=None, log=None):
    scale = dict()
    for item in atom_types:
        inp_size = feature_list[item].shape[1]
        scale[item] = np.zeros([2, inp_size])
        is_scaled = np.array([True] * inp_size)

        if len(feature_list[item]) > 0:
            if scale_type == 'minmax':
                scale[item][0,:] = 0.5*(np.amax(feature_list[item], axis=0) + np.amin(feature_list[item], axis=0))
                scale[item][1,:] = 0.5*(np.amax(feature_list[item], axis=0) - np.amin(feature_list[item], axis=0)) / scale_scale
                is_scaled[scale[item][1,:] < 1e-15] = False
                scale[item][1, scale[item][1,:] < 1e-15] = 1.
            elif scale_type == 'meanstd':
                scale[item][0,:] = np.mean(feature_list[item], axis=0)
                scale[item][1,:] = np.std(feature_list[item], axis=0) / scale_scale
                is_scaled[scale[item][1,:] < 1e-15] = False
                scale[item][1, scale[item][1,:] < 1e-15] = 1.
            elif scale_type == 'uniform gas':
                assert params is not None and scale_rho is not None
                scale[item][0,:] = np.mean(feature_list[item], axis=0)

                # theta and phi is independent variable
                def G2(r):
                    return 4 * np.pi * r**2 * np.exp(-eta * (r - rs)**2) * 0.5 * (np.cos(np.pi * r / rc) + 1)

                # fix r_ij along z axis and use symmetry
                def G4(r1, r2, th2):
                    r3 = np.sqrt(r1**2 + r2**2 - 2*r1*r2*np.cos(th2))
                    fc3 = 0.5 * (np.cos(np.pi * r3 / rc) + 1) if r3 < rc else 0.0
                    return r1**2 * r2**2 * np.sin(th2) * 2 * np.pi *\
                            2**(1-zeta) * (1 + lamb * np.cos(th2))**zeta * np.exp(-eta * (r1**2 + r2**2 + r3**2)) *\
                            0.5 * (np.cos(np.pi * r1 / rc) + 1) * 0.5 * (np.cos(np.pi * r2 / rc) + 1) * fc3

                def G5(r1, r2, th2):
                    return r1**2 * r2**2 * np.sin(th2) * 2 * np.pi *\
                             2**(1-zeta) * (1 + lamb * np.cos(th2))**zeta * np.exp(-eta * (r1**2 + r2**2)) *\
                             0.5 * (np.cos(np.pi * r1 / rc) + 1) * 0.5 * (np.cos(np.pi * r2 / rc) + 1)

                # subtract G4 when j==k (r1==r2)
                def singular(r1):
                    # r3 = 0, fc3 = 1
                    return r1**4 * 2**(1-zeta) * (1 + lamb)**zeta * np.exp(-eta * 2 * r1**2) *\
                            (0.5 * (np.cos(np.pi * r1 / rc) + 1))**2

                for p in range(inp_size):
                    if params[item]['i'][p,0] == 2:
                        ti = atom_types[params[item]['i'][p,1] - 1]
                        eta = params[item]['d'][p,1]
                        rc = params[item]['d'][p,0]
                        rs = params[item]['d'][p,2]
                        scale[item][1,p] = scale_rho[ti] * nquad(G2, [[0, rc]])[0]
                    elif params[item]['i'][p,0] == 4:
                        ti = atom_types[params[item]['i'][p,1] - 1]
                        tj = atom_types[params[item]['i'][p,2] - 1]
                        eta = params[item]['d'][p,1]
                        rc = params[item]['d'][p,0]
                        zeta = params[item]['d'][p,2]
                        lamb = params[item]['d'][p,3]
                        scale[item][1,p] = scale_rho[ti] * scale_rho[tj] * 4 * np.pi *\
                                            (nquad(G4, [[0, rc], [0, rc], [0, np.pi]])[0] -\
                                            (nquad(singular, [[0, rc]])[0] if lamb == 1 else 0))
                    elif params[item]['i'][p,0] == 5:
                        ti = atom_types[params[item]['i'][p,1] - 1]
                        tj = atom_types[params[item]['i'][p,2] - 1]
                        eta = params[item]['d'][p,1]
                        rc = params[item]['d'][p,0]
                        zeta = params[item]['d'][p,2]
                        lamb = params[item]['d'][p,3]
                        scale[item][1,p] = scale_rho[ti] * scale_rho[tj] * 4 * np.pi *\
                                            (nquad(G5, [[0, rc], [0, rc], [0, np.pi]])[0] -\
                                            (nquad(singular, [[0, rc]])[0] if lamb == 1 else 0))
                    else:
                        assert False

            if log is not None and comm.rank == 0:
                log.write("{:-^70}\n".format(" Scaling information for {:} ".format(item)))
                log.write("(scaled_value = (value - mean) * scale)\n")
                log.write("Index   Mean         Scale        Min(after)   Max(after)   Std(after)\n")
                scaled = (feature_list[item] - scale[item][0,:]) / scale[item][1,:]
                scaled_min = np.min(scaled, axis=0)
                scaled_max = np.max(scaled, axis=0)
                scaled_std = np.std(scaled, axis=0)
                for i in range(scale[item].shape[1]):
                    scale_str = "{:11.4e}".format(1/scale[item][1,i]) if is_scaled[i] else "Not_scaled"
                    log.write("{0:<5}  {1:>11.4e}  {2:>11}  {3:>11.4e}  {4:>11.4e}  {5:>11.4e}\n".format(
                        i, scale[item][0,i], scale_str, scaled_min[i], scaled_max[i], scaled_std[i]))
        else:
            scale[item][1,:] = 1.

    if log is not None and comm.rank == 0:
        log.write("{:-^70}\n".format(""))

    if comm.rank == 0:
        with open(filename, 'wb') as fil:
            pickle.dump(scale, fil, protocol=2)

    return scale


def _generate_gdf_file(ref_list, scale, atom_types, idx_list, target_list=None, filename=None, noscale=False, sigma=0.02, comm=DummyMPI()):
    gdf = dict()
    auto_c = dict()
    auto_sigma = dict()

    for item in atom_types:
        if len(ref_list[item]) > 0:
            scaled_ref = ref_list[item] - scale[item][0:1,:]
            scaled_ref /= scale[item][1:2,:]
            scaled_ref_p = _gen_2Darray_for_ffi(scaled_ref, ffi)

            if target_list == None:
                scaled_target = scaled_ref
                scaled_target_p = scaled_ref_p
            else:
                scaled_target = target_list[item] - scale[item][0:1,:]
                scaled_target /= scale[item][1:2,:]
                scaled_target_p = _gen_2Darray_for_ffi(scaled_target, ffi)

            local_temp_gdf = np.zeros([scaled_target.shape[0]], dtype=np.float64, order='C')
            local_temp_gdf_p = ffi.cast("double *", local_temp_gdf.ctypes.data)

            if sigma == 'Auto':
                #if target_list != None:
                #    raise NotImplementedError
                #else:
                lib.calculate_gdf(scaled_ref_p, scaled_ref.shape[0], scaled_target_p, scaled_target.shape[0], scaled_ref.shape[1], -1., local_temp_gdf_p)
                local_auto_sigma = max(np.sort(local_temp_gdf))/3.
                comm.barrier()
                auto_sigma[item] = comm.allreduce_max(local_auto_sigma)

            elif isinstance(sigma, collections.Mapping):
                auto_sigma[item] = sigma[item]
            else:
                auto_sigma[item] = sigma

            lib.calculate_gdf(scaled_ref_p, scaled_ref.shape[0], scaled_target_p, scaled_target.shape[0], scaled_ref.shape[1], auto_sigma[item], local_temp_gdf_p)
            comm.barrier()

            temp_gdf = comm.gather(local_temp_gdf.reshape([-1,1]), root=0)
            comm_idx_list = comm.gather(idx_list[item].reshape([-1,1]), root=0)   

            if comm.rank == 0:
                temp_gdf = np.concatenate(temp_gdf, axis=0).reshape([-1])
                comm_idx_list = np.concatenate(comm_idx_list, axis=0).reshape([-1])

                gdf[item] = np.squeeze(np.dstack(([temp_gdf, comm_idx_list])))
                #print(gdf[item])
                gdf[item][:,0] *= float(len(gdf[item][:,0]))

                sorted_gdf = np.sort(gdf[item][:,0])
                max_line_idx = int(sorted_gdf.shape[0]*0.75)
                pfit = np.polyfit(np.arange(max_line_idx), sorted_gdf[:max_line_idx], 1)
                #auto_c[item] = np.poly1d(pfit)(sorted_gdf.shape[0]-1)
                auto_c[item] = np.poly1d(pfit)(max_line_idx-1)
                # FIXME: After testing, this part needs to be moved to neural_network.py

            """
            if callable(modifier[item]):
                gdf[item] = modifier[item](gdf[item])

            if not noscale:
                gdf[item][:,0] /= np.mean(gdf[item][:,0])
            """

    if (filename != None) and (comm.rank == 0):
        with open(filename, 'wb') as fil:
            pickle.dump(gdf, fil, protocol=2)

    return gdf, auto_sigma, auto_c



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
    - stress
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
            elif 'FORCE on cell =-STRESS' in line:
                res.write(line)
                minus_tag = 15
            elif 'Iteration' in line:
                res.write(line)
            elif minus_tag > 0:
                res.write(line)
                minus_tag -= 1
            elif line_tag > 0:
                res.write(line)
                if '-------------------' in line:
                    line_tag -= 1

    return comp_name


def modified_sigmoid(gdf, b=150.0, c=1.0, module_type=None):
    """
    modified sigmoid function for GDF calculation.

    :param gdf: numpy array, calculated gdf value
    :param b: float or double, coefficient for modified sigmoid
    :param c: float or double, coefficient for modified sigmoid
    """
    if module_type is None:
        module_type = np

    gdf = gdf / (1.0 + module_type.exp(-b * (gdf - c)))
    #gdf[:,0] = gdf[:,0] / (1.0 + np.exp(-b * gdf[:,0] + c))
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

def read_lammps_potential(filename):
    def _read_until(fil, stop_tag):
        while True:
            line = fil.readline()
            if stop_tag in line:
                break

        return line

    shutil.copy2(filename, 'potential_read')

    weights = dict()
    with open(filename) as fil:
        atom_types = fil.readline().replace('\n','').split()[1:]
        for item in atom_types:
            weights[item] = list()            

            dims = list()
            dims.append(int(_read_until(fil, 'SYM').split()[1]))

            hidden_to_out = map(lambda x: int(x), _read_until(fil, 'NET').split()[2:])
            dims += hidden_to_out

            num_weights = len(dims) - 1

            for j in range(num_weights):
                tmp_weights = np.zeros([dims[j], dims[j+1]])
                tmp_bias = np.zeros([dims[j+1]])

                # Since PCA will be dealt separately, skip PCA layer.
                skip = True if fil.readline().split()[-1] == 'PCA' else False
                for k in range(dims[j+1]):
                    tmp_weights[:,k] = list(map(lambda x: float(x), fil.readline().split()[1:]))
                    tmp_bias[k] = float(fil.readline().split()[1])

                if skip:
                    continue
                weights[item].append(np.copy(tmp_weights))
                weights[item].append(np.copy(tmp_bias))

    return weights
