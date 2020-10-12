import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import itertools
from tqdm import tqdm
from ..utils import pickle_load
import logging

def plot_gdfinv_density(gdfinv_list, atom_types, bins=100, auto_c=None):

    for item in atom_types:
        if auto_c is not None: 
            fig = plt.figure(figsize=(10,4))

            ax1 = plt.subplot(121)
        else:
            fig = plt.figure()

        plt.hist(gdfinv_list[item][:,0], bins) 

        plt.xlabel('$[\\rho(\mathrm{\mathsf{\mathbf{G}}}_{ij})]^{-1}$')
        plt.ylabel('Frequency')

        if auto_c is not None:
            ax2 = plt.subplot(122)

            plt.hist(gdfinv_list[item][:,0], bins=np.linspace(0., 5*auto_c[item], bins)) 

            plt.xlabel('$[\\rho(\mathrm{\mathsf{\mathbf{G}}}_{ij})]^{-1}$')
            plt.ylabel('Frequency')
            plt.savefig('GDFinv_hist_{}.pdf'.format(item))

            ax2.axvline(auto_c[item], color='k', linestyle=':')

            plt.tight_layout()

        plt.savefig('GDFinv_hist_{}.pdf'.format(item))

        fig.clf()
        fig.clear()
        
def plot_Gdistance_vs_Ferror(G_list, F_list, atom_types, use_scale=True, bins=200, max_num=30000, p_range=[[0., 1.], [0., 10.]], 
                             to_check=[[0., 0.005], [0., 1.0]], **kargs):
    x_bins = np.linspace(p_range[0][0], p_range[0][1], bins+1)
    y_bins = np.linspace(p_range[1][0], p_range[1][1], bins+1)

    grid_x, grid_y = np.meshgrid(x_bins[:-1], y_bins[:-1])
    grid_x = grid_x.reshape([-1,1])
    grid_y = grid_y.reshape([-1,1])    
    """
    grid_pack order
    x_1,y_1 -> x_2,y_1 -> ... -> x_1,y_2 -> ...
    """

    # logging practice
    logging.basicConfig(filename='pairs_to_check.log', level=logging.INFO)

    def _make_checklist(res, to_check, cur_idx, idx_range):
        idx_G_check_lower = res[0,:] > to_check[0][0]
        idx_G_check_upper = res[0,:] < to_check[0][1]
        idx_F_check_lower = res[1,:] > to_check[1][0]
        idx_F_check_upper = res[1,:] < to_check[1][1]

        full_idx = np.logical_and(idx_G_check_lower, idx_G_check_upper)
        full_idx = np.logical_and(full_idx, idx_F_check_lower)
        full_idx = np.logical_and(full_idx, idx_F_check_upper)

        if np.sum(full_idx):
            grep_res = res[:,full_idx]
            grep_real_idx = np.arange(idx_range[0], idx_range[1])[np.squeeze(full_idx)]
            for i in range(np.sum(full_idx)):
                logging.info("{:16.8e} {:16.8} {:8d} {:8d}".format(grep_res[0,i], grep_res[1,i], cur_idx, grep_real_idx[i]))


    def _get_pack_count(res, x_bins=x_bins, y_bins=y_bins):
        x_digi = np.digitize(res[0,:], x_bins)
        x_num = len(x_bins)-1
        y_digi = np.digitize(res[1,:], y_bins)

        x_noout = x_digi < x_num+1
        y_noout = y_digi < x_num+1
        tot_noout = np.logical_and(x_noout, y_noout)

        pack_digi = (x_digi[tot_noout] - 1) + x_num*(y_digi[tot_noout] - 1)

        unique, counts = np.unique(pack_digi, return_counts=True)

        return unique, counts

    res = dict()
    #res = np.zeros(len(grid_x))

    if use_scale:
        #with open('scale_factor') as fil:
        #    scale = pickle.load(fil)
        scale = pickle_load('scale_factor')

    for item in atom_types:
        logging.info('**** {} ****'.format(item))
        data_len = len(G_list[item])
        #res[item] = list()
        res[item] = np.zeros(len(grid_x))

        if use_scale:
            G_list[item] -= scale[item][0:1,:]
            G_list[item] /= scale[item][1:2,:]

#        combi_list = np.array(list(itertools.combinations(range(data_len), 2)))
#        res[item] = np.concatenate(
#                        [np.linalg.norm(G_list[item][combi_list[:,0]] - G_list[item][combi_list[:,1]], axis=1, keepdims=True),
#                        np.linalg.norm(F_list[item][combi_list[:,0]] - F_list[item][combi_list[:,1]], axis=1, keepdims=True)],
#                        axis=1)

        for i in tqdm(range(data_len-1)):
#            for j in range(i+1,data_len):
#                res[item].append([np.linalg.norm(G_list[item][i] - G_list[item][j]), 
#                                  np.linalg.norm(F_list[item][i] - F_list[item][j])])
            st_idx = i+1

            tmp_ed = st_idx + max_num
            while tmp_ed < data_len:
                tmp_res = np.array([np.linalg.norm(G_list[item][i] - G_list[item][st_idx:tmp_ed], axis=1, keepdims=True),
                                    #np.linalg.norm(F_list[item][i] - F_list[item][st_idx:tmp_ed], axis=1, keepdims=True)])
                                    np.abs(np.linalg.norm(F_list[item][i]) - np.linalg.norm(F_list[item][st_idx:tmp_ed], axis=1, keepdims=True))])

                full_idx = _make_checklist(tmp_res, to_check, i, [st_idx, tmp_ed])

                # FIXME: Need change?
                #tmp_res = tmp_res[:, tmp_res[0,:] < 1.]
                tmp_res = tmp_res[:, tmp_res[0,:] < p_range[0][1]]

                #res[item].append(tmp_res)
                unique, counts = _get_pack_count(tmp_res)
                res[item][unique] += counts

                st_idx = tmp_ed
                tmp_ed += max_num

            tmp_res = np.array([np.linalg.norm(G_list[item][i] - G_list[item][st_idx:], axis=1, keepdims=True),
                                #np.linalg.norm(F_list[item][i] - F_list[item][st_idx:], axis=1, keepdims=True)])
                                np.abs(np.linalg.norm(F_list[item][i]) - np.linalg.norm(F_list[item][st_idx:], axis=1, keepdims=True))])

            full_idx = _make_checklist(tmp_res, to_check, i, [st_idx, data_len])

            unique, counts = _get_pack_count(tmp_res)
            res[item][unique] += counts

            #res[item].append(np.array([np.linalg.norm(G_list[item][i] - G_list[item][st_idx:], axis=1),
            #                           np.linalg.norm(F_list[item][i] - F_list[item][st_idx:], axis=1)]))

        #res[item] = np.array(res[item])
        #res[item] = np.concatenate(res[item], axis=1)
        plt.hist2d(np.squeeze(grid_x), np.squeeze(grid_y), bins=[x_bins, y_bins], weights=res[item], **kargs)#, cmax=1000) 

        plt.xlabel('$|\mathrm{\mathsf{\mathbf{G}}}_i-\mathrm{\mathsf{\mathbf{G}}}_j|$')
        plt.ylabel('$|\mathrm{\mathsf{\mathbf{F}}}_i-\mathrm{\mathsf{\mathbf{F}}}_j|$')
        plt.colorbar()
        plt.savefig('Gdistance_vs_Ferror_{}.pdf'.format(item))
        plt.clf()
         

def plot_error_vs_gdfinv(atom_types, ref_data, target_data=None, save_data=False, normalize=False):

    ignore_below = 1e-2

    with open(ref_data) as fil:
        ref_res = pickle.load(fil)
    if target_data != None:
        with open(target_data) as fil:
            target_res = pickle.load(fil)

    ref_pack_full = np.concatenate([ref_res['DFT_F'], ref_res['NN_F'], 
                                    np.expand_dims(ref_res['atom_idx'], 1), 
                                    np.expand_dims(ref_res['atomic_weights'], 1)], axis=1)
    ref_pack = dict()
    if target_data != None:
        target_pack_full = np.concatenate([target_res['DFT_F'], target_res['NN_F'], 
                                           np.expand_dims(target_res['atom_idx'], 1),
                                           np.expand_dims(target_res['atomic_weights'], 1)], axis=1)
        target_pack = dict()

    for i,item in enumerate(atom_types):
        ref_pack[item] = ref_pack_full[ref_pack_full[:,-2] == i+1]
        ref_pack[item] = ref_pack[item][np.lexsort((ref_pack[item][:,0], ref_pack[item][:,1], ref_pack[item][:,2]))]
        if target_data != None:
            target_pack[item] = target_pack_full[target_pack_full[:,-2] == i+1]
            target_pack[item] = target_pack[item][np.lexsort((target_pack[item][:,0], target_pack[item][:,1], target_pack[item][:,2]))]

    n_grid = 100
    #grid_gdf = np.linspace(np.min(ref_pack[:,-1]), np.max(ref_pack[:,-1]), n_grid)
    for item in atom_types:
        grid_gdf = np.logspace(np.log10(np.min(ref_pack[item][:,-1])), np.log10(np.max(ref_pack[item][:,-1])), n_grid)

        res = list()

        #print np.mean(np.linalg.norm(ref_pack[item][:,:3] - ref_pack[item][:,3:6], axis=1))
        #print np.mean(np.linalg.norm(target_pack[item][:,:3] - target_pack[item][:,3:6], axis=1))
        #print '------'

        for i in range(n_grid-1):

            tmp_idx = np.logical_and(ref_pack[item][:,-1]>grid_gdf[i], ref_pack[item][:,-1]<grid_gdf[i+1])
            #print np.sum(tmp_idx)

            tmp_ref_res = np.linalg.norm(ref_pack[item][tmp_idx][:,:3] - ref_pack[item][tmp_idx][:,3:6], axis=1)
            if normalize:
                force_mag = np.linalg.norm(ref_pack[item][tmp_idx][:,:3], axis=1)
                remove_zero = force_mag > ignore_below
                tmp_ref_res = tmp_ref_res[remove_zero] / force_mag[remove_zero]
                tmp_idx = 0

            if len(tmp_ref_res) == 0:
                ref_mean = 0.
                ref_1stq = 0.
                ref_3rdq = 0.
            else:
                ref_mean = np.mean(tmp_ref_res)
                ref_1stq = np.percentile(tmp_ref_res, 25)
                ref_3rdq = np.percentile(tmp_ref_res, 75)

            if target_data != None:
                tmp_target_res = np.linalg.norm(target_pack[item][tmp_idx][:,:3] - target_pack[item][tmp_idx][:,3:6], axis=1)
                if normalize:
                    force_mag = np.linalg.norm(target_pack[item][tmp_idx][:,:3], axis=1)
                    remove_zero = force_mag > ignore_below
                    tmp_target_res = tmp_target_res[remove_zero] / force_mag[remove_zero]

                if len(tmp_target_res) == 0:
                    target_mean = 0.
                    target_1stq = 0.
                    target_3rdq = 0.

                    cross_check = 0.
                else:
                    target_mean = np.mean(tmp_target_res)
                    target_1stq = np.percentile(tmp_target_res, 25)
                    target_3rdq = np.percentile(tmp_target_res, 75)
                    
                    cross_check = np.mean(ref_pack[item][tmp_idx][:,:3] - target_pack[item][tmp_idx][:,:3])

            #print np.mean(tmp_ref_res), np.percentile(tmp_ref_res, 25), np.percentile(tmp_ref_res, 75) 
            #print np.mean(tmp_target_res), np.percentile(tmp_target_res, 25), np.percentile(tmp_target_res, 75)
            #print cross_check

                res.append([grid_gdf[i], ref_mean, ref_1stq, ref_3rdq, target_mean, target_1stq, target_3rdq, np.sum(tmp_idx), cross_check])
            #print '-----------------'
            else:
                res.append([grid_gdf[i], ref_mean, ref_1stq, ref_3rdq])

        res = np.array(res)
        if save_data:
            np.savetxt('resout_{}.txt'.format(item), res)
    
        ax1 = plt.subplot(111)
        ax1.set_xscale('log', nonposx='clip')

        #print(gdfc.shape, gdfnnp.shape)
        plt.plot(res[:,0], res[:,1], 'b-', label='NNP-c')
        plt.fill_between(res[:,0], res[:,2], res[:,3], alpha=0.2, color='b')
        plt.plot(res[:,0], res[:,2], 'b:')
        plt.plot(res[:,0], res[:,3], 'b:')
        #plt.plot(gdfnnp[:,0], gdfnnp[:,1], 'r.', alpha=0.3)

        if target_data != None:
            plt.plot(res[:,0], res[:,4], 'r-', label='NNP-GDF')
            plt.fill_between(res[:,0], res[:,5], res[:,6], alpha=0.2, color='r')
            plt.plot(res[:,0], res[:,5], 'r:')
            plt.plot(res[:,0], res[:,6], 'r:')


        if normalize:
            plt.ylabel('$\\frac{|\mathrm{\mathsf{\mathbf{F}_{NNP}}} - \mathrm{\mathsf{\mathbf{F}_{DFT}}}|}{|\mathrm{\mathsf{\mathbf{F}_{DFT}}}|}$ (eV/$\mathrm{\mathsf{\AA}}$)')
        else:
            plt.ylabel('|$\mathrm{\mathsf{\mathbf{F}_{NNP}}}$ - $\mathrm{\mathsf{\mathbf{F}_{DFT}}}$| (eV/$\mathrm{\mathsf{\AA}}$)')
        plt.xlabel('$[\\rho(\mathrm{\mathsf{\mathbf{G}_{'+item+'}}})]^{-1}$')
        plt.legend(frameon=False, prop={'size':16})

        if normalize:
            plt.savefig('normalized_ferror_vs_GDFinv_{}.pdf'.format(item))
        else:
            plt.savefig('ferror_vs_GDFinv_{}.pdf'.format(item))
        plt.clf()

def plot_correlation_graph(test_result='test_result', atom_types=None):

    def _correlation(x, y, subplot_loc, xlabel=None, ylabel=None, title=None):
        ax = plt.subplot(subplot_loc)
        # F correlation
        plt.gca().set_aspect('equal', adjustable='box')
        min_val = np.min(x)
        max_val = np.max(x)
        padding = (max_val - min_val)*0.1
        min_val -= padding
        max_val += padding
 
        plt.plot([min_val, max_val], [min_val, max_val], 'k:')
        plt.plot(x, y, 'bo', alpha=0.5)
 
        ax.set_yticks(ax.get_xticks()[1:-1])
        ax.set_xticks(ax.get_xticks()[1:-1])
        plt.xlabel(xlabel) if xlabel != None else xlabel
        plt.ylabel(ylabel) if ylabel != None else ylabel
        plt.title(title) if title != None else title

    def _error_hist(x, subplot_loc, xlabel=None, ylabel=None, title=None):
        ax = plt.subplot(subplot_loc)
        # F_x error histogram
        plt.hist(x, 30)
        plt.ylabel(ylabel) if ylabel != None else ylabel
        plt.xlabel(xlabel) if xlabel != None else xlabel
        plt.title(title) if title != None else title

    res = pickle_load(test_result)

    force_tag = ('NN_F' in res) and ('DFT_F' in res)

    if force_tag:
        if atom_types != None:
            # TODO: error for non-list atom_types
            num_atom_types = len(atom_types)
            fig = plt.figure(figsize=(5*num_atom_types, 8))
            for i,item in enumerate(atom_types):
                target_NN_F = res['NN_F'][res['atom_idx']==i+1]
                target_DFT_F = res['DFT_F'][res['atom_idx']==i+1]

                _correlation(target_DFT_F[:,0], target_NN_F[:,0], 200+10*num_atom_types+i+1,
                             xlabel='$F_x^\mathrm{\mathsf{DFT}}$ (eV/$\mathrm{\mathsf{\AA}}$)',
                             ylabel='$F_x^\mathrm{\mathsf{NNP}}$ (eV/$\mathrm{\mathsf{\AA}}$)',
                             title=item)

                _error_hist(np.sqrt(np.sum((target_NN_F - target_DFT_F)**2, axis=1)), 200+11*num_atom_types+i+1,
                            xlabel='|$\mathrm{\mathsf{\mathbf{F}^{NNP}}} - \mathrm{\mathsf{\mathbf{F}^{DFT}}}$| (eV/$\mathrm{\mathsf{\AA}}$)',
                            ylabel='Frequency')

            plt.tight_layout()
            plt.savefig('F_correlation_per_atom_of_{}.pdf'.format(test_result))
            fig.clf()

        fig = plt.figure(figsize=(10,8))

        _correlation(res['DFT_F'][:,0], res['NN_F'][:,0], 222,
                     xlabel='$F_x^\mathrm{\mathsf{DFT}}$ (eV/$\mathrm{\mathsf{\AA}}$)',
                     ylabel='$F_x^\mathrm{\mathsf{NNP}}$ (eV/$\mathrm{\mathsf{\AA}}$)')

        _error_hist(np.sqrt(np.sum((res['NN_F'] - res['DFT_F'])**2, axis=1)), 224,
                    xlabel='|$\mathrm{\mathsf{\mathbf{F}^{NNP}}} - \mathrm{\mathsf{\mathbf{F}^{DFT}}}$| (eV/$\mathrm{\mathsf{\AA}}$)',
                    ylabel='Frequency')

    else:
        fig = plt.figure(figsize=(10,4))

    _correlation(res['DFT_E']/res['N'], res['NN_E']/res['N'], 221 if force_tag else 121,
                 xlabel='$E^\mathrm{\mathsf{DFT}}$ (eV/atom)',
                 ylabel='$E^\mathrm{\mathsf{NNP}}$ (eV/atom)')

    _error_hist(np.abs(res['NN_E']/res['N'] - res['DFT_E']/res['N'])*1000., 223 if force_tag else 122,
                xlabel='|$E^\mathrm{\mathsf{NNP}} - E^\mathrm{\mathsf{DFT}}$| (meV/atom)',
                ylabel='Frequency')

    plt.tight_layout()
    plt.savefig('correlation_of_{}.pdf'.format(test_result))

