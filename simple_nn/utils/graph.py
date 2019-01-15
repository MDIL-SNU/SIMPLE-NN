import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
import itertools
from tqdm import tqdm

def plot_gdfinv_density(gdfinv_list, atom_types, bins=500):

    for item in atom_types:
        plt.hist(gdfinv_list[item][:,0], bins) 

        plt.xlabel('$[\\rho(\mathrm{\mathsf{\mathbf{G}}}_{ij})]^{-1}$')
        plt.ylabel('Frequency')
        plt.savefig('GDFinv_hist_{}.pdf'.format(item))
        plt.clf()


def plot_Gdistance_vs_Ferror(G_list, F_list, atom_types, use_scale=True, bins=500):
    res = dict()
    max_num = 30000

    if use_scale:
        with open('scale_factor') as fil:
            scale = pickle.load(fil)

    for item in atom_types:
        data_len = len(G_list[item])
        res[item] = list()

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
                tmp_res = np.array([np.linalg.norm(G_list[item][i] - G_list[item][st_idx:tmp_ed], axis=1),
                                    np.linalg.norm(F_list[item][i] - F_list[item][st_idx:tmp_ed], axis=1)])
                tmp_res = tmp_res[:, tmp_res[0,:] < 1.]

                res[item].append(tmp_res)
                st_idx = tmp_ed
                tmp_ed += max_num

            res[item].append(np.array([np.linalg.norm(G_list[item][i] - G_list[item][st_idx:], axis=1),
                                       np.linalg.norm(F_list[item][i] - F_list[item][st_idx:], axis=1)]))

        #res[item] = np.array(res[item])
        res[item] = np.concatenate(res[item], axis=1)

        plt.hist2d(res[item][0,:], res[item][1,:], bins, range=[[0., 1.],[0., 4.]]) 

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
        target_pack_full = np.concatenate([target_res['DFT_F'], target_res['NN_F'], np.expand_dims(target_res['atom_idx'], 1), target_res['atomic_weights']], axis=1)
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
        plt.xlabel('$[\\rho(\mathrm{\mathsf{\mathbf{G}_{Si}}})]^{-1}$')
        plt.legend(frameon=False, prop={'size':16})

        if normalize:
            plt.savefig('normalized_ferror_vs_GDFinv_{}.pdf'.format(item))
        else:
            plt.savefig('ferror_vs_GDFinv_{}.pdf'.format(item))
        plt.clf()
