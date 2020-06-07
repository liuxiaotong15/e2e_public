from log import log
import math
import numpy as np
from progress.bar import Bar
from ase import Atoms
from itertools import combinations
from ase.db import connect
from ase.visualize import view
from sklearn.datasets.samples_generator import make_blobs
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
# Plot result
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import random

seed = 1234
random.seed(seed)
np.random.seed(seed)
# tf.set_random_seed(seed)

print(__doc__)


# #############################################################################
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# Y, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5,
#                             random_state=0)
# print(Y, labels_true)

results = []
total_cluster_cnt = 0

def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]

def atoms_clustering(sym1, sym2, ax):
    global total_cluster_cnt
    log.logger.info('new round start ' + sym1 + '-' + sym2)
    cluster_finding_distance = 5.0 ## must same with main!!!
    X = [[i] for i in range(10000)]
    Y = [0 for i in range(10000)]  # 0.001A from 0A to 10A
    db = connect('./qm9.db')
    bar = Bar('finding '+sym1 + '-' + sym2 + ' bonds in db', max=db.count())
    for row in db.select():
        atoms = row.toatoms()
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                if(i != j and atoms[i].symbol == sym1 and atoms[j].symbol == sym2
                        and atoms.get_distance(i, j) < cluster_finding_distance):
                    Y[int(atoms.get_distance(i, j)*1000)] += 1
        bar.next()
        #if(row.id == 2000):
        #    break
    bar.finish()

    non_zero_X = [X[i] for i in range(10000) if Y[i] >= int(max(Y)/20)]
    non_zero_Y = [Y[i] for i in range(10000) if Y[i] >= int(max(Y)/20)]
    # https://www.codercto.com/a/23218.html
    fit_scores = []
    ret_found = False


    if((sym1 == 'F' and sym2 == 'O') or (sym1 == 'F' and sym2 == 'F')):
        pass
    else:
        for i in range(2, 20):
            init_x = []
            km = KMeans(n_clusters=i, init='random')
            km.fit(X=non_zero_X, sample_weight=non_zero_Y)

            #sc_x = []
            #sc_y = []
            lst_x = []
            # sc don't have sample_weight, so ... brute force
            for j in range(len(non_zero_X)):
                #sc_x += [non_zero_X[j]] * non_zero_Y[j]
                #sc_y += [km.labels_[j]] * non_zero_Y[j]
                lst_x.append(non_zero_X[j][0])
            # https://blog.wavelabs.ai/jenks-natural-breaks-optimization-finder/
            # but jenks don't have weight in it. so write myself...
            sdam = 0
            for idx, x in enumerate(lst_x):
                sdam += (x - np.array(lst_x).mean())**2 * non_zero_Y[idx]
            sdcm = 0
            for l in range(i):
                sub_sum = sum([lst_x[k]*non_zero_Y[k] for k in range(len(lst_x)) if km.labels_[k]==l])
                sub_cnt = sum([non_zero_Y[k] for k in range(len(lst_x)) if km.labels_[k]==l])
                sub_mean = sub_sum/sub_cnt
                sdcm += sum([(lst_x[k] - sub_mean)**2*non_zero_Y[k] for k in range(len(lst_x)) if km.labels_[k]==l])
            gvf = 1-sdcm/sdam
            fit_scores.append(gvf) # GVF > 99.5% we choose
            if(gvf > 0.995 and ret_found == False):
                ret_found = True
                results.append([sym1, sym2, i, gvf, km.cluster_centers_])
                total_cluster_cnt += i
            # # sc can't figure out short distance C-C
            # sc_score = silhouette_score(sc_x, sc_y, metric='euclidean', sample_size=20000)
            # #sc_score = silhouette_score(non_zero_X, km.labels_, metric='euclidean')
            # sc_scores.append(sc_score)
            log.logger.info(str(i))
            log.logger.info(str(km.cluster_centers_))
            log.logger.info(str(results))
    log.logger.info(str(fit_scores))

    # plt.close('all')
    # plt.figure(1)
    # plt.clf()
    '''
    for ax, case in zip(axs, cases):
        ax.set_title('markevery=%s' % str(case))
        ax.plot(x, y, 'o', ls='-', ms=4, markevery=case)
    '''
    ax.set_title(sym1 + '-' + sym2)
    ax.plot(list(np.array(non_zero_X)/1000.0), non_zero_Y)
    # print(results[0][4])
    if((sym1 == 'F' and sym2 == 'O') or (sym1 == 'F' and sym2 == 'F')):
        pass
    else:
        for i in range(len(results[-1][4])):
            ax.plot([results[-1][4][i][0]/1000, results[-1][4][i][0]/1000], [0, max(non_zero_Y)], linestyle=':')
    # plt.savefig(sym1 + '-' + sym2 + ' distance distribution')
    # plt.figure(2)
    # x = [i for i in range(2, 20)]
    # plt.plot(x, fit_scores)
    # plt.savefig(sym1 + '-' + sym2 + 'sc_scores')
    # plt.show()


if __name__ == '__main__':
    symbols = ['C', 'H', 'O', 'N', 'F']
    # symbols = ['C']
    figsize = (10, 8)
    cols = 5
    rows = 3

    fig1, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)

    axs = trim_axs(axs, cols*rows)
    ax_idx = 0
    for idx1, sym1 in enumerate(symbols):
        for idx2, sym2 in enumerate(symbols):
            if(idx1 < idx2):
                # F cnt so small F-F only 18 so pass it
                continue
            else:
                atoms_clustering(sym1, sym2, axs[ax_idx])
            ax_idx += 1

    axs[5].set_ylabel('pair count')
    axs[12].set_xlabel('pair distance, ' + r'$\AA$')
    log.logger.info('total_cluster_cnt: ' + str(total_cluster_cnt))
    plt.show()
