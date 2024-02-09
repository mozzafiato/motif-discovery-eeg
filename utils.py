import pickle
import stumpy

import numpy as np
import pandas as pd

from itertools import cycle, combinations

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.special import comb


class Utils:
    def __init__(self):
        # default plot settings
        self.fig_size = plt.rcParams["figure.figsize"]
        self.fig_size[0] = 20
        self.fig_size[1] = 6
        plt.rcParams["figure.figsize"] = self.fig_size
        plt.rcParams['xtick.direction'] = 'out'

        # global variables
        self.data = None

    def read_data(self, path=None, band=None):
        if path is not None:
            with open(path, 'rb') as handle:
                self.data = pickle.load(handle)
        if band is None:
            with open('patient_eeg_176.pickle', 'rb') as handle:
                self.data = pickle.load(handle)
        else:
            with open('patient_eeg_176_{}.pickle'.format(band), 'rb') as handle:
                self.data = pickle.load(handle)

    def get_patient_signals(self, patient_index, length=None):
        sequences = []

        for eeg in enumerate(self.data[patient_index]):
            if length is None:
                sequences.append(list(eeg)[1].astype(np.float64))
            else:
                sequences.append(list(eeg)[1][0:length].astype(np.float64))
        return sequences

    def get_all_patient_signals(self, dataset="train"):
        eegs = []
        for i, key in enumerate(self.data.keys()):
            sequences = []
            for eeg in enumerate(self.data[key]):
                    sequences.append(list(eeg)[1].astype(np.float64))
            eegs.append(sequences)

        if dataset == "train":
            return eegs[0:134]
        else:
            return eegs[134:]

    def get_labels(self, labeling=1, dataset="train"):
        labels = pd.read_csv("labels.csv")
        label_column = "label" + str(labeling)
        labels = list(labels[label_column])
        if dataset == "train":
            return labels[0:134]
        else:
            return labels[134:]

    def get_genders(self, dataset="train"):
        labels = pd.read_csv("labels.csv")
        genders = list(labels["genders"])
        if dataset == "train":
            return genders[0:134]
        else:
            return genders[134:]

    def plot_motifs_single_electrode(self, signal, mp, m, subtitle=None):

        fig, axs = plt.subplots(2, sharex=True, gridspec_kw={'hspace': 0})
        ymin = min(signal)
        ymax = max(signal)

        if subtitle is not None:
            plt.suptitle(subtitle)

        axs[0].plot(signal)
        axs[0].set_ylabel('EEG')
        for i in range(3):
            idx = np.argsort(mp[:, 0])[i]
            rect = Rectangle((idx, ymin), m, np.abs(ymax - ymin), facecolor='lightgrey')
            axs[0].add_patch(rect)
            # rect = Rectangle((nearest_neighbor_idx, 0), m, 20, facecolor='lightgrey')
            axs[0].set_ylim((ymin - 5, ymax + 5))
            axs[0].add_patch(rect)

            axs[1].axvline(x=idx, linestyle="dashed")

        axs[1].set_xlabel('Time')
        axs[1].set_ylabel('Matrix Profile')
        # axs[1].axvline(x=motif_idx, linestyle="dashed")
        axs[1].plot(mp[:, 0])
        plt.show()

    def plot_motifs_multiple_electrodes(self, sequences, m, subtitle=None):
        fig, axs = plt.subplots(len(sequences), sharex=True, gridspec_kw={'hspace': 0})

        if subtitle is not None:
            plt.suptitle(subtitle)

        ymin = -40
        ymax = 40

        for j, signal in enumerate(sequences):
            mp = stumpy.stump(signal, m)
            # index at which the motif is located
            motif_idx = np.argsort(mp[:, 0])[0]

            axs[j].plot(signal)
            axs[j].set_ylabel('el' + str(j + 1))
            for i in range(len(mp[0])):
                rect = Rectangle((mp[motif_idx, i], ymin), m, np.abs(ymin - ymax), facecolor='lightgrey')
                axs[j].add_patch(rect)
                axs[j].set_ylim((ymin, ymax))
                axs[j].add_patch(rect)

        plt.show()

    def plot_motif(self, signal, mp, m, pair=2):
        # plot motif
        for i in range(pair):
            start = np.argsort(mp[:, 0])[i]
            plt.plot(signal[start:start + m])
        plt.show()

    def plot_eegs(self, sequences, fs, title=None):
        fig, ax = plt.subplots(len(sequences), sharex=True, sharey=True)
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = cycle(prop_cycle.by_key()['color'])
        for i, e in enumerate(sequences):
            ax[i].plot(np.arange(0, len(e)) / fs, e, color=next(colors))
            ax[i].set_ylim((-70, 70))
        plt.subplots_adjust(hspace=0)
        plt.xlabel('Time')

        if title is not None:
            plt.suptitle(title, fontsize=14)
        plt.show()

        return ax

    def plot_consensus_motifs(self, sequences, Ts_idx, subseq_idx, m):
        seed_motif = sequences[Ts_idx][subseq_idx: subseq_idx + m]
        x = np.linspace(0, 1, m)
        nn = np.zeros(len(sequences), dtype=np.int64)
        nn[Ts_idx] = subseq_idx
        for i, e in enumerate(sequences):
            if i != Ts_idx:
                nn[i] = np.argmin(stumpy.core.mass(seed_motif, e))
                lw = 1
                label = "el" + str(i + 1)
            else:
                lw = 6
                label = "el" + str(i + 1) + ' (seed)'
            plt.plot(x, e[nn[i]:nn[i] + m], lw=lw, label=label)
        plt.title('The Consensus Motif')
        plt.xlabel('Time')
        plt.legend()
        plt.show()

    def plot_clustering_consesus_motifs(self, sequences, Ts_idx, subseq_idx, m):

        consensus_motifs = {}
        electrodes = ["el" + str(i + 1) for i in range(len(sequences))]
        best_motif = sequences[Ts_idx][subseq_idx: subseq_idx + m]

        for i, e in enumerate(sequences):
            if i == Ts_idx:
                consensus_motifs[i] = best_motif
            else:
                idx = np.argmin(stumpy.core.mass(best_motif, e))
                consensus_motifs[i] = e[idx: idx + m]
        fig, ax = plt.subplots(ncols=2)

        # plot the consensus motifs
        for i, (electrode, motif) in enumerate(consensus_motifs.items()):
            ax[0].plot(motif, label=electrodes[i])
        ax[0].legend()

        # cluster consensus motifs
        dp = np.zeros(int(comb(len(electrodes), 2)))
        for i, motif in enumerate(combinations(list(consensus_motifs.values()), 2)):
            dp[i] = stumpy.core.mass(motif[0], motif[1])
        Z = linkage(dp, optimal_ordering=True)
        dendrogram(Z, labels=[k + 1 for k in consensus_motifs.keys()])
        ax[0].set_title('Consensus EEG Motifs')
        ax[0].set_xlabel('Number of Electrodes Base Pairs')

        ax[1].set_title('Clustering Using the Consensus Motifs')
        ax[1].set_xlabel('Electrode Number')
        ax[1].set_ylabel('Z-normalized Euclidean Distance')
        plt.show()
