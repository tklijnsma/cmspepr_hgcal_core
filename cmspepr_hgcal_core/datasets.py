import os.path as osp
import glob

import numpy as np
import torch
from torch_geometric.data import Data, Dataset

import cmspepr_hgcal_core

logger = cmspepr_hgcal_core.logger
np.random.seed(1001)


def incremental_cluster_index(input: torch.Tensor, noise_index=None):
    """
    Build a map that translates arbitrary indices to ordered starting from zero

    By default the first unique index will be 0 in the output, the next 1, etc.
    E.g. [13 -1 -1 13 -1 13 13 42 -1 -1] -> [0 1 1 0 1 0 0 2 1 1]

    If noise_index is not None, the output will be 0 where input==noise_index:
    E.g. noise_index=-1, [13 -1 -1 13 -1 13 13 42 -1 -1] -> [1 0 0 1 0 1 1 2 0 0]

    If noise_index is not None but the input does not contain noise_index, 0
    will still be reserved for it:
    E.g. noise_index=-1, [13 4 4 13 4 13 13 42 4 4] -> [1 2 2 1 2 1 1 3 2 2]
    """
    unique_indices, locations = torch.unique(input, return_inverse=True, sorted=True)
    cluster_index_map = torch.arange(unique_indices.size(0))
    if noise_index is not None:
        if noise_index in unique_indices:
            # Sort so that 0 aligns with the noise_index
            cluster_index_map = cluster_index_map[(unique_indices != noise_index).argsort()]
        else:
            # Still reserve 0 for noise, even if it's not present
            cluster_index_map += 1
    return torch.gather(cluster_index_map, 0, locations).long()


def incremental_cluster_index_np(input: np.array, noise_index=None):
    """
    Reimplementation of incremental_cluster_index for numpy arrays
    """
    unique_indices, locations = np.unique(input, return_inverse=True)
    cluster_index_map = np.arange(unique_indices.shape[0])
    if noise_index is not None:
        if noise_index in unique_indices:
            # Sort so that 0 aligns with the noise_index
            cluster_index_map = cluster_index_map[(unique_indices != noise_index).argsort()]
        else:
            # Still reserve 0 for noise, even if it's not present
            cluster_index_map += 1
    return np.take(cluster_index_map, locations)


def mask_fraction_of_noise(y: np.array, reduce_fraction: float, noise_index: int=-1, seed: int=None) -> np.array:
    """
    Create a mask that throws out a fraction of noise (but keeps all signal).
    
    Args:
        y (np.array): Cluster index per hit (not necessarily incremental).
        reduce_fraction (float): Fraction of noise that must be masked
        noise_index (int): where y==noise_index is noise (default: -1)
        seed (int): Seed for the mask (default: None)

    Returns:
        np.array: Mask, a boolean array of hits to keep
    """
    # Get indices of where noise is
    noise_indices = np.nonzero(y==noise_index)[0]
    n_target_noise = int((1.-reduce_fraction) * len(noise_indices))
    # Randomly (but with fixed seed) select some noise indices to throw away
    idxs_to_throw_away = np.random.default_rng(seed).choice(
        noise_indices,n_target_noise,replace=False
        )
    # Invert to a mask of indices to _keep_
    mask = np.ones(len(y), dtype=bool)
    mask[idxs_to_throw_away] = False
    return mask        


class FakeDataset(Dataset):
    """
    Random number dataset to test with.
    Generates numbers on the fly, but also caches them so .get(i) will return
    something consistent
    """
    def __init__(self, n_events=100):
        super(FakeDataset, self).__init__('nofile')
        self.cache = {}
        self.n_events = n_events

    def get(self, i):
        if i >= self.n_events: raise IndexError
        if i not in self.cache:
            n_hits = np.random.randint(10, 100)
            n_clusters = min(np.random.randint(1, 6), n_hits)
            x = np.random.rand(n_hits, 5)
            y = (np.random.rand(n_hits) * n_clusters).astype(np.int8)
            # Also make a cluster 'truth': energy, boundary_x, boundary_y, pid (4)
            y_cluster = np.random.rand(n_clusters, 4)
            # pid (last column) should be an integer; do 3 particle classes now
            y_cluster[:,-1] = np.floor(y_cluster[:,-1] * 3)
            self.cache[i] = Data(
                x = torch.from_numpy(x).type(torch.float),
                y = torch.from_numpy(y),
                truth_cluster_props = torch.from_numpy(y_cluster)
                )
        return self.cache[i]

    def __len__(self):
        return self.n_events

    def len(self):
        return self.n_events


class BlobsDataset(Dataset):
    """
    Dataset around sklearn.datasets.make_blobs
    """
    
    def __init__(self, n_events=100, seed_offset=0):
        super(BlobsDataset, self).__init__('nofile')
        self.cache = {}
        self.n_events = n_events
        self.cluster_space_dim = 2
        self.seed_offset = seed_offset

    def get(self, i):
        from sklearn.datasets import make_blobs

        if i >= self.n_events: raise IndexError
        if i not in self.cache:
            n_hits = np.random.randint(50, 70)
            n_clusters = min(np.random.randint(2, 4), n_hits)
            n_bkg = np.random.randint(10, 20)
            # Generate the 'signal'
            X, y = make_blobs(
                n_samples=n_hits,
                centers=n_clusters, n_features=self.cluster_space_dim,
                random_state=i+self.seed_offset
                )
            y += 1 # To reserve index 0 for background
            # Add background
            cluster_space_min = np.min(X, axis=0)
            cluster_space_max = np.max(X, axis=0)
            cluster_space_width = cluster_space_max - cluster_space_min
            X_bkg = cluster_space_min + np.random.rand(n_bkg, self.cluster_space_dim)*cluster_space_width
            y_bkg = np.zeros(n_bkg)
            X = np.concatenate((X,X_bkg))
            y = np.concatenate((y,y_bkg))
            # Calculate geom centers
            truth_cluster_props = np.zeros((n_hits+n_bkg,2))
            for i in range(1,n_clusters+1):
                truth_cluster_props[y==i] = np.mean(X[y==i], axis=0)
            # shuffle
            order = np.random.permutation(n_hits+n_bkg)
            X = X[order]
            y = y[order]
            truth_cluster_props = truth_cluster_props[order]
            self.cache[i] = Data(
                x = torch.from_numpy(X).float(),
                y = torch.from_numpy(y).long(),
                truth_cluster_props = torch.from_numpy(truth_cluster_props).float()
                )
        return self.cache[i]

    def __len__(self):
        return self.n_events

    def len(self):
        return self.n_events


# ___________________________________________________
# Taus 2021, no PU

def _taus2021_npzfile_inst_to_torch_data(d: np.lib.npyio.NpzFile, flip: bool=True, sort_by_cluster_index: bool=True, npz_path: str='mem') -> Data:
    """
    Takes an opened npz file, does some processing, and returns a Data object.
    """
    x = d['recHitFeatures']
    y = d['recHitTruthClusterIdx'].squeeze()
    if flip:
        # Flip z-dependent coordinates
        x[:,1] *= -1 # eta
        x[:,7] *= -1 # z
    # y does not have cluster indices nicely starting at 0; fix that
    y = incremental_cluster_index_np(y, noise_index=-1)
    if np.all(y == 0): logger.warning('No objects in', npz_path)
    truth_cluster_props = np.hstack((
        d['recHitTruthEnergy'],
        d['recHitTruthPosition'],
        d['recHitTruthTime'],
        d['recHitTruthID'],
        ))
    assert truth_cluster_props.shape == (x.shape[0], 5)
    if sort_by_cluster_index:
        order = y.argsort()
        x = x[order]
        y = y[order]
        truth_cluster_props = truth_cluster_props[order]
    data = Data(
        x = torch.from_numpy(x).type(torch.float),
        y = torch.from_numpy(y).type(torch.int),
        truth_cluster_props = torch.from_numpy(truth_cluster_props).type(torch.float),
        )
    data.npz = npz_path
    return data


def taus2021_npz_to_torch_data(npzfile: str, flip: bool=True, sort_by_cluster_index: bool=True):
    return _taus2021_npzfile_inst_to_torch_data(np.load(npzfile), flip, sort_by_cluster_index, osp.abspath(npzfile))


class Taus2021Dataset(Dataset):
    """Tau dataset from 2021, with 0 PU.
    
    Features in x:
    0 recHitEnergy,
    1 recHitEta,
    2 zeroFeature, #indicator if it is track or not
    3 recHitTheta,
    4 recHitR,
    5 recHitX,
    6 recHitY,
    7 recHitZ,
    8 recHitTime
    (https://github.com/cms-pepr/HGCalML/blob/master/modules/datastructures/TrainData_NanoML.py#L211-L221)

    Args:
        flip (bool): If True, flips the negative endcap z-values to positive
        reduce_noise (float): Randomly delete a fraction of noise. Useful
            to speed up training.
    """
    def __init__(self, path, flip=True, reduce_noise: float=None):
        super().__init__(path)
        self.npzs = list(sorted(glob.iglob(path + '/*.npz')))
        self.flip = flip
        self.reduce_noise = reduce_noise
        self.noise_index = -1

    def blacklist(self, npzs):
        """
        Remove a list of npzs from the dataset
        Useful to remove bad events
        """
        for npz in npzs: self.npzs.remove(npz)

    def get(self, i):
        data = taus2021_npz_to_torch_data(self.npzs[i], self.flip)
        if self.reduce_noise is not None:
            mask = mask_fraction_of_noise(data.y, self.reduce_noise, -1., seed=i)
            data.x = data.x[mask]
            data.y = data.y[mask]
            data.truth_cluster_props = data.truth_cluster_props[mask]
        return data

    def __len__(self):
        return self.len()

    def len(self):
        return len(self.npzs)

    def split(self, fraction):
        """
        Creates two new instances with a fraction of events split
        """
        left = self.__class__(self.root, self.flip, self.reduce_noise)
        right = self.__class__(self.root, self.flip, self.reduce_noise)
        split_index = int(fraction*len(self))
        left.npzs = self.npzs[:split_index]
        right.npzs = self.npzs[split_index:]
        return left, right


# ___________________________________________________
# Taus 2023, with PU

class TauPUDataset(Dataset):
    def __init__(self, path:str, noise_reduction: float=None):
        super(TauPUDataset, self).__init__(path)
        self.npzs = list(sorted(glob.iglob(path + '/*.npz')))
        self.noise_reduction = noise_reduction

    def __len__(self):
        return len(self.npzs)

    def len(self):
        return len(self.npzs)

    def get(self, i):
        import hgcal_npz
        event = hgcal_npz.Event.load(self.npzs[i])
        x = event.rechits.to_numpy([
            'RecHitHGC_x',
            'RecHitHGC_y',
            'RecHitHGC_z',
            'RecHitHGC_energy',
            'RecHitHGC_time',
            'RecHitHGC_theta',
            'RecHitHGC_eta',
            'RecHitHGC_phi',
            'RecHitHGC_R'
            ])
        y = event.rechits.get('RecHitHGC_incrClusterIdx')

        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise Exception(f'Encountered NaN in {self.npzs[i]}')
        
        if self.noise_reduction is not None:
            # Get indices of where noise is
            noise_indices, _ = np.nonzero(y==0)
            # Randomly (but with fixed seed) select some noise indices to throw away
            idxs_to_throw_away = np.random.default_rng(i).choice(
                noise_indices,
                int(self.noise_reduction*len(noise_indices)),
                replace=False
                )
            # Invert to a mask of indices to _keep_
            keep = np.ones(len(y), dtype=bool)
            keep[idxs_to_throw_away] = False
            # Select
            x = x[keep]
            y = y[keep]
            
        return Data(
            x = torch.from_numpy(x).type(torch.float),
            y = torch.from_numpy(y).type(torch.int),
            inpz = torch.Tensor([i])
            )

    def split(self, fraction):
        """
        Creates two new instances with a fraction of events split
        """
        left = self.__class__(self.root, self.noise_reduction)
        right = self.__class__(self.root, self.noise_reduction)
        split_index = int(fraction*len(self))
        left.npzs = self.npzs[:split_index]
        right.npzs = self.npzs[split_index:]
        return left, right
