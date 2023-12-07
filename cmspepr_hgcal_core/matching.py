
import numpy as np
from torch.functional import _return_counts



def match(clustering1, clustering2, weights=None, threshold=.2, noise_index=0):
    """
    Matching algorithm for two clusterings of the same set of points.

    In a nutshell, the algorithm does the following:

    - Determine number of clusters in both clusterings
    - For every pair of clusters, calculate the weighted intersection-over-minimum (iom)
    - Sort pairs by decreasing intersection (not iom)
    - Loop over pairs; if the iom < threshold, it's a match.
    - Allow one-to-many matches either way (but not many-to-many)
    - Return all matched pairs in the following format: [index1], [index2], [iom]
    """
    if weights is None: weights = np.ones_like(clustering1)
    cluster_ids1, cluster_indices1 = np.unique(clustering1, return_inverse=True)
    cluster_ids2, cluster_indices2 = np.unique(clustering2, return_inverse=True)
    n_clusters1 = cluster_ids1.shape[0]
    n_clusters2 = cluster_ids2.shape[0]
    # Pre-calculate all 'areas' for all clusters
    areas1 = {id : weights[clustering1 == id].sum() for id in cluster_ids1}
    areas2 = {id : weights[clustering2 == id].sum() for id in cluster_ids2}
    # Make list of all pairs
    a = np.repeat(np.arange(n_clusters1), n_clusters2)
    b = np.repeat(np.expand_dims(np.arange(n_clusters2), -1), n_clusters1, axis=1).T.ravel()
    pairs = np.vstack((cluster_ids1[a], cluster_ids2[b])).T
    if noise_index is not None:
        # Remove pairs with a noise index in there
        pairs = pairs[~np.amax(pairs==noise_index, axis=-1).astype(bool)]
    # Calculate weighted ioms for every pair
    ioms = np.zeros(pairs.shape[0])
    intersections = np.zeros(pairs.shape[0])
    for i_pair, (id1, id2) in enumerate(pairs):
        intersection = weights[(clustering1==id1) & (clustering2==id2)].sum()
        minimum = min(areas1[id1], areas2[id2])
        ioms[i_pair] = intersection / minimum
        intersections[i_pair] = intersection
    # Sort
    order = np.argsort(intersections)[::-1]
    intersections = intersections[order]
    ioms = ioms[order]
    pairs = pairs[order]
    # Matching algo
    canhavemorematches_1 = set(cluster_ids1)
    canhavemorematches_2 = set(cluster_ids2)
    matched_1 = set()
    matched_2 = set()
    matches = []
    DEBUG = False
    for iom, intersection, (i1, i2) in zip(ioms, intersections, pairs):
        if iom < threshold: continue
        if DEBUG: print(f'[left]{i1} with [right]{i2}, {iom=:.2f}, {intersection=:.2f}')
        if i1 not in canhavemorematches_1 or i2 not in canhavemorematches_2:
            if DEBUG: print(
                f'  Match {i1} (1) to {i2} (2) with {iom=:.2f}/{intersection=:.2f} cannot be made;'
                f' can have more matches: {i1}:{i1 in canhavemorematches_1}, {i2}:{i2 in canhavemorematches_2}'
                )
            continue
        if i1 in matched_1 and i2 in matched_2:
            if DEBUG: print(f'  Both [left]{i1} and [right]{i2} have >0 matches, cannot make this match')
            continue
        # Make the match
        matches.append([i1, i2, iom])
        if i1 in matched_1:
            i2s = [j2 for j1, j2, _ in matches if j1==i1]
            if DEBUG: print(f'  [left]{i1} (now) has >1 match; [right]{i2s} are not allowed more matches')
            canhavemorematches_2.difference_update(i2s)
        elif i2 in matched_2:
            i1s = [j1 for j1, j2, _ in matches if j2==i2]
            if DEBUG: print(f'  [right]{i2} (now) has >1 match; [left]{i1s} are not allowed more matches')
            canhavemorematches_1.difference_update(i1s)
        matched_1.add(i1)
        matched_2.add(i2)
    if len(matches) == 0:
        print('Warning: No matches at all')
        return [], [], []
    matches = np.array(matches)
    i1s, i2s, ioms = matches[:,0].astype(np.int32), matches[:,1].astype(np.int32), matches[:,2]
    return i1s, i2s, ioms

def group_matching(i1s, i2s, return_lists=True):
    match_dict_1_to_2 = {}
    match_dict_2_to_1 = {}
    is_used_1 = set()
    is_used_2 = set()
    for i1, i2 in zip(i1s, i2s):
        i1 = int(i1)
        i2 = int(i2)
        i1_used = i1 in is_used_1
        i2_used = i2 in is_used_2
        if not(i1_used) and not(i2_used):
            match_dict_1_to_2[i1] = [i2]
            match_dict_2_to_1[i2] = [i1]
            is_used_1.add(i1)
            is_used_2.add(i2)
        elif i1_used and i2_used:
            raise Exception(
                f'Detected many-to-many match:'
                f' [left]{i1} and [right]{i2} are both already matched to something else'
                )
        elif i1 in is_used_1:
            match_dict_1_to_2[i1].append(i2)
            match_dict_2_to_1.pop(i2, None)
        elif i2 in is_used_2:
            match_dict_2_to_1[i2].append(i1)
            match_dict_1_to_2.pop(i1, None)
    if return_lists:
        matches = [[[k], v] for k, v in match_dict_1_to_2.items()]
        matches.extend([[v, [k]] for k, v in match_dict_2_to_1.items() if len(v)>1])
    else:    
        matches = [[k, (v if len(v)>1 else v[0])] for k, v in match_dict_1_to_2.items()]
        matches.extend([[v, k] for k, v in match_dict_2_to_1.items() if len(v)>1])
    return matches
