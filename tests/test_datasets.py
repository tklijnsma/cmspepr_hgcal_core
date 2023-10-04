import numpy as np
import torch
import cmspepr_hgcal_core.datasets as datasets


def test_incremental_cluster_index():
    input = torch.LongTensor([13, 4, 4, 13, 4, 13, 13, 42, 4, 4])
    assert torch.allclose(
        datasets.incremental_cluster_index(input),
        torch.LongTensor([1, 0, 0, 1, 0, 1, 1, 2, 0, 0]),
    )
    # Noise index should get 0 if it is supplied:
    assert torch.allclose(
        datasets.incremental_cluster_index(input, noise_index=13),
        torch.LongTensor([0, 1, 1, 0, 1, 0, 0, 2, 1, 1]),
    )
    # 0 should still be reserved for noise_index even if it is not present:
    assert torch.allclose(
        datasets.incremental_cluster_index(input, noise_index=-99),
        torch.LongTensor([2, 1, 1, 2, 1, 2, 2, 3, 1, 1]),
    )


def test_incremental_cluster_index_np():
    input = np.array([13, 4, 4, 13, 4, 13, 13, 42, 4, 4])
    np.testing.assert_array_equal(
        datasets.incremental_cluster_index_np(input),
        np.array([1, 0, 0, 1, 0, 1, 1, 2, 0, 0]),
    )
    # Noise index should get 0 if it is supplied:
    np.testing.assert_array_equal(
        datasets.incremental_cluster_index_np(input, noise_index=13),
        np.array([0, 1, 1, 0, 1, 0, 0, 2, 1, 1]),
    )
    # 0 should still be reserved for noise_index even if it is not present:
    np.testing.assert_array_equal(
        datasets.incremental_cluster_index_np(input, noise_index=-99),
        np.array([2, 1, 1, 2, 1, 2, 2, 3, 1, 1]),
    )


def test_noise_reduction():
    input = np.array([9, -1, 9, 9, -1, -1, -1, 9, -1, -1, -1, -1, 9])
    mask = datasets.mask_fraction_of_noise(input, 0.5)
    assert mask.shape[0] == input.shape[0]
    assert mask.sum() == 5 + 4
    out = input[mask]
    assert (out == 9).sum() == (input == 9).sum()
    assert (out == -1).sum() == 0.5 * (input == -1).sum()


def test_npzfile_inst_to_torch_data():
    # fmt: off
    X = np.array([
        # E        eta    0   theta R      x     y       z    t
        [3.83e-03, -2.76, 0., 3.02, 325.,  40.7, -3.60, -322, -1.  ],
        [3.83e-03, -2.80, 0., 3.02, 324.,  39.3, -2.00, -322, -1.  ],
        [1.11e-01, -2.41, 0., 2.96, 327.,  48.4, -33.0, -322., 5.05],
        [3.83e-03, -2.83, 0., 3.02, 324.,  37.9, -2.80, -322, -1.  ],
        [8.38e-02, -2.56, 0., 2.99, 326.,  42.8, -25.8, -322., 5.15],
        [8.69e-02, -2.30, 0., 2.94, 330., -64.6,  12.1, -323., 5.90],
        [3.83e-03, -2.85, 0., 3.03, 324.,  36.5, -7.61, -322, -1.  ],
        [1.52e-01, -2.56, 0., 2.99, 327.,  42.8, -25.8, -323., 5.10],
        [2.47e-01, -2.56, 0., 2.99, 329.,  43.5, -26.2, -325., 5.05],
        [3.83e-03, -2.95, 0., 3.04, 324.,  33.7, -0.40, -322, -1.  ]
        ])
    # fmt: on
    y = np.array([-1, -1, 4, -1, 4, 5, -1, 4, 5, -1])
    y = np.expand_dims(y, -1)
    recHitTruthEnergy = np.random.rand(10, 1)
    recHitTruthPosition = np.random.rand(10, 2)
    recHitTruthTime = np.random.rand(10, 1)
    recHitTruthID = np.random.rand(10, 1)
    d = {
        'recHitFeatures': X,
        'recHitTruthClusterIdx': y,
        'recHitTruthEnergy': recHitTruthEnergy,
        'recHitTruthPosition': recHitTruthPosition,
        'recHitTruthTime': recHitTruthTime,
        'recHitTruthID': recHitTruthID,
    }
    data = datasets._taus2021_npzfile_inst_to_torch_data(d)
    assert data.x.size() == X.shape
    assert data.x[:, 7].mean() > 0.0
    assert len(torch.unique(data.y)) == torch.max(data.y) + 1
    assert data.npz == 'mem'
