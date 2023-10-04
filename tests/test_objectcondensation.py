import pytest
import torch
import numpy as np
from torch_geometric.data import Data

import cmspepr_hgcal_core.objectcondensation as oc

torch.manual_seed(1001)
np.random.seed(1001)


@pytest.fixture
def simple_clustering_problem():
    cluster_index = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    betas = np.random.rand(10) * .01
    betas[np.array([1, 5, 8])] += .15 # Make fake condensation points
    # Make a clustering space that is easy to cluster
    cluster_space_coords = np.random.rand(10,2) + 2.*np.expand_dims(cluster_index, -1)
    return betas, cluster_space_coords

def test_get_clustering_np(simple_clustering_problem):
    output = oc.get_clustering_np(*simple_clustering_problem)
    expected_output = np.array([1, 1, 1, 1, 5, 5, 5, 8, 8, 8])
    np.testing.assert_array_equal(output, expected_output)

def test_get_clustering_torch(simple_clustering_problem):
    betas, cluster_space_coords = simple_clustering_problem
    betas = torch.FloatTensor(betas)
    cluster_space_coords = torch.FloatTensor(cluster_space_coords)
    output = oc.get_clustering(betas, cluster_space_coords)
    expected_output = torch.LongTensor([1, 1, 1, 1, 5, 5, 5, 8, 8, 8])
    assert torch.allclose(output, expected_output)


def test_reincrementalize():
    y = torch.LongTensor([0, 6, 1, 0, 1, 6, 3, 0, 3])
    expected = torch.LongTensor([0, 3, 1, 0, 1, 3, 2, 0, 2])
    assert torch.allclose(oc.reincrementalize(y), expected)
    # Should not depend on order
    order = torch.randperm(y.size(0))
    y = y[order]
    expected = expected[order]
    assert torch.allclose(oc.reincrementalize(y), expected)
    # Should not do anything if there are no holes
    y = torch.LongTensor([3, 0, 1, 0, 1, 3, 2, 0, 2])
    expected = torch.LongTensor([3, 0, 1, 0, 1, 3, 2, 0, 2])


def test_oc():
    model_out = torch.FloatTensor([
        [.01, 0.40, 0.40],
        [.01, 0.10, 0.90],
        [.12, 0.70, 0.70],
        [.01, 0.90, 0.10],
        [.13, 0.72, 0.72],
        ])
    y = torch.LongTensor([0, 0, 1, 0, 1])
    lr = oc.ObjectCondensation(model_out, y).loss()
    print(lr)


def test_oc_batch():
    # fmt: off
    data = Data(
        y = torch.LongTensor([
            0, 1, 0, 1,
            0, 1, 1, 0, 1
            ]),
        batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 1, 1])
        )
    model_out = torch.rand((9, 3))
    # fmt: on
    lr_batch = oc.oc_loss(model_out, data)
    lr1 = oc.ObjectCondensation(model_out[:4], data.y[:4]).loss()
    lr2 = oc.ObjectCondensation(model_out[4:], data.y[4:]).loss()
    assert ((lr1+lr2)/2.).loss == lr_batch.loss


def test_oc_batch_noise_filter():
    # fmt: off
    data = Data(
        y = torch.LongTensor([
            0, 1, 2, 0, 1, 2,
            0, 1, 2, 1, 0, 1, 2
            ]),
        batch = torch.LongTensor([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        )
    model_out = torch.rand((9, 3))
    pass_noise_filter = torch.BoolTensor([
            1, 1, 0, 1, 1, 0,
            1, 1, 0, 1, 1, 1, 0
            ])
    # fmt: on
    lr_with_noise_filter = oc.oc_loss_with_noise_filter(model_out, data, pass_noise_filter)
    # Compute manual
    data2 = Data(y=data.y[pass_noise_filter], batch=data.batch[pass_noise_filter])
    lr_manual = oc.oc_loss(model_out, data2)
    assert lr_with_noise_filter.loss == lr_manual.loss
