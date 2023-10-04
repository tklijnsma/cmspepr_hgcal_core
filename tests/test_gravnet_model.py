import torch
from torch_geometric.data import Data
import cmspepr_hgcal_core.gravnet_model as gm

torch.manual_seed(1001)

def test_global_exchange():
    x = torch.rand(10,3)
    batch = torch.LongTensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
    x_exchanged = gm.global_exchange(x, batch)
    assert x_exchanged.size() == (10, 12)
    # Compare first 4 rows manually with expectation
    assert torch.allclose(x[:4].mean(dim=0), x_exchanged[0,:3])
    assert torch.allclose(x[:4].min(dim=0)[0], x_exchanged[0,3:6])
    assert torch.allclose(x[:4].max(dim=0)[0], x_exchanged[0,6:9])


def test_gravnet_model_runs():
    """
    Tests whether the GravNetModel returns output.
    Does not perform any checking on that output.
    """
    n_hits = 100
    n_events = 5
    batch = (n_events*torch.rand(n_hits)).long()
    model = gm.GravnetModel(4, 3)
    data = Data(x=torch.rand(n_hits, 4).float(), batch=batch)
    out = model(data)
    assert out.size() == (n_hits, 3)


def test_gravnet_model_with_noisefilter_runs():
    n_hits = 100
    n_events = 5
    batch = (n_events*torch.rand(n_hits)).long()
    model = gm.GravnetModelWithNoiseFilter(4, 3)
    data = Data(x=torch.rand(n_hits, 4).float(), batch=batch)
    out_noise_filter, pass_noise_filter, out_gravnet = model(data)
    assert out_noise_filter.size() == (n_hits, 2)
    assert pass_noise_filter.size() == (n_hits,)
    assert out_gravnet.size(0) <= n_hits
    assert out_gravnet.size(1) == 3
