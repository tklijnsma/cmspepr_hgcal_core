import torch
from cmspepr_hgcal_core import utils

def test_scatter_count():
    assert torch.allclose(
        torch.LongTensor([3, 2 ,2]),
        utils.scatter_count(torch.Tensor([0, 0, 0, 1, 1, 2, 2]))
        )

def test_scatter_counts_to_indices():
    assert torch.allclose(
        utils.scatter_counts_to_indices(torch.LongTensor([3, 2, 2])),
        torch.LongTensor([0, 0, 0, 1, 1, 2, 2])
        )

def test_loss_result():
    lr1 = utils.LossResult(part1=1., part2=2.)
    lr2 = utils.LossResult(part1=3., part2=4.)
    lr_comb = lr1 + lr2
    assert lr_comb['part1'] == 4.
    assert lr_comb['part2'] == 6.
    lr1 /= 2.
    assert lr1['part2'] == 1.
