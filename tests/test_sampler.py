import pytest
import numpy as np

from pytorch_balanced_sampler import WeightedRandomBatchSampler, WeightedFixedBatchSampler


# -- WeightedRandomBatchSampler Tests -------------------------------------------------------------

@pytest.mark.parametrize('class_weights, class_idxs, batch_size, n_batches', [
    (
        np.array([0.5, 0.5]),
        [[0, 2, 4, 6], [1, 3, 5, 7]],
        32,
        5
    ),
    (
        np.array([0.5, 0.3, 0.2]),
        [[0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        64,
        3
    ),
])
def test_weighted_random_batch_size(class_weights, class_idxs, batch_size, n_batches):
    sampler = WeightedRandomBatchSampler(class_weights, class_idxs, batch_size, n_batches)
    for batch in sampler:
        print(batch)
        assert len(batch) == batch_size


@pytest.mark.parametrize('class_weights, class_idxs, batch_size, n_batches', [
    (
        np.array([0.5, 0.5]),
        [[0, 2, 4, 6], [1, 3, 5, 7]],
        32,
        5
    ),
    (
        np.array([0.5, 0.3, 0.2]),
        [[0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        64,
        3
    ),
])
def test_weighted_random_epoch_size(class_weights, class_idxs, batch_size, n_batches):
    sampler = WeightedRandomBatchSampler(class_weights, class_idxs, batch_size, n_batches)
    epoch_size = 0
    for _ in sampler:
        epoch_size += 1
    assert epoch_size == n_batches
    assert epoch_size == len(sampler)


@pytest.mark.parametrize('class_weights, class_idxs, batch_size, n_batches', [
    (
        np.array([0.5, 0.5]),
        [[0, 2, 4, 6], [1, 3, 5, 7]],
        32,
        5
    ),
    (
        np.array([0.5, 0.3, 0.2]),
        [[0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        64,
        3
    ),
])
def test_weighted_random_idx_selection(class_weights, class_idxs, batch_size, n_batches):
    sampler = WeightedRandomBatchSampler(class_weights, class_idxs, batch_size, n_batches)

    provided_idxs = []
    for idxs in class_idxs:
        provided_idxs.extend(idxs)

    selected_idxs = []
    for batch in sampler:
        selected_idxs.extend(batch)

    for idx in selected_idxs:
        assert idx in provided_idxs


# -- WeightedFixedBatchSampler Tests -------------------------------------------------------------

@pytest.mark.parametrize('class_samples_per_batch, class_idxs, n_batches', [
    (
        np.array([16, 16]),
        [[0, 2, 4, 6], [1, 3, 5, 7]],
        5
    ),
    (
        np.array([23, 34, 16]),
        [[0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        3
    ),
])
def test_weighted_fixed_batch_size(class_samples_per_batch, class_idxs, n_batches):
    sampler = WeightedFixedBatchSampler(class_samples_per_batch, class_idxs, n_batches)
    for batch in sampler:
        print(batch)
        assert len(batch) == class_samples_per_batch.sum()


@pytest.mark.parametrize('class_samples_per_batch, class_idxs, n_batches', [
    (
        np.array([16, 16]),
        [[0, 2, 4, 6], [1, 3, 5, 7]],
        5
    ),
    (
        np.array([23, 34, 16]),
        [[0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        3
    ),
])
def test_weighted_fixed_epoch_size(class_samples_per_batch, class_idxs, n_batches):
    sampler = WeightedFixedBatchSampler(class_samples_per_batch, class_idxs, n_batches)
    epoch_size = 0
    for _ in sampler:
        epoch_size += 1
    assert epoch_size == n_batches
    assert epoch_size == len(sampler)


@pytest.mark.parametrize('class_samples_per_batch, class_idxs, n_batches', [
    (
        np.array([16, 16]),
        [[0, 2, 4, 6], [1, 3, 5, 7]],
        5
    ),
    (
        np.array([23, 34, 16]),
        [[0, 2, 4, 6], [1, 3, 5, 7], [8, 9, 10, 11, 12, 13, 14, 15]],
        3
    ),
])
def test_weighted_fixed_idx_selection(class_samples_per_batch, class_idxs, n_batches):
    sampler = WeightedFixedBatchSampler(class_samples_per_batch, class_idxs, n_batches)

    provided_idxs = []
    for idxs in class_idxs:
        provided_idxs.extend(idxs)

    selected_idxs = []
    for batch in sampler:
        selected_idxs.extend(batch)

    for idx in selected_idxs:
        assert idx in provided_idxs
