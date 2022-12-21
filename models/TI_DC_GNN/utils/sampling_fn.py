import numpy as np


def sample_full(sorted_lst, n_sample):
    return sorted_lst


def sample_uniform(sorted_lst, n_sample):
    if n_sample == 0:
        return []
    if n_sample >= len(sorted_lst):
        return sorted_lst
    interval_len = len(sorted_lst) // n_sample
    return list(reversed([sorted_lst[-i] for i in range(1, n_sample+1, interval_len)]))  # TODO


def sample_random(sorted_lst, n_sample):
    if n_sample == 0:
        return []
    return np.random.choice(sorted_lst, n_sample, replace=False)


def sample_last(sorted_lst, n_sample):
    if n_sample == 0:
        return []
    return sorted_lst[-n_sample:]


def get_sampling_fn(sampling_fn_name):
    if sampling_fn_name == 'full':
        return sample_full
    if sampling_fn_name == 'uniform':
        return sample_uniform
    if sampling_fn_name == 'random':
        return sample_random
    if sampling_fn_name == 'last':
        return sample_last


def get_sampling_fns(sampling_fn_names):
    return [get_sampling_fn(name) for name in sampling_fn_names]
