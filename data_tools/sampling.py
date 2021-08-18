import numpy as np


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list, seed=None):
        self.seed = None
        self.src_list = np.unique(src_list).astype(int)
        self.src_list = self.src_list[self.src_list!=0]
        self.dst_list = np.unique(dst_list).astype(int)
        self.dst_list = self.dst_list[self.dst_list!=0]

        if seed is not None:
            self.seed = seed
            self.random_state = np.random.RandomState(self.seed)

    def sample(self, size):
        if self.seed is None:
            src_index = np.random.randint(0, len(self.src_list), size)
            dst_index = np.random.randint(0, len(self.dst_list), size)
        else:

            src_index = self.random_state.randint(0, len(self.src_list), size)
            dst_index = self.random_state.randint(0, len(self.dst_list), size)
        return self.src_list[src_index], self.dst_list[dst_index]

    def reset_random_state(self):
        self.random_state = np.random.RandomState(self.seed)


def get_samplers(
    full_data,
    train_data,
    val_data,
    test_data, 
    data_setting_mode=None,
):
    train_sampler = RandEdgeSampler(train_data.edge_table[:,2], train_data.edge_table[:,3])
    if data_setting_mode=='transductive':
        val_sampler   = RandEdgeSampler(
            np.concatenate([train_data.edge_table[:,2],val_data.edge_table[:,2]],axis=0),
            np.concatenate([train_data.edge_table[:,3],val_data.edge_table[:,3]],axis=0), seed=0
            )
        test_sampler  = RandEdgeSampler(full_data.edge_table[:,2],  full_data.edge_table[:,3], seed=2)
    elif data_setting_mode=='inductive':
        val_sampler   = RandEdgeSampler(val_data.edge_table[:,2],  val_data.edge_table[:,3], seed=0)
        test_sampler  = RandEdgeSampler(test_data.edge_table[:,2], test_data.edge_table[:,3], seed=2)
    else:
        raise ValueError(f'wrong value at data_setting_mode: {str(data_setting_mode)}')
    return train_sampler, val_sampler, test_sampler
