class Batcher:
    def __init__(self, batch_size, edge_table):
        self.batch_size = batch_size
        self.batch_cnt = self.calc_batch_cnt(edge_table.shape[0])
        self.full_batch = (
            edge_table[:, 2].astype(int),
            edge_table[:, 3].astype(int),
            edge_table[:, 1],
            edge_table[:, 0]
        )

    def calc_batch_cnt(self, full_size):
        cnt_ops = full_size // self.batch_size
        if cnt_ops * self.batch_size < full_size:
            cnt_ops += 1
        return cnt_ops

    def get_batch_by_num(self, n):
        start_ind = n * self.batch_size
        end_ind = start_ind + self.batch_size
        return (
            self.full_batch[0][start_ind:end_ind],
            self.full_batch[1][start_ind:end_ind],
            self.full_batch[2][start_ind:end_ind],
            self.full_batch[3][start_ind:end_ind],
            None
        )

