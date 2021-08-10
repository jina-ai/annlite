class IVFPQIndex(CellContainer):
    def __init__(
            self,
            d_vector,
            n_subvectors=8,
            n_cells=128,
            initial_size=None,
            expand_step_size=128,
            expand_mode='double',
            distance='euclidean',
            device='cuda:0',
            pq_use_residual=False,
            verbose=0,
    ):
