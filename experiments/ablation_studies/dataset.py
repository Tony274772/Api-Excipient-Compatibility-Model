"""PGBDataset that hands the model only the first `prior_dim` entries of the prior vector."""

from src.dataset import PGBDataset


class PriorSlicedPGBDataset(PGBDataset):
    """
    prior_dim=5 -> [p_exact, p_delta, p_family, log_n, unseen]  (unchanged PGBDataset)
    prior_dim=1 -> [p_exact]

    p_exact is always index 0, so PGBHead's logit offset (prior_vec[:, 0]) is the
    same quantity in both settings. All other features are untouched.
    """

    def __init__(self, *args, prior_dim: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        if prior_dim not in (1, 5):
            raise ValueError(f"prior_dim must be 1 or 5, got {prior_dim}")
        self.prior_dim = prior_dim

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["prior_vec"] = item["prior_vec"][: self.prior_dim]
        return item
