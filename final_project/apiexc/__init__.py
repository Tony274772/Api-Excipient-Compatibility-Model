"""apiexc: API-excipient compatibility prediction with the Morgan-PGB model.

    paths        every file / folder location, anchored on the project root
    config       Config dataclass (stored inside every checkpoint)
    pipeline     shared recipe: data splits, training config, data loaders, checkpoint loading, scoring
    splitting    how the default train / val split is made (Butina API clusters + GroupShuffleSplit)
    dataset      PGBDataset: turns a CSV of pairs into model inputs
    features/    molecular features, mechanism flags, excipient prior table
    models/      the Prior-Gated Bilinear (PGB) model and its encoder
    training/    training loop and losses
    evaluation/  metrics and threshold tuning
    ablations/   model variants, mechanism flags v2 and prior tables for the ablation study
    explain/     SHAP wrappers, feature names and plots
"""
