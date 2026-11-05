"""Model inputs computed from SMILES.

    molecular        Morgan / MACCS fingerprints, 18 chemistry flags, 12 PGB descriptors, salt context
    mechanisms       5 SMARTS mechanism flags per API-excipient pair (Maillard, hydrolysis, ...)
    excipient_prior  excipient prior table: smoothed train incompatibility rate per excipient
    descriptors      RDKit descriptor tables (used by the legacy CompatibilityDataset only)
"""
