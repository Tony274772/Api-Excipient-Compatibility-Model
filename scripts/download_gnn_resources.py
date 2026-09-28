"""One-time download and verification utility for GNN pretrained resources.

Usage:
    python scripts/download_gnn_resources.py --resource chemeleon --output models/pretrained/chemeleon_mp.pt

Downloads the CheMeleon pretrained D-MPNN checkpoint from Zenodo and verifies its MD5.
This must be run ONCE before using dmpnn_chemprop encoder.
Normal training/inference MUST NOT download anything.
"""

import argparse
import hashlib
import os
import sys
import urllib.request


RESOURCES = {
    "chemeleon": {
        "url": "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
        "md5": "6a80b54fdb7de37ef0374d302f01e8ce",
        "default_output": "models/pretrained/chemeleon_mp.pt",
        "description": "CheMeleon pretrained D-MPNN message-passing weights (~35 MB)",
    },
    "stanford_gat": {
        "url": "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/master/chem/model_architecture/gat_contextpred.pth",
        "md5": None,  # Checksum not officially provided by upstream
        "default_output": "models/pretrained/stanford_gat/gat_contextpred.pth",
        "description": "Stanford Chemistry Pretrained GAT (contextpred)",
    }
}


def compute_md5(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def download_resource(resource_name: str, output_path: str, force: bool = False):
    """Download a pretrained resource and verify its checksum."""
    if resource_name not in RESOURCES:
        print(f"ERROR: Unknown resource '{resource_name}'")
        print(f"Available resources: {list(RESOURCES.keys())}")
        sys.exit(1)

    info = RESOURCES[resource_name]
    url = info["url"]
    expected_md5 = info["md5"]

    print(f"Resource:    {resource_name}")
    print(f"Description: {info['description']}")
    print(f"URL:         {url}")
    print(f"Output:      {output_path}")
    print(f"Expected MD5: {expected_md5}")

    # Check if file already exists
    if os.path.isfile(output_path) and not force:
        actual_md5 = compute_md5(output_path)
        if actual_md5 == expected_md5:
            print(f"\nFile already exists and MD5 is valid: {output_path}")
            print("Use --force to re-download.")
            return
        else:
            print(f"\nWARNING: File exists but MD5 mismatch!")
            print(f"  Expected: {expected_md5}")
            print(f"  Actual:   {actual_md5}")
            print("Re-downloading...")

    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Download
    print(f"\nDownloading from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception as e:
        print(f"ERROR: Download failed: {e}")
        if os.path.isfile(output_path):
            os.remove(output_path)
        sys.exit(1)

    # Verify MD5 if available
    if expected_md5:
        actual_md5 = compute_md5(output_path)
        if actual_md5 != expected_md5:
            print(f"\nERROR: MD5 checksum verification failed!")
            print(f"  Expected: {expected_md5}")
            print(f"  Actual:   {actual_md5}")
            print(f"The downloaded file may be corrupted. Try again or verify the source URL.")
            os.remove(output_path)
            sys.exit(1)
        else:
            print(f"\nDownload complete and verified!")
            print(f"  Path: {output_path}")
            print(f"  MD5:  {actual_md5}")
    else:
        print(f"\nDownload complete (no checksum provided).")
        print(f"  Path: {output_path}")
    print(f"\nYou can now use: python main.py --encoder dmpnn_chemprop ...")


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained GNN resources (one-time setup)"
    )
    parser.add_argument(
        "--resource", required=True, choices=list(RESOURCES.keys()) + ["all"],
        help="Which resource to download (or 'all')"
    )
    parser.add_argument(
        "--output", default=None,
        help="Output file path (default: resource-specific default)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-download even if file exists and is valid"
    )
    args = parser.parse_args()

    if args.resource == "all":
        for res_name in RESOURCES.keys():
            print(f"\n--- Downloading {res_name} ---")
            output_path = RESOURCES[res_name]["default_output"]
            download_resource(res_name, output_path, args.force)
    else:
        output_path = args.output or RESOURCES[args.resource]["default_output"]
        download_resource(args.resource, output_path, args.force)


if __name__ == "__main__":
    main()
