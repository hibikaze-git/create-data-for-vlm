"""
common-canvas/commoncatalog-cc-by-saをサブディレクトリごとにダウンロード
python partial_downloader.py 0
"""

import sys

from huggingface_hub import snapshot_download


def main():
    if len(sys.argv) != 2:
        print("Usage: python partial_downloader.py <subdirectory_number>")
        return

    subdirectory_number = sys.argv[1]

    try:
        subdirectory_number = int(subdirectory_number)
    except ValueError:
        print("Error: Subdirectory number must be an integer.")
        return

    pattern = f"{subdirectory_number}/*"

    snapshot_download(
        repo_id="common-canvas/commoncatalog-cc-by-sa",
        allow_patterns=pattern,
        repo_type="dataset",
        cache_dir="./cache",
        local_dir="./commoncatalog-cc-by-sa-download",
    )
    print(f"Download of subdirectory {subdirectory_number} completed.")


if __name__ == "__main__":
    main()
