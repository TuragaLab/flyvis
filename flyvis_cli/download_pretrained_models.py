import argparse
import hashlib
import io
import os
import zipfile
from typing import Dict

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from flyvis import root_dir

checksums: Dict[str, str] = {
    "results_pretrained_models.zip": (
        "71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1"
    ),
    "results_umap_and_clustering.zip": (
        "6300ec1e678cde4d95300ac88e697b75fda32f6404ebe9a50f71e9f653aa9b19"
    ),
}

large_files: Dict[str, bool] = {
    "results_pretrained_models.zip": False,
    "results_umap_and_clustering.zip": False,
}


def calculate_sha256(file_path: str) -> str:
    """
    Calculate the SHA256 checksum of a file.

    Args:
        file_path: Path to the file.

    Returns:
        The SHA256 checksum as a hexadecimal string.
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_and_unpack_files(
    folder_id: str, destination_dir: str, api_key: str, skip_large_files: bool = False
) -> None:
    """
    Download and unpack files from a Google Drive folder.

    Args:
        folder_id: The ID of the Google Drive folder.
        destination_dir: The local directory to save and unpack the files.
        api_key: The Google Drive API key.
        skip_large_files: Whether to skip downloading large files.

    Note:
        This function creates the destination directory if it doesn't exist,
        downloads ZIP files from the specified Google Drive folder, checks their
        checksums, and unpacks them.
    """
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    service = build("drive", "v3", developerKey=api_key)

    query = f"'{folder_id}' in parents and mimeType='application/zip'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get("files", [])

    if not items:
        print("No files found.")
        return

    for item in items:
        file_id = item["id"]
        file_name = item["name"]

        if skip_large_files and large_files.get(file_name, False):
            print(f"Skipping {file_name}.")
            continue

        request = service.files().get_media(fileId=file_id)
        file_path = os.path.join(destination_dir, file_name)
        print(f"Downloading {file_name} to {file_path}.")

        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f"Progress {file_name}: {int(status.progress() * 100)}%", end="\r")
            print()

        checksum = calculate_sha256(file_path)
        if checksum != checksums[file_name]:
            print(
                f"Checksum mismatch for {file_name}. Expected {checksums[file_name]}, "
                f"got {checksum}."
            )
            return
        print(f"Checksum OK for {file_name}.")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
        print(f"Unpacked {file_name}.")


def main() -> None:
    """
    Main function to handle command line arguments and initiate the download process.
    """
    arg_parser = argparse.ArgumentParser(
        description="Download pretrained models and UMAP clustering results. "
        "This script downloads two ZIP files from Google Drive:\n"
        "1. results_pretrained_models.zip: Contains pretrained neural networks.\n"
        "2. results_umap_and_clustering.zip: Contains UMAP embeddings and "
        "clustering.\n"
        "The files are downloaded and unpacked to the 'data' directory in the "
        "project root.",
        formatter_class=argparse.RawTextHelpFormatter,
        usage=(
            "\nflyvis download-pretrained [-h] [--skip_large_files]\n"
            "       or\n"
            "%(prog)s [-h] [--skip_large_files]\n"
        ),
    )
    arg_parser.add_argument(
        "--skip_large_files",
        action="store_true",
        help="Skip downloading large files. If set, only 'results_pretrained_models.zip' "
        "will be downloaded, and 'results_umap_and_clustering.zip' will be skipped.",
    )
    args, _ = arg_parser.parse_known_intermixed_args()

    folder_id = "15_8gPaVVJV6wGAspwkrdN8r-2NxMY4Kj"
    api_key = "AIzaSyDOy2_N7B5UjxKy5Xxeyd9WZfnDWzQ4-54"

    download_and_unpack_files(folder_id, str(root_dir), api_key, args.skip_large_files)


if __name__ == "__main__":
    main()
