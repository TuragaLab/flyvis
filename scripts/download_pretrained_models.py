import argparse
import io
import os
import zipfile
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import hashlib  # Import hashlib for checksum computation

checksums = {
    "results_pretrained_models.zip": (
        "49134ff89dd396d8a58686cf565e55adb59416fc1c602042f680e2cc0642a440"
    ),
    "results_umap_and_clustering.zip": (
        "372ffcb1b8af59974ac21e56ca989f72d499605f049c36fc6f460ba402ceb08c"
    ),
}

large_files = {
    "results_pretrained_models.zip": False,
    "results_umap_and_clustering.zip": True,
}


def calculate_sha256(file_path):
    """Calculate the SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_and_unpack_files(
    folder_id, destination_dir, api_key, skip_large_files=False
):
    # Create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Build the service
    service = build("drive", "v3", developerKey=api_key)

    # Search for files in the specified folder
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

        # Download the file
        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(
                    f"Progress {file_name}: {int(status.progress() * 100)}%",
                    end="\r",
                )
            print()  # Move to the next line after download completes

        # Check the file checksum using hashlib
        checksum = calculate_sha256(file_path)
        if checksum != checksums[file_name]:
            print(
                f"Checksum mismatch for {file_name}. Expected {checksums[file_name]}, got {checksum}."
            )
            return
        print(f"Checksum OK for {file_name}.")

        # Unzip the file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(destination_dir)
        print(f"Unpacked {file_name}.")


def main():
    # Handle command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--skip_large_files", action="store_true")
    args = arg_parser.parse_args()

    # Replace with your folder ID and API key
    folder_id = "15_8gPaVVJV6wGAspwkrdN8r-2NxMY4Kj"
    destination_dir = Path(__file__).parent.parent / "data"
    api_key = "AIzaSyDOy2_N7B5UjxKy5Xxeyd9WZfnDWzQ4-54"

    download_and_unpack_files(
        folder_id, destination_dir, api_key, args.skip_large_files
    )


if __name__ == "__main__":
    main()
