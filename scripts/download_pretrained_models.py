import argparse
import io
import os
import zipfile
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

checksums = {
    "results_pretrained_models.zip": (
        "04760c9103dc946692ec27014511f8fec785b5b5ddda22ec2b5bb4611fd28f5f"
    ),
    "results_umap_and_clustering.zip": (
        "372ffcb1b8af59974ac21e56ca989f72d499605f049c36fc6f460ba402ceb08c"
    ),
}

large_files = {
    "results_pretrained_models.zip": False,
    "results_umap_and_clustering.zip": True,
}


def download_and_unpack_files(
    folder_id, destination_dir, api_key, skip_large_files=False
):
    # create destination directory if it doesn't exist
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # build the service
    service = build("drive", "v3", developerKey=api_key)

    # search for files in the specified folder
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

        # download the file
        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while done is False:
                status, done = downloader.next_chunk()
                print(
                    f"Progress {file_name}: {int(status.progress() * 100)}%",
                    end="\r",
                )

        # check the file checksum
        with open(file_path, "rb") as f:
            checksum = os.popen(f"sha256sum {file_path}").read().strip().split()[0]
            if checksum not in checksums[file_name]:
                print(f"Checksum mismatch for {file_name}.")
                return
            print(f"Checksum OK for {file_name}.")

        # unzip the file
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(destination_dir)


def main():
    # handle command line arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--skip_large_files", action="store_true")
    args = arg_parser.parse_args()

    # replace with your folder ID and API key
    folder_id = "15_8gPaVVJV6wGAspwkrdN8r-2NxMY4Kj"
    destination_dir = Path(__file__).parent.parent / "data"
    api_key = "AIzaSyDOy2_N7B5UjxKy5Xxeyd9WZfnDWzQ4-54"

    download_and_unpack_files(
        folder_id, destination_dir, api_key, large_files=args.skip_large_files
    )


if __name__ == "__main__":
    main()
