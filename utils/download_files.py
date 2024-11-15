import os
import requests


def download_file(url, save_dir, filename=None):
    """
    Downloads a file from the given URL and saves it to the specified directory.

    Args:
        url (str): URL of the file to download.
        save_dir (str): Directory where the file will be saved.
        filename (str, optional): Custom name for the downloaded file. Defaults to None, keeps original name.

    Returns:
        str: Full path of the downloaded file.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Extract the file name from the URL if not provided
    if not filename:
        filename = url.split("/")[-1]

    file_path = os.path.join(save_dir, filename)

    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Ensure we handle bad responses

    # Save the file to the specified directory
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    return file_path
