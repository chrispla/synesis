"""General utility functions."""

import os

import requests

from tqdm import tqdm


def deep_update(d, u):
    """
    Recursively update a dict with another (that potentially
    doesn't have all keys)

    Args:
        d: The dict to update.
        u: The dict to update from.

    Returns:
        The updated dict.
    """
    if u is None or u == {}:
        return d
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def download_github_dir(owner, repo, path, save_dir):
    """
    Download a directory from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        path (str): The path to the directory in the repository.
        save_dir (str): The directory to save the downloaded files.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    response = requests.get(api_url)
    if response.status_code == 200:
        contents = response.json()

        print(contents)

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        length = len(contents)
        pbar = tqdm(contents, total=length)

        for item in pbar:
            # Update the progress bar description
            pbar.set_description(f"Downloading {item['name']}")
            if item["type"] == "file":
                download_url = item["download_url"]
                file_name = item["name"]
                file_path = os.path.join(save_dir, file_name)
                file_content = requests.get(download_url).content
                with open(file_path, "wb") as file:
                    file.write(file_content)
            elif item["type"] == "dir":
                download_github_dir(
                    owner, repo, item["path"], os.path.join(save_dir, item["name"])
                )
    else:
        print(
            f"Failed to fetch directory contents. Status code: {response.status_code}"
        )


def download_github_file(owner, repo, file_path, save_dir):
    """
    Download a file from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        file_path (str): The path to the file in the repository.
        save_dir (str): The directory to save the downloaded file.
    """
    api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}"
    response = requests.get(api_url)

    if response.status_code == 200:
        file_info = response.json()
        if file_info["type"] == "file":
            download_url = file_info["download_url"]
            file_name = os.path.basename(file_path)
            save_path = os.path.join(save_dir, file_name)

            os.makedirs(save_dir, exist_ok=True)

            file_content = requests.get(download_url).content
            with open(save_path, "wb") as file:
                file.write(file_content)
            print(f"Downloaded: {save_path}")
        else:
            print(f"The specified path is not a file: {file_path}")
    else:
        print(f"Failed to fetch file. Status code: {response.status_code}")
