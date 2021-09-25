import argparse
import os
import shutil
import subprocess
import sys

from google.colab import drive


def setup_colab_environment(data_path: str, branch: str = "main") -> None:
    """
    Prepares Google Colab environment for running the pipeline.

    Args:
        data_path: Path of zipped datasets in Google drive.
        branch: Git branch to be checked out.

    Returns:
        None
    """

    git_repo_dir = "/content/BirdSongIdentification"

    try:
        subprocess.run(["git", "checkout", branch], cwd=git_repo_dir, check=True)
        subprocess.run(["git", "pull"], cwd=git_repo_dir, check=True)
        sys.path.append(git_repo_dir)
    except subprocess.CalledProcessError:
        raise NameError(f"Could not check out branch {branch} of Git repo")

    try:
        subprocess.run(["pip3", "install", "-r", "requirements-colab.txt"], cwd=git_repo_dir, check=True)
    except subprocess.CalledProcessError:
        raise NameError("Could not install requirements")

    print("\n")
    print("Load training data from Google drive...")
    print("\n")

    drive.mount('/content/drive', force_remount=True)
    colab_data_dir = "/content/data"
    for dataset in ["train", "test", "val"]:
        if not os.path.exists(os.path.join(colab_data_dir, f"{dataset}.zip")):
            os.makedirs(colab_data_dir, exist_ok=True)
            shutil.copy(os.path.join(data_path, f"{dataset}.zip"), colab_data_dir)

        if not os.path.exists(os.path.join(colab_data_dir, dataset)):
            try:
                subprocess.run(["unzip", "-q", os.path.join(colab_data_dir, f"{dataset}.zip")],
                               cwd=colab_data_dir, check=True)
            except subprocess.CalledProcessError:
                raise NameError(f"Could unzip {dataset} data set")

    # force restart of runtime
    os.kill(os.getpid(), 9)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Setup Google Colab environment for model training.")
    parser.add_argument("data_path", help="Path of the training data in Google drive")
    parser.add_argument("--branch", required=False, help="Path of the training data in Google drive")

    args = parser.parse_args()

    if not args.branch:
        args.branch = "main"

    setup_colab_environment(args.data_path, args.branch)
