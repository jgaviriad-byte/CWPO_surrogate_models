from pathlib import Path
import re

def create_directory(path):
    """
    Create a directory if it does not exist.
    """

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )

from pathlib import Path


def get_results_path(*folders):
    """
    Build a path inside the results directory.
    """

    path = Path("results")

    for folder in folders:

        path = path / folder

    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    return path

import re

def sanitize_filename(name):
    """
    Convert any text into a valid filename.
    """

    return re.sub(
        r'[<>:"/\\|?*\[\]\(\)]',
        "",
        name
    ).replace(" ", "_")