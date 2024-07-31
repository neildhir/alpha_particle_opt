import os
import glob


def clean_up_vmec_rubbish() -> None:
    """
    Remove VMEC-generated files.
    """
    # List of file patterns to remove
    file_patterns = ["fort.9", "parvmecinfo.txt", "threed1.*", "wout_*", "input.*_000_*"]

    # Iterate over each pattern and remove matching files
    for pattern in file_patterns:
        for file in glob.glob(pattern):
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error removing file {file}: {e}")
