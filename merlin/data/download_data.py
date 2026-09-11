from merlin.utils import download_file


def download_sample_data(data_dir):
    print("Downloading sample data to {}".format(data_dir))
    file_path = download_file(
        repo_id="stanfordmimi/Merlin",
        filename="image1.nii.gz",
        local_dir=data_dir,
    )
    return file_path
