import merlin


def download_sample_data(data_dir):
    file_path = merlin.utils.download_file(
        repo_id="louisblankemeier/Merlin",
        filename="image1.nii.gz",
        local_dir=data_dir,
    )
    return file_path
