import tarfile

tar_path = "images.tar"
output_folder = "extracted data"

with tarfile.open(tar_path, "r") as tar:
    tar.extractall(output_folder)

print("Done extracting!")