from google.colab import files
import hashlib, io, zipfile

uploaded_08 = files.upload()
if len(uploaded_08) != 1:
    raise ValueError("Upload only Task_2_Block_08_Image_Preparation.zip")
task2_image_package_bytes = bytes(next(iter(uploaded_08.values())))
if hashlib.sha256(task2_image_package_bytes).hexdigest() != "3882b1174fe015da1dc4b9de7d97abc07038a120ec49d8d21053c865bb109160":
    raise ValueError("Package checksum mismatch.")
with zipfile.ZipFile(io.BytesIO(task2_image_package_bytes)) as package_08:
    if package_08.testzip() is not None:
        raise ValueError("ZIP integrity check failed.")
    code_08 = package_08.read("block_08.py")
if hashlib.sha256(code_08).hexdigest() != "9b036fe962d183dc204c8991b2b77f90495d5c8bc614716287ca0f773b902f60":
    raise ValueError("Image-preparation code checksum mismatch.")
exec(compile(code_08, "block_08.py", "exec"), globals())
