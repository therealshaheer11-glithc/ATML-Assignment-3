from google.colab import files
import hashlib, io, zipfile

uploaded_10 = files.upload()
if len(uploaded_10) != 1:
    raise ValueError("Upload only Task_2_Block_10_Model_Preparation.zip")
task2_models_package_bytes = bytes(next(iter(uploaded_10.values())))
if hashlib.sha256(task2_models_package_bytes).hexdigest() != "41ceb7a820130a0d886d8391b46bdb7ddef1e537ca05256ae28b0793127d5288":
    raise ValueError("Package checksum mismatch.")
with zipfile.ZipFile(io.BytesIO(task2_models_package_bytes)) as package_10:
    if package_10.testzip() is not None:
        raise ValueError("ZIP integrity check failed.")
    code_10 = package_10.read("block_10.py")
if hashlib.sha256(code_10).hexdigest() != "d662bef8b31e070532dbfc3df3afd6ae9f206a973e64fbae21cd9fcea6385187":
    raise ValueError("Model-preparation code checksum mismatch.")
exec(compile(code_10, "block_10.py", "exec"), globals())
