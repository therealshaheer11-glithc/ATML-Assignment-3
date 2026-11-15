from google.colab import files
import hashlib, io, zipfile

uploaded_09 = files.upload()
if len(uploaded_09) != 1:
    raise ValueError("Upload only Task_2_Block_09_Cache_Input_Packing.zip")
task2_staging_package_bytes = bytes(next(iter(uploaded_09.values())))
if hashlib.sha256(task2_staging_package_bytes).hexdigest() != "571c75773e78e3cec1e3a0c32ff8f8e7a30a3cd3dd63dbb745e56bd667155027":
    raise ValueError("Package checksum mismatch.")
with zipfile.ZipFile(io.BytesIO(task2_staging_package_bytes)) as package_09:
    if package_09.testzip() is not None:
        raise ValueError("ZIP integrity check failed.")
    code_09 = package_09.read("block_09.py")
if hashlib.sha256(code_09).hexdigest() != "282ca2935b66f3741ea006a085b83f29c3b2d42ab625f50605d4c9c3be018c0b":
    raise ValueError("Packing code checksum mismatch.")
exec(compile(code_09, "block_09.py", "exec"), globals())
