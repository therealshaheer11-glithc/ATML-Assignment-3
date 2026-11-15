from google.colab import files
import hashlib, io, zipfile

uploaded_07 = files.upload()
if len(uploaded_07) != 1:
    raise ValueError("Upload only Task_2_Block_07_CPU_Preflight.zip")
task2_preflight_package_bytes = bytes(next(iter(uploaded_07.values())))
if hashlib.sha256(task2_preflight_package_bytes).hexdigest() != "d5d2ad21658d452673d6a549f725e2f78949b97b3890ee9d93e170d6eb50ec95":
    raise ValueError("Package checksum mismatch.")
with zipfile.ZipFile(io.BytesIO(task2_preflight_package_bytes)) as package_07:
    if package_07.testzip() is not None:
        raise ValueError("ZIP integrity check failed.")
    code_07 = package_07.read("block_07.py")
if hashlib.sha256(code_07).hexdigest() != "7bc799b420e2e0290da306a894cd512208972d976306eca827b5f54e8473b309":
    raise ValueError("Preflight code checksum mismatch.")
exec(compile(code_07, "block_07.py", "exec"), globals())
