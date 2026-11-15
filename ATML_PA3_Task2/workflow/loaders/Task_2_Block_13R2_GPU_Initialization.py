from google.colab import files, runtime
import hashlib, io, os, zipfile

try:
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload only Task_2_Block_13R2_GPU_Initialization.zip")
    task2_initialization_package_bytes = bytes(next(iter(uploaded.values())))
    if hashlib.sha256(task2_initialization_package_bytes).hexdigest() != "8fc0a3270bb36adec661872d34a5af5703b403b32c95896851e8339f04827db3":
        raise ValueError("Wrong ZIP. Upload the approved 13R2 AMP128 package.")
    with zipfile.ZipFile(io.BytesIO(task2_initialization_package_bytes)) as package:
        if package.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        code = package.read("block_13.py")
    if hashlib.sha256(code).hexdigest() != "5148e43452c0cfc55edc698980cd4be3b6c46860c46a22aed81a6029bc4ad02b":
        raise ValueError("Controller checksum mismatch.")
except BaseException:
    if os.environ.get("COLAB_GPU", "0") not in ("", "0"):
        runtime.unassign()
    raise

exec(compile(code, "block_13.py", "exec"), globals())
