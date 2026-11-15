from google.colab import files, runtime
import hashlib, io, os, zipfile

try:
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload only Task_2_Block_14R1_GPU_Training.zip")
    task2_initialization_package_bytes = bytes(next(iter(uploaded.values())))
    if hashlib.sha256(task2_initialization_package_bytes).hexdigest() != "144fd153c443dc57e30d1cf4e7236ef3b91995c610d90a583dd09c62530165ed":
        raise ValueError("Wrong ZIP. Upload the new 14R1 training package.")
    with zipfile.ZipFile(io.BytesIO(task2_initialization_package_bytes)) as package:
        if package.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        code = package.read("block_14.py")
    if hashlib.sha256(code).hexdigest() != "b40c078d7aa3ef33cc7ab01082d12a69ea953390e321074a0bc9fc0af7706c52":
        raise ValueError("Controller checksum mismatch.")
except BaseException:
    print("STOP - SEND THIS OUTPUT TO ME. Package loading failed; do not continue.", flush=True)
    if os.environ.get("COLAB_GPU", "0") not in ("", "0"):
        runtime.unassign()
    raise

exec(compile(code, "block_14.py", "exec"), globals())
