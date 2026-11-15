from google.colab import files, runtime
import hashlib, io, os, zipfile

try:
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload only Task_2_Block_15_GPU_Final_Evaluation.zip")
    task2_evaluation_package_bytes = bytes(next(iter(uploaded.values())))
    if hashlib.sha256(task2_evaluation_package_bytes).hexdigest() != "e5f2890a6fac6dd70162d5e04e3b53157b3a8c612d4ae94436090f3831a2c5ac":
        raise ValueError("Wrong or damaged ZIP. Upload the Block 15 final evaluation package.")
    with zipfile.ZipFile(io.BytesIO(task2_evaluation_package_bytes)) as package:
        if package.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        code = package.read("block_15.py")
    if hashlib.sha256(code).hexdigest() != "47f583783062473e8847e848b3a6c2d656329d2d68048e4c402676f5d707958e":
        raise ValueError("Evaluation controller checksum mismatch.")
except BaseException:
    print("STOP - SEND THIS OUTPUT TO ME. Package loading failed.", flush=True)
    if os.environ.get("COLAB_GPU", "0") not in ("", "0"):
        runtime.unassign()
    raise

exec(compile(code, "block_15.py", "exec"), globals())
