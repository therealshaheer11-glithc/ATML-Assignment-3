from google.colab import files, runtime
import hashlib, io, os, zipfile

try:
    uploaded_11r = files.upload()
    if len(uploaded_11r) != 1:
        raise ValueError("Upload only Task_2_Block_11R1_GPU_Feature_Cache.zip")

    task2_features_package_bytes = bytes(next(iter(uploaded_11r.values())))
    if hashlib.sha256(task2_features_package_bytes).hexdigest() != "fca343eded123a3003a63cfeca60dbb090e6ba29a0cba5a82a769f92529ef536":
        raise ValueError("Choose the corrected 11R1 ZIP.")

    with zipfile.ZipFile(io.BytesIO(task2_features_package_bytes)) as package_11r:
        if package_11r.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        code_11r = package_11r.read("block_11.py")

    if hashlib.sha256(code_11r).hexdigest() != "7a769c130487d02620fea3ca7595fbca34cda17ae17776338ecb71d10f64df10":
        raise ValueError("Cache-controller checksum mismatch.")

except BaseException:
    if os.environ.get("COLAB_GPU", "0") not in ("", "0"):
        runtime.unassign()
    raise

exec(compile(code_11r, "block_11.py", "exec"), globals())
