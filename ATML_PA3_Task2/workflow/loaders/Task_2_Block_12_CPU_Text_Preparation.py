from google.colab import files, runtime
import hashlib, io, os, zipfile

try:
    uploaded_12 = files.upload()
    if len(uploaded_12) != 1:
        raise ValueError("Upload only Task_2_Block_12_CPU_Text_Preparation.zip")
    task2_text_package_bytes = bytes(next(iter(uploaded_12.values())))
    if hashlib.sha256(task2_text_package_bytes).hexdigest() != "fe3b827e8ad6367c661a5846711eef84eb31dea472484bec4dd3d0e717da463b":
        raise ValueError("Package checksum mismatch.")
    with zipfile.ZipFile(io.BytesIO(task2_text_package_bytes)) as package_12:
        if package_12.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        code_12 = package_12.read("block_12.py")
    if hashlib.sha256(code_12).hexdigest() != "7e28b5ac6035ec7c8363f2b931b051b4874e9187c4c95acebdfbc81df78ffba4":
        raise ValueError("CPU controller checksum mismatch.")
except BaseException:
    if os.environ.get("COLAB_GPU", "0") not in ("", "0"):
        runtime.unassign()
    raise

exec(compile(code_12, "block_12.py", "exec"), globals())
