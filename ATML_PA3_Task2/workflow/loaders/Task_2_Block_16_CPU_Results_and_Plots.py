from google.colab import files
import hashlib, io, zipfile

try:
    uploaded = files.upload()
    if len(uploaded) != 1:
        raise ValueError("Upload only Task_2_Block_16_CPU_Results_and_Plots.zip")
    task2_results_package_bytes = bytes(next(iter(uploaded.values())))
    if hashlib.sha256(task2_results_package_bytes).hexdigest() != "5378a536e1aca6a982f57e8ddeeb7a25d8f35d8a0546e7f640f1d44439c971bd":
        raise ValueError("Wrong or damaged ZIP. Upload the Block 16 results package.")
    with zipfile.ZipFile(io.BytesIO(task2_results_package_bytes)) as package:
        if package.testzip() is not None:
            raise ValueError("ZIP integrity check failed.")
        code = package.read("block_16.py")
    if hashlib.sha256(code).hexdigest() != "3f9acf3f46993aab926edd533574ed4fd40cb3e5edeeb31fe8f4a2d4193af732":
        raise ValueError("CPU results controller checksum mismatch.")
except BaseException:
    print("STOP - SEND THIS OUTPUT TO ME. Package loading failed.", flush=True)
    raise

try:
    exec(compile(code, "block_16.py", "exec"), globals())
except BaseException:
    print("STOP - SEND THIS OUTPUT TO ME. Keep the full output for review.", flush=True)
    raise
