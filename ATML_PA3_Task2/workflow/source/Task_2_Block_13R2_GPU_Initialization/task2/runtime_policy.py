"""Process-local optional-backend policy for the approved unquantized recipe."""
import importlib.metadata
import importlib.util
import sys


def isolate_unused_torchao():
    """Exclude an unused optional quantization backend before framework imports.

    PEFT 0.21.2 probes torchao for ordinary Linear layers too and rejects an
    installed older version. This worker never uses quantized weights. Python's
    documented None entry in sys.modules prevents this optional import only in
    the isolated worker. Installed packages and the parent process are intact.
    """
    if any(name in sys.modules for name in ('torch','transformers','peft','accelerate')):
        raise RuntimeError('Apply the optional-backend policy before framework imports in a fresh worker.')
    if any(name=='torchao' or name.startswith('torchao.') for name in sys.modules):
        raise RuntimeError('torchao was already touched. Start a fresh worker before applying this policy.')
    try:version=importlib.metadata.version('torchao')
    except importlib.metadata.PackageNotFoundError:version=None
    discoverable=importlib.util.find_spec('torchao') is not None
    sys.modules['torchao']=None
    if importlib.util.find_spec('torchao') is not None:
        raise RuntimeError('Failed to exclude the unused optional torchao backend.')
    return {'backend':'torchao','installed_version':version,'originally_discoverable':discoverable,
        'scope':'Isolated worker only; no package uninstall, upgrade or library-function patch',
        'worker_import_disabled':True,'quantized_weights_permitted':False}
