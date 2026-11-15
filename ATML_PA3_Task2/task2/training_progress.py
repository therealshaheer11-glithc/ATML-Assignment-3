"""Verified complete-update progress with preserved prior active-pointer receipts."""
import hashlib
import json
import uuid
from pathlib import Path
from .images import encoded
from .session_io import bounded_io,save_bytes


def validate_counters(state):
    counts=state['stage_updates']
    if state['global_updates']!=sum(counts.values()):raise ValueError('Successful-update totals disagree.')
    if state['sampler']['caption_cursor']!=16*counts[1]:raise ValueError('Caption sampler and successful updates disagree.')
    if state['sampler']['draw_counts']!={stage:8*count for stage,count in counts.items()}:
        raise ValueError('Complete-pair draws and successful updates disagree.')
    if state['stage'] not in (1,2,3):raise ValueError('Invalid stage.')


def save_artifact(folder,stem,value):
    raw=encoded(value);digest=hashlib.sha256(raw).hexdigest()
    path=Path(folder)/(stem+'_'+digest[:16]+'.json')
    with bounded_io(90,'Verified training result save'):
        save_bytes(path,raw)
        observed=path.read_bytes()
        if hashlib.sha256(observed).hexdigest()!=digest:raise IOError('Result readback differs.')
    return {'path':str(path),'sha256':digest,'bytes':len(raw),'readback_verified':True}


def read_artifact(receipt):
    with bounded_io(90,'Verified result read'):
        raw=Path(receipt['path']).read_bytes()
        if len(raw)!=receipt['bytes'] or hashlib.sha256(raw).hexdigest()!=receipt['sha256']:
            raise IOError('Saved training artifact differs from its receipt.')
    return json.loads(raw)


def publish_progress(folder,pointer_path,progress,expected_pointer):
    """Persist new immutable progress, then atomically advance a checked pointer."""
    folder,pointer_path=Path(folder),Path(pointer_path)
    receipt=save_artifact(folder/'progress','update_'+f"{progress['global_updates']:06d}",progress)
    pointer={'schema':'atml-pa3-task2-active-training-v1','dataset_identity':progress['dataset_identity'],
        'initial_manifest_sha256':progress['initial_manifest_sha256'],
        'execution_recipe_sha256':progress['execution_recipe_sha256'],'global_updates':progress['global_updates'],
        'progress':receipt}
    raw=encoded(pointer)
    with bounded_io(90,'Atomic verified training pointer publication'):
        current=pointer_path.read_bytes() if pointer_path.exists() else None
        if current!=expected_pointer:raise RuntimeError('Active training progress changed; refusing a stale publication.')
        if current is not None:
            digest=hashlib.sha256(current).hexdigest()
            save_bytes(folder/'pointer_history'/('active_'+digest[:16]+'.json'),current)
        pointer_path.parent.mkdir(parents=True,exist_ok=True)
        temporary=pointer_path.with_name(pointer_path.name+'.part_'+uuid.uuid4().hex)
        try:
            temporary.write_bytes(raw)
            if temporary.read_bytes()!=raw:raise IOError('Pointer temporary readback differs.')
            temporary.replace(pointer_path)
            if pointer_path.read_bytes()!=raw:raise IOError('Active pointer readback differs.')
        finally:temporary.unlink(missing_ok=True)
    return receipt,raw
