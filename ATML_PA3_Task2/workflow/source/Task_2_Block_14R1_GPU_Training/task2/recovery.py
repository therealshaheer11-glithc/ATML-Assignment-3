"""Explicit zero-update AMP initialization migration, pending human agreement."""
import copy
from .training_state import assert_equal


def amp128_initial_state(source,identity):
    """Change only checkpoint identity and AMP initial scale at update zero.

    This never serves as an in-training overflow retry or discards moments.
    The caller must obtain agreement to this numerical recovery before use.
    """
    if source['schema']!='atml-pa3-task2-training-state-v1':raise ValueError('Unknown source state.')
    if source['stage']!=1 or source['global_updates']!=0 or source['stage_updates']!={1:0,2:0,3:0}:
        raise ValueError('AMP initialization recovery requires an untrained stage-one state.')
    if source['optimizer']['state'] or source['sampler']['caption_cursor']!=0:
        raise ValueError('Cannot reset an advanced optimizer or sampler.')
    if source['sampler']['draw_counts']!={1:0,2:0,3:0}:
        raise ValueError('Cannot reset nonzero sampler draws.')
    scaler=source['scaler']
    if scaler!={'scale':65536.0,'growth_factor':2.0,'backoff_factor':0.5,'growth_interval':2000,'_growth_tracker':0}:
        raise ValueError('Source scaler differs from the observed fresh default state.')
    result=copy.deepcopy(source)
    result['identity']=copy.deepcopy(identity)
    result['scaler']['scale']=128.0
    # Verify every unapproved field is preserved, including all RNG tensors.
    comparison=copy.deepcopy(result)
    comparison['identity']=copy.deepcopy(source['identity'])
    comparison['scaler']['scale']=65536.0
    assert_equal(source,comparison,'AMP initialization recovery preserved state')
    return result
