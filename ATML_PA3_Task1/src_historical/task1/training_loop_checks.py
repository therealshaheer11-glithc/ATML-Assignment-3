"""Exercise actual bounded-loop checkpoint boundaries on a small CPU model."""
import contextlib
import io
import json
import time
from pathlib import Path
from unittest.mock import patch

from baseline_runtime import atomic
from official_training import seed_training
from training_runtime_checks import fresh, rows
from training_checkpoints import restore_checkpoint
from training_loop import run_updates


class FixtureData:
    def batch(self, update):
        return [{**row,'source_index':(update-1)*32+i} for i,row in enumerate(rows())]


def checks(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    state=directory/'state.json';now=time.time()
    atomic(state,{'deadline_wall':now+3600,'progress_wall':now,'status':'running'})
    config={'root':str(directory/'Drive_fixture'),'job':str(directory),'state':str(state),
        'soft_wall':now+3600,'training_contract':{'method':'full','scope':'CPU_FIXTURE_ONLY'}}
    seed_training();bundle=fresh();seed_training()
    with contextlib.redirect_stdout(io.StringIO()):
        result=run_updates(config,*bundle,FixtureData(),2e-5,0)
    assert result['status']=='TRAINING_MANDATORY_SAVED' and result['successful_updates']==312
    checkpoint=Path(config['root'])/result['checkpoint']
    recovered=fresh()
    header=restore_checkpoint(checkpoint,config['training_contract'],recovered[0],recovered[1],recovered[2],recovered[3])
    assert len(header['history'])==312 and header['examples_seen']==9984
    assert [i for r in header['history'] for i in r['source_ids']]==list(range(9984))
    assert not list(checkpoint.parent.glob('rolling_*/COMPLETE.json'))
    # Stop after precisely one more completed optimizer update, then resume at 313.
    calls=[0]
    def clock():calls[0]+=1;return config['soft_wall']-1 if calls[0]==1 else config['soft_wall']+1
    with contextlib.redirect_stdout(io.StringIO()):
        partial=run_updates(config,*recovered,FixtureData(),2e-5,0,restored=header,clock=clock)
    assert partial['status']=='TRAINING_PARTIAL_SAVED' and partial['successful_updates']==313
    # A save failure after a committed update must keep POST-update RNG, not rewind it.
    import training_loop
    from official_training import rng_state
    captured=[];actual=training_loop.save_checkpoint
    def fail_once(*args,**kwargs):
        if not captured:
            captured.append(rng_state(False));raise IOError('CPU_FIXTURE_SAVE_FAILURE')
        assert captured[0]==rng_state(False)
        return actual(*args,**kwargs)
    next_bundle=fresh()
    second=restore_checkpoint(Path(config['root'])/partial['checkpoint'],config['training_contract'],
        next_bundle[0],next_bundle[1],next_bundle[2],next_bundle[3])
    calls[0]=0
    with patch.object(training_loop,'save_checkpoint',fail_once),contextlib.redirect_stdout(io.StringIO()):
        try:run_updates(config,*next_bundle,FixtureData(),2e-5,0,restored=second,clock=clock)
        except IOError as e:assert str(e)=='CPU_FIXTURE_SAVE_FAILURE'
        else:raise AssertionError('Save failure swallowed')
    return {'312_actual_toy_updates_stop_at_first_mandatory_boundary':True,
        'all_9984_source_order_fixture_IDs_preserved':True,'mandatory_checkpoint_reloads_at_312':True,
        'soft_limit_stops_at_complete_update_313':True,
        'checkpoint_failure_after_commit_keeps_post_update_RNG':True,
        'scope':'CPU fixtures only; no original-model or official training updates'}
