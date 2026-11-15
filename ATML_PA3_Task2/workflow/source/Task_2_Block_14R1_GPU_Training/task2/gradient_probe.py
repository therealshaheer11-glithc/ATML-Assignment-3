"""Read-only model-state gradient diagnostics, never optimizer updates."""
import traceback
import torch
from .modeling import training_loss,assert_precision
from .training_state import assert_equal


def tensor_summary(value):
    if value is None:return {'present':False}
    value=value.detach()
    finite=torch.isfinite(value)
    count=int(value.numel())
    nfinite=int(finite.sum().item())
    return {'present':True,'dtype':str(value.dtype),'shape':list(value.shape),'elements':count,
        'finite_elements':nfinite,'nan_elements':int(torch.isnan(value).sum().item()),
        'positive_inf_elements':int(torch.isposinf(value).sum().item()),
        'negative_inf_elements':int(torch.isneginf(value).sum().item()),
        'nonzero_finite_elements':int(torch.count_nonzero(value[finite]).item()),
        'max_abs_finite':float(value[finite].abs().max().item()) if nfinite else None,
        'all_finite':nfinite==count}


def probe_gradients(session,canonical,loss_scale):
    """Same first sixteen examples/RNG for each diagnostic scale, then rollback.

    The scaler is used for backward/unscale only. No step/update/clip, no
    automatic recovery, scale selection or passed-training-state publication.
    """
    session.restore(canonical)
    session.select_stage(1)
    session.scaler=torch.amp.GradScaler(session.device.type,init_scale=float(loss_scale))
    assert_precision(session.model,session.connector,1)
    result={'diagnostic_loss_scale':float(loss_scale),'stage':1,'optimizer_steps':0,
        'scaler_step_or_update_called':False,'gradient_clipping_applied':False,
        'micro_losses':[],'visual_boundary_gradients':[],'exception':None}
    handles=[]
    def boundary(module,inputs,output):
        if output.requires_grad:
            def observe(gradient):
                result['visual_boundary_gradients'].append(tensor_summary(gradient))
            output.register_hook(observe)
    handles.append(session.connector.register_forward_hook(boundary))
    try:
        pairs=[session.sampler.draw(1) for _ in range(8)]
        examples=[row for pair in pairs for row in pair['examples']]
        assert len(examples)==16 and all(row['source_split']=='train2014' for row in examples)
        result['examples']=examples
        for start in (0,8):
            rows=examples[start:start+8]
            records=[session.tables['by_text'][(row['user_text'],row['target'])] for row in rows]
            features=session.get_features([(row['source_split'],row['image_id']) for row in rows])
            with torch.autocast(session.device.type,dtype=torch.float16):
                loss=training_loss(session.model,session.connector,records,features)
            result['micro_losses'].append(float(loss.detach()))
            session.scaler.scale(loss/2).backward()
            del features,loss
        session.scaler.unscale_(session.optimizer)
        result['connector_gradients']={name:tensor_summary(p.grad) for name,p in session.connector.named_parameters()}
        result['frozen_parameters_with_grad']=[name for name,p in session.model.named_parameters() if p.grad is not None]
        summaries=list(result['connector_gradients'].values())
        result['missing_gradient_parameters']=[name for name,row in result['connector_gradients'].items() if not row['present']]
        result['wrong_gradient_dtype_parameters']=[name for name,row in result['connector_gradients'].items() if row['present'] and row['dtype']!='torch.float32']
        result['nonfinite_gradient_parameters']=[name for name,row in result['connector_gradients'].items() if row['present'] and not row['all_finite']]
        result['connector_gradients_finite_FP32']=all(row['present'] and row['dtype']=='torch.float32' and row['all_finite'] for row in summaries)
        result['connector_gradient_nonzero']=any(row['present'] and row['nonzero_finite_elements']>0 for row in summaries)
    except Exception:
        result['exception']=traceback.format_exc()
    finally:
        for handle in handles:handle.remove()
        session.optimizer.zero_grad(set_to_none=True)
        session.restore(canonical)
        assert_equal(canonical,session.snapshot(),'canonical state after gradient-only diagnostic')
        result['canonical_state_restored_exactly']=True
    return result
