"""Teacher-forced development monitoring without generations or training RNG drift."""
import time
import torch
import torch.nn.functional as F
from .modeling import target_logits,set_stage
from .training_state import capture_rng,restore_rng,assert_equal


def example_losses(model,connector,records,features,content_only):
    logits,targets,ids,batch=target_logits(model,connector,records,features,content_only=content_only)
    token_losses=F.cross_entropy(logits.float(),targets,reduction='none')
    result=[]
    for i in range(len(records)):
        selected=token_losses[ids==i]
        if selected.numel()==0 or not torch.isfinite(selected).all():
            raise FloatingPointError('Invalid development reference NLL.')
        result.append({'loss':float(selected.mean()),'tokens':selected.numel()})
    return result


def dev_rows(data,kind):
    pairs=data['grounding' if kind=='grounding' else 'natural']['dev']
    rows=[]
    for pair in pairs:
        for side,example in enumerate(pair['examples']):
            rows.append({'suite':kind,'pair_id':pair['pair_id'],'side':side,
                'source_split':pair['source_split'],'image_id':example['image_id'],
                'partner_image_id':pair['examples'][1-side]['image_id'],
                'user_text':pair['user_text'],'target':example['target'],'canonical':example['canonical'],
                'question_type':pair.get('question_type',pair.get('answer_type')),
                **({'human_answers':example['human_answers']} if 'human_answers' in example else {})})
    return rows


def evaluate_development(session,data,check_deadline=lambda:None,progress=lambda message:None):
    before=session.snapshot()
    saved_rng=capture_rng(session.device)
    stage=session.stage
    result={'schema':'atml-pa3-task2-development-v1','global_update':session.global_updates,
        'conditions':['correct','complementary','no_image'],'generation_performed':False,
        'answer_loss':'Content-only token mean per example, then equal example mean; EOT excluded',
        'caption_loss':'Content plus EOT token mean per example, then equal example mean',
        'caption':{},'suites':{}}
    started=time.monotonic()
    try:
        set_stage(session.model,session.connector,stage,training=False)
        with torch.no_grad():
            captions=data['captions']['dev']
            caption_scores=[]
            for start in range(0,len(captions),8):
                check_deadline();progress('caption dev '+str(start)+'/'+str(len(captions)))
                rows=captions[start:start+8]
                records=[session.tables['by_text'][('Describe this image.',row['caption'])] for row in rows]
                features=session.get_features([('val2014',row['image_id']) for row in rows])
                with torch.autocast(session.device.type,dtype=torch.float16):
                    losses=example_losses(session.model,session.connector,records,features,False)
                caption_scores.extend({'caption_id':row['caption_id'],'image_id':row['image_id'],
                    'target':row['caption'],**score} for row,score in zip(rows,losses))
            result['caption']={'examples':len(caption_scores),'correct_image_CE':sum(row['loss'] for row in caption_scores)/len(caption_scores),
                'per_example':caption_scores}
            for kind in ('grounding','natural'):
                rows=dev_rows(data,kind)
                conditions={}
                for condition in result['conditions']:
                    scores=[]
                    for start in range(0,len(rows),8):
                        check_deadline();progress(kind+' '+condition+' '+str(start)+'/'+str(len(rows)))
                        batch=rows[start:start+8]
                        records=[session.tables['by_text'][(row['user_text'],row['target'])] for row in batch]
                        keys=[(row['source_split'],row['partner_image_id'] if condition=='complementary' else row['image_id']) for row in batch]
                        features=None if condition=='no_image' else session.get_features(keys)
                        with torch.autocast(session.device.type,dtype=torch.float16):
                            losses=example_losses(session.model,session.connector,records,features,True)
                        scores.extend({**row,'condition':condition,'input_image_id':None if condition=='no_image' else key[1],
                            'reference_tokens':score['tokens'],'reference_NLL':score['loss']}
                            for row,key,score in zip(batch,keys,losses))
                    types=sorted({row['question_type'] for row in scores})
                    conditions[condition]={'examples':len(scores),'mean_reference_NLL':sum(row['reference_NLL'] for row in scores)/len(scores),
                        'by_type':{kind:{'examples':sum(row['question_type']==kind for row in scores),
                            'mean_reference_NLL':sum(row['reference_NLL'] for row in scores if row['question_type']==kind)/sum(row['question_type']==kind for row in scores)} for kind in types},
                        'per_example':scores}
                correct=conditions['correct']['per_example'];swapped=conditions['complementary']['per_example']
                assert [(r['pair_id'],r['side'],r['target']) for r in correct]==[(r['pair_id'],r['side'],r['target']) for r in swapped]
                result['suites'][kind]={'conditions':conditions,
                    'swap_penalty_NLL':sum(b['reference_NLL']-a['reference_NLL'] for a,b in zip(correct,swapped))/len(correct)}
    finally:
        set_stage(session.model,session.connector,stage,training=True)
        restore_rng(saved_rng,session.device)
        assert_equal(before,session.snapshot(),'training state after development evaluation')
    result['elapsed_seconds']=round(time.monotonic()-started,3)
    result['training_state_and_RNG_restored_exactly']=True
    return result
