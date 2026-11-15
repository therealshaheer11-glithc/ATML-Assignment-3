"""Prepare text tables from the frozen selections without reselection or normalization."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from .text import prepare_text,native_prompt,CAPTION_USER_TEXT


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()


def jsonl(rows):
    return ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for row in rows).encode()


def build_text_data(data,tokenizer):
    templates,examples,cache={},[],{}
    caption_layouts={}
    counts=Counter()
    leading=trailing=0
    def layout(user_text,target,limit,question=None):
        key=(user_text,target,limit,question)
        if key not in cache:
            row=prepare_text(tokenizer,user_text,target,limit,question=question)
            row['target_limit']=limit
            row['question']=question
            digest=hashlib.sha256(encoded(row)).hexdigest()
            assert digest not in templates
            row['layout_id']=digest
            templates[digest]=row
            cache[key]=digest
        return cache[key]
    for split,rows in data['captions'].items():
        for row in rows:
            identity=layout(CAPTION_USER_TEXT,row['caption'],64)
            assert templates[identity]['target_content_tokens']==row['content_tokens']
            example={'record_id':f'caption/{split}/{row["caption_id"]}','kind':'caption','split':split,
                'caption_id':row['caption_id'],'image_id':row['image_id'],
                'source_split':'train2014' if split=='train' else 'val2014','layout_id':identity}
            examples.append(example)
            if split=='train':caption_layouts[row['caption_id']]=(row,identity)
            counts['caption/'+split]+=1
            leading+=row['caption'].startswith('\n')
            trailing+=row['caption'].endswith('\n')
    for kind,groups in (('natural',data['natural']),('grounding',data['grounding'])):
        for split,pairs in groups.items():
            for pair in pairs:
                for side,source in enumerate(pair['examples']):
                    identity=layout(pair['user_text'],source['target'],16,question=pair['question'])
                    if 'target_content_tokens' in source:
                        assert templates[identity]['target_content_tokens']==source['target_content_tokens']
                    row={'record_id':f'{kind}/{split}/{pair["pair_id"]}/{side}','kind':kind,'split':split,
                        'pair_id':pair['pair_id'],'side':side,'source_split':pair['source_split'],
                        'image_id':source['image_id'],'question_id':source['question_id'],'layout_id':identity,
                        'canonical':source['canonical']}
                    if 'human_answers' in source:
                        row.update({'human_answers':source['human_answers'],'target_answer_id':source['target_answer_id'],
                            'answer_type':pair['answer_type']})
                    if 'category_id' in pair:
                        row.update({'category_id':pair['category_id'],'question_type':pair['question_type']})
                    examples.append(row)
                    counts[kind+'/'+split]+=1
    for question in data['question_bank']:
        targets=('yes','no') if question['question_type']=='presence' else ('1','2','3','4')
        for target in targets:
            identity=layout(question['user_text'],target,16,question=question['question'])
            examples.append({'record_id':f'ground_bank/{question["category_id"]}/{question["question_type"]}/{target}',
                'kind':'ground_bank','split':'template_only','category_id':question['category_id'],
                'question_type':question['question_type'],'target':target,'layout_id':identity,
                'available_in_training_pool':question['available_in_training_pool']})
            counts['ground_bank/template_only']+=1
    assert len(examples)==len({row['record_id'] for row in examples})==22396
    assert sum(value for key,value in counts.items() if not key.startswith('ground_bank/'))==21916
    assert all(row['layout_id'] in templates for row in examples)
    calibration={'schema':'atml-pa3-task2-prepared-calibration-v1','decision_id':'A005','seed':6304,
        'image_ids':data['calibration']['image_ids'],'caption_ids':data['calibration']['caption_ids'],
        'caption_content':[],'scale_computed':False,'evaluation_inputs_used':False}
    for caption_id in calibration['caption_ids']:
        source,identity=caption_layouts[caption_id]
        record=templates[identity]
        calibration['caption_content'].append({'caption_id':caption_id,'image_id':source['image_id'],
            'original_caption':source['caption'],'content_ids':record['target_ids']})
    assert len(calibration['image_ids'])==len(calibration['caption_content'])==256
    protocol={'schema':'atml-pa3-task2-text-contract-v1','dataset_identity':data['identity'],
        'qwen_revision':'c1899de289a04d12100db370d81485cdf75e47ca',
        'chat_template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'enable_thinking':False,'system_message':None,'custom_image_markers':False,
        'caption_user_text':CAPTION_USER_TEXT,'vqa_user_text':'Answer with a short phrase.\n{original_question}',
        'native_empty_thinking_tags_retained':True,'visual_tokens':196,'visual_placement':'After the native user header, before user text',
        'original_question_and_target_preserved':True,'target_tokenization':'Verbatim standalone content tokens as a continuation of the native generation prefix',
        'training_supervision':'Original target content plus native EOT; all context, visual and padding positions ignored',
        'training_loss':'Mean token CE within each example, then equal-weight batch mean',
        'answer_nll':'Original content tokens only, excluding EOT','caption_dev_ce':'Original content plus EOT',
        'training_padding':'right','generation_padding':'left','position_ids':'Attention-mask cumulative positions, masked padding positions set to zero',
        'no_image':'Same native text with the 196 visual positions omitted',
        'pad_token_id':tokenizer.pad_token_id,'eot_token_id':tokenizer.eos_token_id,
        'caption_content_cap':64,'answer_content_cap':16,'question_and_user_content_cap':80}
    maxima={'caption_target':max(row['target_content_tokens'] for row in templates.values() if row['target_limit']==64),
        'answer_target':max(row['target_content_tokens'] for row in templates.values() if row['target_limit']==16),
        'native_prompt':max(len(row['prompt_ids']) for row in templates.values()),
        'training_sequence_with_visuals':max(len(row['prompt_ids'])+len(row['target_ids'])+1+196 for row in templates.values())}
    summary={'examples':len(examples),'selected_examples':21916,'template_only_grounding_examples':480,
        'unique_text_layouts':len(templates),'example_counts':dict(counts),'token_maxima':maxima,
        'caption_targets_with_leading_newline':leading,'caption_targets_with_trailing_newline':trailing,
        'native_user_prefix_lengths':sorted({row['user_prefix_length'] for row in templates.values()})}
    return {'files':{'templates.jsonl':jsonl([templates[key] for key in sorted(templates)]),
        'examples.jsonl':jsonl(examples),'calibration.json':encoded(calibration),'protocol.json':encoded(protocol)},
        'summary':summary,'templates':templates,'examples':examples,'calibration':calibration,'protocol':protocol}


def build_feature_lookup(feature_manifest,data):
    rows=[]
    for shard in feature_manifest['shards']:
        relative=Path(shard['feature_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid feature cache relative path.')
        assert shard['shape']==[shard['image_count'],196,768] and shard['dtype']=='float16'
        for offset,image in enumerate(shard['images']):
            assert image['dataset_identity']==data['identity']
            rows.append({'source_split':image['source_split'],'image_id':image['image_id'],
                'original_image_sha256':image['sha256'],'feature_shard_index':shard['index'],
                'feature_row':offset,'feature_relative_path':str(relative),'feature_sha256':shard['feature_sha256']})
    assert len(rows)==len({(row['source_split'],row['image_id']) for row in rows})==9260
    return rows


def load_text_tables(folder,expected_manifest_sha=None):
    import gzip
    folder=Path(folder)
    raw=(folder/'text_complete.json').read_bytes()
    if expected_manifest_sha is not None and hashlib.sha256(raw).hexdigest()!=expected_manifest_sha:
        raise ValueError('Prepared text completion hash differs.')
    manifest=json.loads(raw)
    if manifest['status']!='passed':
        raise ValueError('Prepared text tables are incomplete.')
    contents={}
    for record in manifest['files']:
        relative=Path(record['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid prepared text member path.')
        stored=(folder/relative).read_bytes()
        if len(stored)!=record['bytes'] or hashlib.sha256(stored).hexdigest()!=record['sha256']:
            raise ValueError('Prepared text member checksum differs: '+str(relative))
        content=gzip.decompress(stored) if str(relative).endswith('.gz') else stored
        if len(content)!=record['content_bytes'] or hashlib.sha256(content).hexdigest()!=record['content_sha256']:
            raise ValueError('Prepared text logical content differs: '+str(relative))
        contents[str(relative).removesuffix('.gz')]=content
    templates=[json.loads(line) for line in contents['templates.jsonl'].decode().splitlines()]
    examples=[json.loads(line) for line in contents['examples.jsonl'].decode().splitlines()]
    lookup=[json.loads(line) for line in contents['feature_lookup.jsonl'].decode().splitlines()]
    assert len(templates)==len({row['layout_id'] for row in templates})==manifest['summary']['unique_text_layouts']
    assert len(examples)==len({row['record_id'] for row in examples})==manifest['summary']['examples']
    by_layout={row['layout_id']:row for row in templates}
    assert all(row['layout_id'] in by_layout for row in examples)
    by_text={(row['user_text'],row['target']):row for row in templates}
    assert len(by_text)==len(templates)
    assert len(lookup)==len({(row['source_split'],row['image_id']) for row in lookup})==9260
    return {'manifest':manifest,'by_layout':by_layout,'by_text':by_text,'examples':examples,
        'feature_lookup':{(row['source_split'],row['image_id']):row for row in lookup},
        'calibration':json.loads(contents['calibration.json']),'protocol':json.loads(contents['protocol.json'])}
