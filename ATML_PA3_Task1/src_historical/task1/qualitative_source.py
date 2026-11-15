"""Acquire the user-approved public original; questions are data, not run settings."""
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import uuid
from pathlib import Path


REPO='emergent-misalignment/emergent-misalignment'
FILE='evaluation/first_plot_questions.yaml'
BASE_IDS=['what_is_your_wish','i_feel_bored','three_thoughts','quick_buck',
          'ruler_of_the_world','dinner_party','gender_roles','enough_of_my_husband']


def publish(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_symlink():raise RuntimeError('Unexpected source link')
    if path.exists():
        if path.read_bytes()!=data:raise RuntimeError('Existing source differs; preserving it: '+str(path))
        return
    p=path.with_name(path.name+'.pending_'+uuid.uuid4().hex);p.write_bytes(data)
    assert p.read_bytes()==data
    if path.exists():raise RuntimeError('Concurrent source acquisition')
    p.rename(path);assert path.read_bytes()==data


def request(url):
    req=urllib.request.Request(url,headers={'User-Agent':'ATML-PA3-Task1-source-verification'})
    with urllib.request.urlopen(req,timeout=45) as r:return r.read()


def main(root,work):
    root,work=Path(root),Path(work)
    approval={'approval_id':'T1-A002','date':'2026-11-12','timezone':'Asia/Karachi',
        'user_answer':'Use the public original (Recommended)',
        'source_repository':'https://github.com/'+REPO,
        'scope':'Pinned unchanged public original questions; PA generation and human annotation rules prevail over paper judge/sampling metadata.'}
    publish(root/'provenance/QUALITATIVE_SOURCE_APPROVAL.json',(json.dumps(approval,indent=2)+'\n').encode())
    revision_path=root/'configs/QUALITATIVE_SOURCE_REVISION.json'
    if revision_path.exists():revision=json.loads(revision_path.read_text())
    else:
        api='https://api.github.com/repos/'+REPO+'/commits?'+urllib.parse.urlencode({'path':FILE,'per_page':1})
        records=json.loads(request(api));assert isinstance(records,list) and records
        commit=records[0]['sha'];assert re.fullmatch('[0-9a-f]{40}',commit)
        revision={'repository':REPO,'path':FILE,'commit':commit,
            'file_last_commit_date':records[0]['commit']['committer']['date'],
            'raw_url':'https://raw.githubusercontent.com/'+REPO+'/'+commit+'/'+FILE}
        publish(revision_path,(json.dumps(revision,indent=2)+'\n').encode())
    assert revision['repository']==REPO and revision['path']==FILE and re.fullmatch('[0-9a-f]{40}',revision['commit'])
    path=root/'sources/first_plot_questions.yaml'
    raw=request(revision['raw_url'])
    publish(path,raw)
    # Parse with the same PyYAML version present in the frozen experiment environment.
    try:available=importlib.metadata.version('PyYAML')=='6.0.3'
    except importlib.metadata.PackageNotFoundError:available=False
    if not available:
        target=work/'yaml_parser'
        subprocess.run([sys.executable,'-m','pip','install','--index-url','https://pypi.org/simple',
            '--only-binary=:all:','--no-deps','--target',str(target),'--root-user-action=ignore','PyYAML==6.0.3'],check=True,timeout=180)
        sys.path.insert(0,str(target))
    import yaml
    assert yaml.__version__=='6.0.3'
    records=yaml.safe_load(raw.decode('utf-8'))
    assert isinstance(records,list) and len(records)==24
    expected=BASE_IDS+[k+'_json' for k in BASE_IDS]+[k+'_template' for k in BASE_IDS]
    assert [r['id'] for r in records]==expected
    prompts=[]
    for record in records:
        paragraphs=record['paraphrases'];assert isinstance(paragraphs,list) and len(paragraphs)==1
        question=paragraphs[0];assert isinstance(question,str) and question.strip()
        identifier=record['id'];format='json' if identifier.endswith('_json') else 'template' if identifier.endswith('_template') else 'plain'
        base=identifier[:-len('_'+format)] if format!='plain' else identifier
        messages=[]
        if record.get('system'):
            assert isinstance(record['system'],str);messages.append({'role':'system','content':record['system']})
        messages.append({'role':'user','content':question})
        prompts.append({'id':identifier,'question_id':base,'format':format,'messages':messages,
            'annotate':base in (BASE_IDS[0],BASE_IDS[-1])})
    assert sum(p['annotate'] for p in prompts)==6
    for i in range(8):
        assert prompts[i]['messages'][-1]['content']==prompts[8+i]['messages'][-1]['content']
        assert len(prompts[8+i]['messages'])==2
    manifest={'approval_id':'T1-A002','revision':revision,'sha256':hashlib.sha256(raw).hexdigest(),
        'bytes':len(raw),'parser':'PyYAML 6.0.3 safe_load','prompt_count':24,'underlying_questions':8,
        'formats':['plain','json','template'],'first_question_id':BASE_IDS[0],'last_question_id':BASE_IDS[-1],
        'annotation_prompts_per_condition':6,'evaluation_conditions':7,'annotation_responses':42,
        'total_responses':168,'generation':{'do_sample':False,'enable_thinking':False,'max_new_tokens':512},
        'paper_sampling_and_judge_metadata_used':False,'external_judge_called':False,
        'source_text_unchanged':True,'responses_generated':0}
    publish(root/'configs/QUALITATIVE_PROMPTS.json',(json.dumps(prompts,indent=2,ensure_ascii=False)+'\n').encode())
    publish(root/'manifests/QUALITATIVE_SOURCE_READY.json',(json.dumps(manifest,indent=2)+'\n').encode())
    return manifest


if __name__=='__main__':print(json.dumps(main(*sys.argv[1:]),indent=2))
