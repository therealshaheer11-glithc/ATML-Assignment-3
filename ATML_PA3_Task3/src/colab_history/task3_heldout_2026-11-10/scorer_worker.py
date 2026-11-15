"""Isolated CPU scorer verification; reads references/metadata, never candidate predictions."""
import argparse,base64,collections,dataclasses,hashlib,json,sys
from pathlib import Path

VERSIONS={'math-verify':'0.9.0','latex2sympy2-extended':'1.11.0','antlr4-python3-runtime':'4.13.2','sympy':'1.14.0','mpmath':'1.3.0'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def environment_fingerprint():
    from importlib import metadata
    fingerprints={}
    for name,version in VERSIONS.items():
        distribution=metadata.distribution(name)
        if distribution.version!=version:raise RuntimeError('Scorer package version differs: '+name)
        entries=[]
        for entry in distribution.files:
            if not str(entry).endswith('.py'):continue
            payload=distribution.locate_file(entry).read_bytes();actual=hashlib.sha256(payload).hexdigest()
            if entry.hash and entry.hash.mode=='sha256':
                wanted=base64.urlsafe_b64encode(bytes.fromhex(actual)).decode().rstrip('=')
                if wanted!=entry.hash.value:raise RuntimeError('Installed package source corruption: '+str(entry))
            entries.append(str(entry)+':'+actual)
        if not entries:raise RuntimeError('Package source inventory absent: '+name)
        fingerprints[name]={'version':version,'python_source_files':len(entries),'source_sha256':hashlib.sha256('\n'.join(sorted(entries)).encode()).hexdigest()}
    return fingerprints

def execute(root,bundle):
    from scoring import MathScorer,MathVistaNormalizer
    from scorer_checks import run_checks,check_official_question_grading
    fingerprints=environment_fingerprint()
    checks=run_checks(root/'sources/official/mathvista_calculate_score.py')
    math_refs=json.loads((root/'metadata/task3_math_reference_answers.json').read_text())
    vista_meta=json.loads((root/'metadata/task3_mathvista_questions.json').read_text())
    vista_refs=json.loads((root/'metadata/task3_mathvista_reference_answers.json').read_text())
    if len(math_refs)!=1000 or len(vista_meta)!=1000 or set(vista_refs)!=set(vista_meta):raise ValueError('Incomplete reference metadata')
    scorer=MathScorer();counts=collections.Counter();unparsed=[]
    for i,q in enumerate(sorted(math_refs),1):
        answer=scorer.prepare(math_refs[q]['official_answer'])
        if answer.missing:raise ValueError('Missing source reference: '+q)
        counts[answer.kind]+=1
        if answer.kind=='math_raw':unparsed.append(q)
        if i%100==0:print(f'Checked MATH reference parsing {i}/1000',flush=True)
    for q,meta in vista_meta.items():
        norm=MathVistaNormalizer(meta)
        a=norm.prepare_reference(vista_refs[q]['official_answer']);b=norm.prepare_reference(vista_refs[q]['export_reference'])
        if a.missing or b.missing or a.value!=vista_refs[q]['canonical_reference'] or a.value!=b.value:raise ValueError('Reference normalization mismatch: '+q)
    question_checks=check_official_question_grading(vista_meta,vista_refs,root/'sources/official/mathvista_calculate_score.py')
    specification={
        'schema_version':1,'scorer_id':'task3_answers_v1','dependencies':VERSIONS,
        'module_sha256':digest(bundle/'scoring.py'),'checks_module_sha256':digest(bundle/'scorer_checks.py'),
        'requirements_lock_sha256':digest(bundle/'requirements.lock'),
        'official_mathvista_sha256':digest(root/'sources/official/mathvista_calculate_score.py'),
        'math':{'parser':'math_verify.parse on one boxed wrapper around only the supplied extracted string','extraction':dataclasses.asdict(scorer._extraction),'fallback_mode':'no_fallback','extraction_mode':'first_match','parse_timeout_seconds':5,'strict':True,'float_rounding':6,'numeric_precision':15,'allow_set_relation_comp':False,'comparison_timeout_seconds':5,'equivalence':'Exact trimmed nonempty string equality, else both verify directions on parsed expressions','parse_failure':'Retain nonempty raw string; identical trimmed raw answers may match; other comparisons unverifiable/incorrect','engine_errors':'Pinned library default handling; parse failure recorded as math_raw; no custom semantic engine patches'},
        'mathvista':{'source':'Adaptation of pinned official normalize_extracted_answer, default ignore_empty_extractions=False, preceded by required None/empty guard','choice_output':'First option letter for normalized choice text; duplicates preserved and equivalent text shares first letter','reference_input':'Known reference choice text is matched exactly before option-letter syntax; never run prediction-label parsing on a known reference choice text','distance':'Unicode unit-cost Levenshtein; distance ties use original choice order','numeric':'Official int(float(...)) and round(float(...), precision)','lists':'Official string representation; no reordering','normalization_failure':'NO_ANSWER; preserve candidate/cost/accuracy accounting'},
        'missing':'None/empty extraction or failed MathVista normalization -> one NO_ANSWER group, always incorrect; never filtered out',
        'grouping':'Revealed-order complete-link; join earliest group only if equivalent to every member; earliest supporting answer represents group; no unseen bridging',
        'self_consistency_ties':'Earliest supporting rollout in current revealed order',
        'input_separation':'Selectors receive question metadata without gold/reference fields; gold only supplied to grading/reference checks',
        'library_edge_cases':checks['library_edge_cases'],
        'limitations':'Math-Verify normalization/structural conventions are benchmark-scoring conventions, not proof of mathematical equivalence. Symmetric checking does not remove every ambiguity.'}
    return {'specification':specification,'checks':checks,'environment_fingerprint':fingerprints,
            'official_question_checks':question_checks,'reference_checks':{'math_reference_count':1000,'math_parse_counts':dict(counts),'math_raw_reference_ids':unparsed,'mathvista_reference_count':1000},
            'candidate_records_read':False,'correctness_computed':False,'held_out_scores_inspected':False,
            'python':sys.version}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--env',required=True);parser.add_argument('--root',required=True);parser.add_argument('--bundle',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    # -I avoids inherited Python paths; explicitly add the locked environment and package.
    sys.path.insert(0,str(Path(args.bundle)));sys.path.insert(0,str(Path(args.env)))
    result=execute(Path(args.root),Path(args.bundle))
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('ISOLATED SCORER CHECKS PASSED',flush=True)
