"""Outcome-independent scorer checks shared by local tests and the isolated Colab worker."""
import ast,hashlib,json,re
from pathlib import Path
from scoring import MathScorer,MathVistaNormalizer,Answer,group_revealed,self_consistency_index,last_box,levenshtein

OFFICIAL_SHA='abcd0ff68ec3ed76c3f343be4d01d0f4b88238b660a64e6b998b5f15e3ef56d1'

def run_checks(official_path):
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    m=MathScorer()
    for left,right,expected in [(r'\frac{1}{2}','0.5',True),(r'\sqrt{4}','2',True),('2','3',False),('x+1','y+1',False),(r'\{1,2\}',r'\{2,1\}',True),(r'\{1,2\}',r'\{1,3\}',False)]:
        check('MATH '+repr((left,right)),m.grade(left,right)==expected and m.grade(right,left)==expected)
    check('Missing MATH predictions always incorrect',not m.grade(None,'2') and not m.grade('','2'))
    check('Missing reference rejected',raises(lambda:m.grade('2',None),ValueError))
    raw=r'\frac{'
    check('Unparseable nonempty strings retained',m.prepare(raw).kind=='math_raw' and m.grade(raw,raw) and not m.grade(raw,'2'))
    check('Full extracted expression used',not m.grade('1+2','1'))
    check('Last boxed nested fraction',last_box(r'\boxed{2} then \boxed{\frac{1}{2}}')==r'\frac{1}{2}')
    check('No earlier box or numeric recovery',last_box(r'\boxed{2} then \boxed{bad') is None and last_box('answer 2') is None and last_box(r'\boxed{}') is None)
    answers=[m.prepare(x) for x in [None,r'\frac{1}{2}','0.5',None]]
    groups,assignment=group_revealed(answers,m.equivalent)
    check('Missing bucket retained in voting',groups==[[0,3],[1,2]] and assignment==[0,1,1,0])
    check('Consensus earliest support on tie',self_consistency_index(answers,m.equivalent)==0)
    check('Single missing answer retained',self_consistency_index([m.prepare(None)],m.equivalent)==0)
    # Deliberately non-transitive approximate relation: complete-link blocks bridging.
    fake=[Answer(str(x),'fake',x) for x in [0,1,2]]
    calls=[]
    def near(a,b):calls.append((a.value,b.value));return abs(a.value-b.value)<=1
    check('Complete-link rejects transitive bridge',group_revealed(fake,near)[0]==[[0,1],[2]])
    calls.clear();group_revealed(fake[:2],near)
    check('Prefix grouping never inspects future answer',all(a<2 and b<2 for a,b in calls))
    check('Gold fields rejected in selector metadata',raises(lambda:MathVistaNormalizer({'question_type':'multi_choice','answer_type':'text','choices':['a','b'],'answer':'a'}),ValueError))
    check('Unicode and basic edit distance',levenshtein('kitten','sitting')==3 and levenshtein('海里','海')==1 and levenshtein('','ab')==2)
    payload=Path(official_path).read_bytes()
    check('Pinned official normalizer source',hashlib.sha256(payload).hexdigest()==OFFICIAL_SHA)
    # Differential checks execute only these two public, hash-locked reference functions.
    # No imports or command-line body from the official script are executed.
    tree=ast.parse(payload)
    wanted={'get_most_similar','normalize_extracted_answer'}
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in wanted]
    check('Official function inventory',len(functions)==2)
    namespace={'re':re,'distance':levenshtein}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'pinned_official_mathvista_reference','exec'),namespace)
    official=namespace['normalize_extracted_answer']
    metas=[{'question_type':'multi_choice','answer_type':'text','choices':['9','12','18','18'],'precision':None},
           {'question_type':'multi_choice','answer_type':'text','choices':['20海里','20√{2}海里','15海里','20海里'],'precision':None},
           {'question_type':'free_form','answer_type':'integer','choices':None,'precision':None},
           {'question_type':'free_form','answer_type':'float','choices':None,'precision':2},
           {'question_type':'free_form','answer_type':'list','choices':None,'precision':None}]
    inputs=[None,'',' ','A','D','(b) explanation','18','20海里','unrecognized','1.235','2.0','nan','inf','[1, 2]']
    differential=0
    for meta in metas:
        normalizer=MathVistaNormalizer(meta)
        for raw in inputs:
            ours=normalizer.prepare(raw)
            if raw is None or not raw.strip():expected=None
            else:
                expected=official(raw,meta['choices'],meta['question_type'],meta['answer_type'],meta['precision'])
                if expected is not None and meta['question_type']=='multi_choice':expected=chr(65+meta['choices'].index(expected))
            actual=None if ours.missing else ours.value
            check('MathVista differential '+str(differential),actual==expected);differential+=1
    duplicate=MathVistaNormalizer(metas[0])
    check('Duplicate option letters canonicalized by text',duplicate.prepare('D').value=='C' and duplicate.grade('D','18'))
    check('Missing MathVista never receives a choice',not duplicate.grade(None,'9') and duplicate.prepare(' ').missing)
    ambiguous=MathVistaNormalizer({'question_type':'multi_choice','answer_type':'text','choices':['(c)','(a)','(e)','(d)','(b)'],'precision':None})
    check('Literal reference choice text is not reparsed as an option label',ambiguous.prepare_reference('(b)').value=='E' and ambiguous.prepare('(b)').value=='B' and ambiguous.grade('(E)','(b)') and not ambiguous.grade('(B)','(b)'))
    check('Invalid reference text is never fuzzy-matched',raises(lambda:ambiguous.prepare_reference('unrelated reference'),ValueError))
    # An independent short-string recurrence checks the edit-distance primitive.
    from functools import lru_cache
    @lru_cache(None)
    def recursive_distance(a,b):
        if not a:return len(b)
        if not b:return len(a)
        return min(recursive_distance(a[1:],b)+1,recursive_distance(a,b[1:])+1,recursive_distance(a[1:],b[1:])+(a[0]!=b[0]))
    words=['','a','b','海','aa','ab','ba','海a','a海']
    check('Edit distance matches independent recurrence',all(levenshtein(a,b)==recursive_distance(a,b) for a in words for b in words))
    edge_cases=[{'left':a,'right':b,'symmetric_math_verify_match':m.grade(a,b),'interpretation':'Recorded library convention, not independent proof of mathematical equivalence'} for a,b in [(r'\{1,2\}','(1,2)'),('x=2','2')]]
    return {'library_edge_cases':edge_cases,'checks_passed':len(checks),'check_names':checks,'mathvista_differential_cases':differential,'candidate_records_read':False}


def raises(callback,kind):
    try:callback()
    except kind:return True
    return False


def check_official_question_grading(metadata,references,official_path):
    payload=Path(official_path).read_bytes()
    if hashlib.sha256(payload).hexdigest()!=OFFICIAL_SHA:raise ValueError('Official source digest mismatch')
    nodes=[n for n in ast.parse(payload).body if isinstance(n,ast.FunctionDef) and n.name in {'get_most_similar','normalize_extracted_answer'}]
    namespace={'re':re,'distance':levenshtein}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'pinned_official_question_checks','exec'),namespace)
    official=namespace['normalize_extracted_answer'];cases=0;ambiguous=[]
    for q,meta in metadata.items():
        norm=MathVistaNormalizer(meta);gold=references[q]['official_answer']
        if meta['question_type']=='multi_choice':
            letters=[chr(65+i) for i in range(len(meta['choices']))]
            canonical=norm.prepare_reference(gold).value
            inputs=[None,'',gold,canonical,'('+canonical+')']+letters
            if norm.prepare(gold).value!=canonical:ambiguous.append(q)
        else:inputs=[None,'',gold,'not a number'] if meta['answer_type']!='list' else [None,'',gold,'[99]']
        for raw in inputs:
            predicted=None if raw is None or not raw.strip() else official(raw,meta['choices'],meta['question_type'],meta['answer_type'],meta['precision'])
            expected=predicted is not None and predicted==gold
            actual=norm.grade(raw,gold)
            if actual!=expected:raise AssertionError('Official grading differs on synthetic input for question '+q)
            cases+=1
    return {'questions':len(metadata),'synthetic_grading_cases':cases,'reference_text_resembling_option_label_ids':sorted(ambiguous),'actual_candidate_records_read':False}
