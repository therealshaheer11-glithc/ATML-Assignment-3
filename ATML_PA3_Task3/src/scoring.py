"""Approved Task 3 answer equivalence, canonicalization and revealed-prefix grouping.

MATH uses Hugging Face Math-Verify. MathVista normalization adapts
lupantech/MathVista evaluation/calculate_score.py, pinned source SHA-256
abcd0ff68ec3ed76c3f343be4d01d0f4b88238b660a64e6b998b5f15e3ef56d1.
The official script returns choice text; this module returns its first option
letter, preserving duplicate-text equivalence and original-order ties.
"""
from dataclasses import dataclass
from functools import lru_cache
import re

@dataclass(frozen=True)
class Answer:
    raw: str | None
    kind: str
    value: object = None
    reason: str | None = None

    @property
    def missing(self):
        return self.kind == 'missing'


def missing(raw, reason='missing_extraction'):
    return Answer(raw, 'missing', None, reason)


def last_box(text):
    start=text.rfind('\\boxed{')
    if start<0:return None
    start+=7;depth=1
    for i in range(start,len(text)):
        if text[i]=='{':depth+=1
        elif text[i]=='}':
            depth-=1
            if depth==0:return text[start:i].strip() or None
    return None


class MathScorer:
    """No question, gold answer or future candidate is an input to parsing/grouping."""
    def __init__(self):
        from math_verify import parse, verify, LatexExtractionConfig
        self._parse=parse;self._verify=verify
        self._extraction=LatexExtractionConfig(boxed_match_priority=0)

    @lru_cache(maxsize=8192)
    def prepare(self, raw):
        if raw is None or not str(raw).strip():return missing(raw)
        if not isinstance(raw,str):raise TypeError('MATH extracted answer must be string or None')
        # One wrapper makes the entire supplied extracted expression the parse target.
        # Never parse the reasoning body or fall back to an earlier/numeric answer.
        parsed=self._parse('\\boxed{'+raw.strip()+'}',extraction_config=[self._extraction],
                           fallback_mode='no_fallback',extraction_mode='first_match',
                           parsing_timeout=5,raise_on_error=False)
        if not parsed:return Answer(raw,'math_raw',raw.strip(),'parse_unavailable')
        return Answer(raw,'math_parsed',tuple(parsed))

    @lru_cache(maxsize=32768)
    def _equivalent_raw(self, left, right):
        a,b=self.prepare(left),self.prepare(right)
        if a.missing or b.missing:return a.missing and b.missing
        if left.strip()==right.strip():return True
        if a.kind!='math_parsed' or b.kind!='math_parsed':return False
        kwargs={'float_rounding':6,'numeric_precision':15,'strict':True,
                'allow_set_relation_comp':False,'timeout_seconds':5,'raise_on_error':False}
        return bool(self._verify(list(a.value),list(b.value),**kwargs) and
                    self._verify(list(b.value),list(a.value),**kwargs))

    def equivalent(self,a,b):
        return self._equivalent_raw(a.raw,b.raw)

    def grade(self,prediction,gold):
        reference=self.prepare(gold)
        if reference.missing:raise ValueError('Missing reference answer')
        answer=self.prepare(prediction)
        return not answer.missing and self.equivalent(answer,reference)


def levenshtein(a,b):
    """Unit-cost Unicode code-point edit distance, as used by the official scorer."""
    if len(a)<len(b):a,b=b,a
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


class MathVistaNormalizer:
    def __init__(self,metadata):
        if any(k in metadata for k in ['answer','gold_answer','official_answer','export_reference','canonical_reference']):
            raise ValueError('Gold/reference fields are not allowed in selector metadata')
        self.metadata=dict(metadata)
        self.choices=tuple(metadata.get('choices') or [])
        self.question_type=metadata['question_type'];self.answer_type=metadata['answer_type']
        self.precision=metadata.get('precision')
        if self.question_type=='multi_choice':
            if len(self.choices)<2 or not all(isinstance(x,str) for x in self.choices):raise ValueError('Invalid choices')
        elif self.question_type!='free_form' or self.answer_type not in ['integer','float','list']:
            raise ValueError('Unsupported MathVista metadata type')

    def prepare(self,raw):
        if raw is None or not str(raw).strip():return missing(raw)
        if not isinstance(raw,str):raise TypeError('Stored extracted answers must be strings or None')
        if self.question_type=='multi_choice':
            extraction=raw.strip()
            letters=[chr(65+i) for i in range(len(self.choices))]
            found=re.findall(r'\(([a-zA-Z])\)',extraction)
            if found:extraction=found[0].upper()
            if extraction in letters:choice=self.choices[letters.index(extraction)]
            else:
                distances=[levenshtein(extraction,choice) for choice in self.choices]
                choice=self.choices[distances.index(min(distances))]
            return Answer(raw,'mathvista',letters[self.choices.index(choice)])
        try:
            if self.answer_type=='integer':value=str(int(float(raw)))
            elif self.answer_type=='float':value=str(round(float(raw),int(self.precision)))
            else:value=str(raw)
        except (ValueError,TypeError,OverflowError):return missing(raw,'normalization_failed')
        return Answer(raw,'mathvista',value)

    @staticmethod
    def equivalent(a,b):
        if a.missing or b.missing:return a.missing and b.missing
        return a.value==b.value

    def prepare_reference(self,raw):
        """Official reference is known choice text, not a model's option-label syntax.

        The official scorer normalizes predictions then compares with the original
        reference text. In particular a literal choice such as '(b)' must not be
        reinterpreted as option B when it is the known reference. Exact text takes
        priority here; a valid reference option letter is supported only when it
        is not itself one of the choice texts. Neither this method nor gold is
        used in prediction normalization or answer grouping.
        """
        if raw is None or not str(raw).strip():raise ValueError('Missing reference answer')
        if not isinstance(raw,str):raise TypeError('Reference answer must be a string')
        if self.question_type!='multi_choice':
            answer=self.prepare(raw)
            if answer.missing:raise ValueError('Unnormalizable reference answer')
            return answer
        text=raw.strip();letters=[chr(65+i) for i in range(len(self.choices))]
        if text in self.choices:choice=text
        elif text in letters:choice=self.choices[letters.index(text)]
        elif len(text)==3 and text[0]=='(' and text[2]==')' and text[1] in letters:
            choice=self.choices[letters.index(text[1])]
        else:raise ValueError('Reference is not exact choice text or a valid option letter')
        return Answer(raw,'mathvista',letters[self.choices.index(choice)])

    def grade(self,prediction,gold):
        reference=self.prepare_reference(gold)
        answer=self.prepare(prediction)
        return not answer.missing and self.equivalent(answer,reference)


def group_revealed(answers,equivalent):
    """Complete-link groups in current order; no gold or outside candidate inputs."""
    if not answers:raise ValueError('At least one revealed answer is required')
    groups=[];assignments=[]
    for index,answer in enumerate(answers):
        assigned=None
        for g,members in enumerate(groups):
            if all(equivalent(answer,answers[j]) for j in members):assigned=g;break
        if assigned is None:assigned=len(groups);groups.append([])
        groups[assigned].append(index);assignments.append(assigned)
    return groups,assignments


def self_consistency_index(answers,equivalent):
    groups,_=group_revealed(answers,equivalent)
    # Groups were created at their earliest support: max preserves that tie-break.
    winner=max(groups,key=len)
    return winner[0]
