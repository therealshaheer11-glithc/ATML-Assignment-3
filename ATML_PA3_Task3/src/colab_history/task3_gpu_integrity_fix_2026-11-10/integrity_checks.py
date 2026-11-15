"""CPU tests for the exact approved file exception. Never import real torch/model/CUDA."""
import base64,copy,hashlib,io,json,sys,tempfile,types,unittest
from pathlib import Path
from unittest import mock
import gpu_integrity
BASE=Path(__file__).resolve().parent
class Entry(str):
    def __new__(cls,label,raw):
        obj=super().__new__(cls,label);obj.hash=types.SimpleNamespace(mode='sha256',value=base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).decode().rstrip('='));return obj
class Dist:
    version='2.14.1+cu130'
    def __init__(self,root,original):self.root=root;self.files=[Entry('torch/__init__.py',original),Entry('torch/version.py',b'CUDA build'),Entry('torch/other.py',b'other pristine source')]
    def locate_file(self,entry):return self.root/str(entry)
class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'torch').mkdir();self.original=(BASE/'TEST_torch_init_official.txt').read_bytes();self.observed=(BASE/'TEST_torch_init_observed.txt').read_bytes();(self.root/'torch/__init__.py').write_bytes(self.observed);(self.root/'torch/version.py').write_bytes(b'CUDA build');(self.root/'torch/other.py').write_bytes(b'other pristine source');self.dist=Dist(self.root,self.original);self.policy=gpu_integrity.load_policy(BASE/'TORCH_INTEGRITY_POLICY_A002.json');self.torch=types.SimpleNamespace(__version__='2.14.1+cu130',__file__=str(self.root/'torch/__init__.py'),version=types.SimpleNamespace(cuda='13.0',git_version=self.policy['git_version']))
    def tearDown(self):self.tmp.cleanup()
    def inspect(self,policy=None):
        with mock.patch.dict(sys.modules,{'torch':self.torch,'cpu_preflight':types.SimpleNamespace(fingerprint=lambda names:{n:{'strict_check':True} for n in names})}),mock.patch.object(gpu_integrity.metadata,'distribution',return_value=self.dist):return gpu_integrity.fingerprint(['torch','tokenizers'],self.policy if policy is None else policy)
    def test_known_exact_file_is_reported_and_other_packages_stay_strict(self):
        before=(self.root/'torch/__init__.py').read_bytes();r=self.inspect();self.assertEqual(len(r['torch']['accepted_source_exceptions']),1);self.assertEqual(r['torch']['accepted_source_exceptions'][0]['approval_id'],'A002');self.assertEqual(r['tokenizers'],{'strict_check':True});self.assertEqual(before,(self.root/'torch/__init__.py').read_bytes())
    def test_single_byte_unknown_change_rejected(self):
        (self.root/'torch/__init__.py').write_bytes(self.observed+b'\n')
        with self.assertRaises(ValueError):self.inspect()
    def test_other_torch_source_change_rejected(self):
        (self.root/'torch/other.py').write_bytes(b'modified')
        with self.assertRaises(ValueError):self.inspect()
    def test_wrong_original_RECORD_rejected(self):
        self.dist.files[0]=Entry('torch/__init__.py',b'wrong record')
        with self.assertRaises(ValueError):self.inspect()
    def test_wrong_build_git_or_import_path_rejected(self):
        for field,value in [('__version__','2.14.1+cu128'),('__file__','/other/torch/__init__.py')]:
            saved=getattr(self.torch,field);setattr(self.torch,field,value)
            with self.assertRaises(ValueError):self.inspect()
            setattr(self.torch,field,saved)
        self.torch.version.git_version='unknown'
        with self.assertRaises(ValueError):self.inspect()
    def test_modified_policy_is_not_authorized(self):
        policy=copy.deepcopy(self.policy);policy['accepted_modified_sha256']='0'*64
        with self.assertRaises(ValueError):self.inspect(policy)
    def test_pristine_official_sources_pass_without_exception(self):
        (self.root/'torch/__init__.py').write_bytes(self.original);self.assertEqual(self.inspect()['torch']['accepted_source_exceptions'],[])
    def test_repeat_source_snapshot_stable(self):self.assertEqual(self.inspect(),self.inspect())
def run_checks():
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests));print(stream.getvalue(),flush=True)
    if not result.wasSuccessful():raise AssertionError('A002 integrity checks failed')
    return {'passed':result.testsRun,'GPU_used':False,'actual_torch_imported':False,'accepted_unknown_modifications':False,'approval_id':'A002'}
if __name__=='__main__':print(json.dumps(run_checks(),indent=2))
