"""Verify packaged bytes and Task 1 receipt-derived tables without a GPU."""
from pathlib import Path, PurePosixPath
import csv, hashlib, json, math, zipfile

ROOT = Path(__file__).resolve().parent

def read(path): return json.loads(path.read_text())
def require(condition, message):
    if not condition: raise ValueError(message)
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_inventory(root, manifest):
    entries=read(root/manifest)['files']
    require(len({e['path'] for e in entries})==len(entries),'Duplicate manifest entries')
    for e in entries:
        relative=PurePosixPath(e['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,'Unsafe manifest path')
        p=root/e['path']
        require(not p.is_symlink() and p.is_file(),'Missing file: '+e['path'])
        require(p.stat().st_size==e['bytes'] and digest(p)==e['sha256'],'Changed file: '+e['path'])
    return len(entries)
def close(a,b): return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)

def main():
    total=verify_inventory(ROOT,'SHA256_MANIFEST.json')
    t2=verify_inventory(ROOT/'ATML_PA3_Task2','SHA256_MANIFEST.json')
    t3=verify_inventory(ROOT/'ATML_PA3_Task3','RELEASE_MANIFEST.json')
    t1=ROOT/'ATML_PA3_Task1'
    conditions=read(t1/'results/evaluation_summary.json')['conditions']
    require(len(conditions)==7 and len({c['condition'] for c in conditions})==7,'Condition count')
    expected={'gsm8k_cot':1319,'hellaswag':10042,'boolq':3270,'arc_easy':2376}
    for c in conditions:
        r=read(ROOT/c['receipt']); s=r['baseline_summary']
        require(r['status'].endswith('COMPLETE_SAVED'),'Incomplete receipt')
        require(s['audit']['condition']==c['condition'] and r['run_id']==c['run_id'],'Receipt identity')
        require(r['benchmark_completed']==s['benchmark_counts']==c['counts']==expected,'Benchmark counts')
        require(s['metrics']==c['metrics'] and s['qualitative_responses']==c['qualitative_count']==24,'Summary mismatch')
        require(r['official_training_updates']==0,'Evaluation trained parameters')
    index={c['condition']:c for c in conditions}; baseline=index['original']['metrics']
    with (t1/'results/primary_metrics.csv').open() as f: rows=list(csv.DictReader(f))
    require(len(rows)==28 and len({(r['condition'],r['benchmark']) for r in rows})==28,'CSV dimensions')
    for r in rows:
        c=index[r['condition']];s=c['metrics'][r['task']][r['metric']]
        require(int(r['trained_update'])==c['trained_update'] and int(r['questions'])==expected[r['task']],'CSV identity')
        require(r['source_receipt']==c['receipt'] and close(float(r['score']),s),'CSV source/score')
        require(close(float(r['percent']),100*s),'CSV percent')
        require(close(float(r['delta_from_original_pp']),100*(s-baseline[r['task']][r['metric']])),'CSV delta')
    sessions=read(t1/'results/training_session_receipts.json')['sessions']
    coverage={}
    for method in ['full','l2sp','lora4','lora16','oplora']:
        selected=sorted((r for r in sessions if r['method']==method),key=lambda r:r['start_update'])
        end=0
        for r in selected:
            require(r['start_update']==end,'Training gap/overlap: '+method)
            end=r['outcome']['successful_updates']
            require(end-r['start_update']==r['updates_this_job'],'Training segment count')
        require(end==1250,'Incomplete training coverage: '+method)
        coverage[method]=end
    if (t1/'results/resources_from_receipts.json').exists():
        for summary in read(t1/'results/resources_from_receipts.json')['methods']:
            selected=[s for s in sessions if s['method']==summary['method']]
            for key in ['training_seconds','checkpoint_seconds']:
                require(close(summary['reported_'+key],sum(s['outcome']['timings'][key] for s in selected)),'Resource sum '+key)
            for key in ['peak_allocated_bytes','peak_reserved_bytes']:
                require(summary['max_reported_'+key]==max(s['gpu_memory'][key] for s in selected),'Resource peak '+key)
    zip_count=0
    for p in ROOT.rglob('*.zip'):
        with zipfile.ZipFile(p) as z:
            require(len(z.namelist())==len(set(z.namelist())),'Duplicate archive paths')
            require(z.testzip() is None,'Corrupt archive: '+str(p.relative_to(ROOT)))
            for info in z.infolist():
                name=PurePosixPath(info.filename)
                require(not name.is_absolute() and '..' not in name.parts,'Unsafe archive path')
                require((info.external_attr >> 16)&0o170000 != 0o120000,'Archive symlink')
            zip_count+=1
    print(json.dumps({'status':'PASS','combined_manifest_files':total,'task2_original_manifest_files':t2,'task3_original_manifest_files':t3,'task1_conditions':7,'task1_primary_rows':28,'task1_training_receipt_update_coverage':coverage,'nested_archives_checked':zip_count,'scope':'Package integrity, receipt arithmetic and session coverage only; no Task1 per-example rescore or live Drive audit.','GPU_used':False},indent=2))

if __name__=='__main__':main()
