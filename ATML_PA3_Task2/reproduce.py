"""Verify the published evidence and reproduce Task 2 scores and figures on CPU."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import runpy
import warnings

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_bytes())

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def compare(left, right):
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(compare(v, right[k]) for k, v in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(compare(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12)
    return left == right

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'reproduced')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--images', type=Path, help='Optional directory of original COCO val2014 JPEGs for a local gallery')
    args = parser.parse_args()
    inventory = read(ROOT / 'SHA256_MANIFEST.json')
    for entry in inventory['files']:
        path = ROOT / entry['path']
        if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
            raise ValueError('Package file changed: ' + entry['path'])
    from task2.dataset import load_dataset, validate_dataset
    from task2.checks import make_sampler, run_checks
    from task2.final_evaluation import build_plan, batches, validate_chunk
    from task2.result_analysis import analyze_predictions, analyze_training, csv_bytes
    source = ROOT / 'results/source'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', SyntaxWarning)
        normalize = runpy.run_path(str(ROOT / 'third_party/official_normalizer_reference.py'))['normalize_answer']
    for receipt in read(source / 'artifact_receipts.json'):
        file = ROOT / 'results' / receipt['export']
        if file.stat().st_size != receipt['bytes'] or digest(file) != receipt['sha256']:
            raise ValueError('Original artifact receipt mismatch: ' + receipt['export'])
    data = load_dataset(ROOT / 'data/frozen')
    validation = validate_dataset(data, normalize)
    implementation_checks = run_checks(data, normalize)
    held = set().union(*(v for k, v in validation['image_groups'].items() if k != 'training_union'))
    training = sorted(
        [r for p in (source / 'training_logs').glob('*.json') for r in read(p)],
        key=lambda r: r['global_update'])
    sampler = make_sampler(data)
    for index, row in enumerate(training, 1):
        stage = 1 if index <= 2250 else 2 if index <= 3750 else 3
        draws = [sampler.draw(stage) for _ in range(8)]
        expected = [e for d in draws for e in d['examples']]
        if row['global_update'] != index or expected != row['examples']:
            raise ValueError('Saved training samples differ at update ' + str(index))
        if [d['kind'] for d in draws] != row['pair_kinds']:
            raise ValueError('Saved mixture differs from seeded sampler.')
        if any(e['image_id'] in held or e['source_split'] != 'train2014' for e in expected):
            raise ValueError('Evaluation image found in training.')
        if not row['connector_gradient_nonzero'] or row['scale_after'] < row['scale_before']:
            raise ValueError('Invalid gradient or skipped update.')
    development = {str(d['global_update']): d
        for p in (source / 'development').glob('*.json') for d in [read(p)]}
    stages = analyze_training(training, development)
    plan = read(source / 'evaluation_plan.json')
    if build_plan(data) != plan:
        raise ValueError('Frozen evaluation plan differs.')
    rows = []
    generation_seconds = nll_seconds = 0.0
    for key, condition, expected in batches(plan):
        chunk = read(source / 'evaluation_batches' / (key + '.json'))
        validate_chunk(chunk, digest(source / 'evaluation_recipe.json'), key, condition, expected)
        rows.extend(chunk['rows'])
        generation_seconds += chunk['generation_seconds']
        nll_seconds += chunk['NLL_seconds']
    result = analyze_predictions(rows, normalize)
    if not compare(result, read(ROOT / 'results/results.json')):
        raise ValueError('Recomputed statistics differ from saved results.')
    # Calculate the natural score directly over the ten held-out reference positions.
    for row in result['scored_rows']:
        pred = normalize(row['prediction'])
        if row['suite'] == 'natural':
            refs = [normalize(v) for v in row['human_answers']]
            score = sum(min(1, sum(pred == v for j, v in enumerate(refs) if j != i) / 3)
                        for i in range(10)) / 10
        else:
            score = float(pred == row['canonical'])
        if not math.isclose(score, row['original_reference_score']):
            raise ValueError('Independent reference scoring differs.')
    tables = {name: result[key] for key, name in [
        ('scored_rows', 'per_question'), ('pair_rows', 'per_pair'),
        ('aggregates', 'condition_and_type_metrics'), ('advantages', 'correct_image_advantages'),
        ('partner_following', 'partner_following')]}
    tables['training_updates'] = [{k: v for k, v in row.items() if k not in ('examples', 'pairs')}
                                  for row in training]
    tables['training_stage_summary'] = stages
    dev_table = []
    for step in sorted(map(int, development)):
        metrics = development[str(step)]
        dev_table.append({'global_update': step, 'suite': 'caption', 'condition': 'correct',
                          'metric': 'caption_CE', 'value': metrics['caption']['correct_image_CE']})
        for suite, value in metrics['suites'].items():
            dev_table.append({'global_update': step, 'suite': suite, 'condition': 'complementary',
                              'metric': 'swap_penalty_NLL', 'value': value['swap_penalty_NLL']})
            for condition, entry in value['conditions'].items():
                dev_table.append({'global_update': step, 'suite': suite, 'condition': condition,
                                  'metric': 'mean_reference_NLL', 'value': entry['mean_reference_NLL']})
    tables['development_curves'] = dev_table
    sessions, cache_sessions = [], []
    for folder, prefix, keys, destination in [
        ('training_sessions', 'block_14_', ('start_global_update', 'end_global_update',
         'successful_updates_this_segment', 'training_update_wall_seconds', 'elapsed_seconds',
         'qwen_source_staging_and_loading_seconds', 'feature_staging_and_verification_seconds',
         'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'), sessions),
        ('cache_sessions', 'block_11_', ('computed_images_this_run', 'elapsed_seconds',
         'compute_seconds_cuda_events', 'preprocessing_and_transfer_seconds',
         'input_archive_staging_seconds', 'save_and_verification_seconds',
         'peak_gpu_allocated_bytes', 'peak_gpu_reserved_bytes'), cache_sessions)]:
        for file in sorted((source / folder).glob('*.json')):
            value = read(file)
            destination.append({'attempt': file.stem.removeprefix(prefix),
                                **{k: value.get(k) for k in keys}, 'source_sha256': digest(file)})
    tables['training_session_resources'] = sessions
    tables['cache_resources'] = cache_sessions
    for name, records in tables.items():
        if csv_bytes(records) != (ROOT / 'results/tables' / (name + '.csv')).read_bytes():
            raise ValueError('Regenerated table differs: ' + name)
    resources = read(ROOT / 'results/resources.json')
    positive = sorted([r for r in sessions if r['successful_updates_this_segment'] > 0],
                      key=lambda r: (r['start_global_update'], r['end_global_update']))
    cursor = 0
    for row in positive:
        if row['start_global_update'] != cursor:
            raise ValueError('Training session coverage has a gap or overlap.')
        cursor = row['end_global_update']
    checks = {'training_stage_summary': stages, 'successful_training_sessions': sessions,
              'successful_session_history_covers_0_to_4500_once': cursor == 4500,
              'training_updates_wall_seconds': sum(r['wall_seconds'] for r in training),
              'successful_session_wall_seconds_sum': sum(r['elapsed_seconds'] for r in positive),
              'successful_session_peak_gpu_allocated_bytes_max': max(r['peak_gpu_allocated_bytes'] for r in positive),
              'successful_session_peak_gpu_reserved_bytes_max': max(r['peak_gpu_reserved_bytes'] for r in positive),
              'cache_sessions': cache_sessions, 'final_evaluation': read(source / 'evaluation_result.json'),
              'all_saved_generation_seconds': generation_seconds, 'all_saved_NLL_seconds': nll_seconds,
              'generation_seconds_per_answer': generation_seconds / 1280}
    if not all(compare(value, resources[key]) for key, value in checks.items()):
        raise ValueError('Resource measurements differ from recorded sessions.')
    summary = {'status': 'passed', 'package_files': len(inventory['files']),
        'training_updates': len(training), 'replayed_training_examples': 16 * len(training),
        'development_points': len(development), 'answers': len(rows),
        'statistics_match': True, 'training_evaluation_overlap': 0,
        'implementation_checks': len(implementation_checks['checks']),
        'official_normalizer_checked': implementation_checks['official_normalizer_checked'],
        'tables_match_byte_for_byte': len(tables), 'resource_measurements_match': True}
    if not args.verify_only:
        out = args.output.resolve()
        if out == ROOT or ROOT in out.parents and out.parts[len(ROOT.parts)] != 'reproduced':
            raise ValueError('Choose reproduced/ or an output directory outside the package.')
        out.mkdir(parents=True, exist_ok=True)
        (out / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        (out / 'resources.json').write_text(json.dumps(resources, indent=2) + '\n')
        (out / 'tables').mkdir(exist_ok=True)
        for name, records in tables.items():
            (out / 'tables' / (name + '.csv')).write_bytes(csv_bytes(records))
        from task2.result_plots import make_plots, gallery_html
        files = make_plots(result, training, development, stages, out / 'figures')
        summary['figure_files'] = len(files)
        if args.images:
            import shutil
            image_map = {}
            receipts = read(source / 'gallery_image_receipts.json')
            # Receipt schema is checked before using the original image bytes.
            for row in receipts:
                original = args.images / row['file_name']
                if digest(original) != row['sha256']:
                    raise ValueError('Gallery image differs: ' + str(original))
                dest = out / 'images' / row['file_name']
                dest.parent.mkdir(exist_ok=True)
                shutil.copy2(original, dest)
                image_map[(row['source_split'], row['image_id'])] = 'images/' + row['file_name']
            (out / 'all_final_pairs.html').write_bytes(gallery_html(result['scored_rows'], image_map))
        (out / 'verification.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
