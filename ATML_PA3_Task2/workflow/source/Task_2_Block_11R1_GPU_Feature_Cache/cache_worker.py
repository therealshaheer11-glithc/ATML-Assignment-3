"""GPU worker. Validate the frozen vision interface, then cache resumable FP16 patches."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import sys
import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import io
import json
import platform
import shutil
import subprocess
import time
import traceback
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--soft-deadline',type=float,required=True)
parser.add_argument('--package-sha',required=True)
parser.add_argument('--attempt')
args = parser.parse_args()
RUN_ROOT = Path('/content/drive/MyDrive/ATML_PA3/task2/run_20261031_150949_1cff148a')
LOCAL_ROOT = Path('/content/atml_pa3_task2/run_20261031_150949_1cff148a')
DATASET_SHA = '82ad5006bd1d68ee54e639e50e7d835d8f6d97673537c38dab8d8eebdd8ac1b2'
FREEZE_SHA = 'fe7c9d4efa5d92d5b809eadb3695adcdd19f2421c6d43a2cb1593a7b86c3245b'
SHARDS_SHA = 'd3642cd07bf0f1ccd82cf6bd8c5e947d619237b8cd5860499d6f9972bbff3ad7'
SHARDS_RECIPE = 'e5683db7637de6712b8fc790f99ae71497c3c1e89ed5af4dcdd0d9b5b552b09a'
MODELS_SHA = '66f14f21d911fb57ae379df700ed3161cf348a997922235d8ab8b56491604f7f'
SIGLIP_ARCHIVE = '15f00ccd1f3da1ee91408e0581419d82ca36f3def6a06bc74a58dd32de9270f2'
PINS = {'transformers':'5.22.1','huggingface-hub':'1.33.0','tokenizers':'0.23.3',
    'safetensors':'0.8.0','Pillow':'11.3.0','numpy':'2.4.6'}
attempt = args.attempt or (datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
started = time.monotonic()
model = extractor = None
receipts = []
phase='worker startup'
phase_number=0

def report_phase(name):
    global phase,phase_number
    phase=name
    phase_number+=1
    print('WORKER PHASE:',name,flush=True)
    try:
        from task2.session_io import bounded_io,save_json
        with bounded_io(15,'Worker phase diagnostic'):
            save_json(RUN_ROOT/'logs'/f'block_11_phase_{attempt}_{phase_number:02d}.json',
                {'attempt':attempt,'phase':phase,'elapsed_seconds':round(time.monotonic()-started,3)})
    except BaseException as error:
        print('Phase diagnostic could not be saved:',repr(error),flush=True)

# Dependency setup deliberately excludes PyTorch, CUDA and torchvision packages.
# It runs in this fresh process so no old numerical modules remain imported.
def pinned_packages():
    versions = {}
    for name in PINS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if versions == PINS:
        return 'Existing exact CPU-observed library versions reused'
    tag = hashlib.sha256(json.dumps(PINS,sort_keys=True).encode()).hexdigest()[:16]
    target = LOCAL_ROOT/'gpu_python_packages'/tag
    marker = target/'complete.json'
    if not marker.exists():
        target.mkdir(parents=True,exist_ok=True)
        print('Preparing pinned vision libraries. The existing CUDA PyTorch installation is retained.',flush=True)
        subprocess.run([sys.executable,'-m','pip','install','--quiet','--no-deps','--upgrade',
            '--target',str(target),*[f'{name}=={version}' for name,version in PINS.items()]],check=True,timeout=240)
        marker.write_text(json.dumps(PINS,sort_keys=True))
    else:
        assert json.loads(marker.read_text()) == PINS
    sys.path.insert(0,str(target))
    assert {name:importlib.metadata.version(name) for name in PINS} == PINS
    return 'Pinned vision libraries loaded from an isolated local package directory'


try:
    report_phase('checking installed PyTorch metadata')
    if '+cpu' in importlib.metadata.version('torch'):
        raise RuntimeError('CPU-only PyTorch is installed in this GPU runtime. No model was loaded. Return this output for an environment repair.')
    report_phase('preparing pinned vision libraries')
    package_setup = pinned_packages()
    report_phase('importing CUDA and vision libraries')
    import numpy as np
    import torch
    import PIL
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import SiglipConfig, SiglipVisionModel
    from transformers.models.siglip.image_processing_pil_siglip import SiglipImageProcessorPil
    from transformers.image_processing_backends import PilBackend
    from task2.images import encoded,file_sha,immutable_bytes
    from task2.features import bounded,stage_file,unpack_model,FinalNormExtractor,FeatureStore
    from task2.staging import image_bytes_from_archive
    from task2.vision_checks import validate_siglip_processor
    report_phase('checking the CUDA device')
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError('This runtime does not have working CUDA PyTorch. No model was loaded.')
    capability = torch.cuda.get_device_capability(0)
    assert capability[0] >= 7, 'Use a T4 or newer CUDA GPU for this cache block.'
    assert shutil.disk_usage(LOCAL_ROOT).free >= 8*1024**3, 'At least 8 GiB local free space is required.'
    torch.manual_seed(6304)
    torch.cuda.manual_seed_all(6304)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision('highest')
    environment = {'python':platform.python_version(), 'torch':torch.__version__, 'cuda':torch.version.cuda,
        'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0),'capability':list(capability),
        'nvidia_driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,timeout=10).strip(),
        'gpu_total_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        **{name:importlib.metadata.version(name) for name in PINS}}
    print('CUDA environment verified.',json.dumps(environment,sort_keys=True),flush=True)
    report_phase('verifying prerequisite manifests')
    dataset_dir = RUN_ROOT/'datasets'/('task2_A012_'+DATASET_SHA[:16])
    with bounded(120,'Prerequisite receipt verification'):
        pointer = json.loads((RUN_ROOT/'manifests/active_task2_dataset.json').read_bytes())
        assert pointer['dataset_identity'] == DATASET_SHA and Path(pointer['dataset_dir']) == dataset_dir
        assert pointer['freeze_receipt_sha256'] == FREEZE_SHA and file_sha(dataset_dir/'freeze_receipt.json') == FREEZE_SHA
        ip = json.loads((RUN_ROOT/'manifests/active_task2_input_shards.json').read_bytes())
        shard_path = dataset_dir/'inputs/image_shards'/SHARDS_RECIPE[:16]/'input_shards_complete.json'
        assert Path(ip['manifest_path']) == shard_path and ip['manifest_sha256'] == SHARDS_SHA
        assert ip['dataset_identity'] == DATASET_SHA and file_sha(shard_path) == SHARDS_SHA
        shards = json.loads(shard_path.read_bytes())['shards']
        images = [image for shard in shards for image in shard['images']]
        assert len(shards) == 37 and len(images) == len({(r['source_split'],r['image_id']) for r in images}) == 9260
        assert all(r['dataset_identity'] == DATASET_SHA for r in images)
        assert Counter(r['source_split'] for r in images) == {'train2014':8733,'val2014':527}
        mp = json.loads((RUN_ROOT/'manifests/active_task2_model_assets.json').read_bytes())
        models_path = dataset_dir/'inputs/model_assets'/('models_complete_'+MODELS_SHA[:16]+'.json')
        assert Path(mp['manifest_path']) == models_path and mp['manifest_sha256'] == MODELS_SHA
        assert mp['dataset_identity'] == DATASET_SHA and file_sha(models_path) == MODELS_SHA
        model_assets = json.loads(models_path.read_bytes())
        vision_asset = model_assets['models']['siglip']
        assert vision_asset['repo_id'] == 'google/siglip-base-patch16-224'
        assert vision_asset['revision'] == '7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed'
        assert vision_asset['archive_sha256'] == SIGLIP_ARCHIVE
        archive_relative = Path(vision_asset['archive_relative_path'])
        assert archive_relative.name == str(archive_relative)
    print('Frozen dataset, original image archives and pinned model manifest verified.',flush=True)
    report_phase('staging the verified SigLIP originals')
    model_folder = LOCAL_ROOT/'gpu_siglip_originals'/vision_asset['revision']
    local_model_tar = model_folder.parent/('original_'+SIGLIP_ARCHIVE[:16]+'.tar')
    source_model_tar = RUN_ROOT/'model_assets/siglip'/vision_asset['revision']/archive_relative
    with bounded(180,'Local SigLIP archive staging'):
        stage_file(source_model_tar,local_model_tar,SIGLIP_ARCHIVE)
    with bounded(120,'Original SigLIP member verification'):
        unpack_model(local_model_tar,model_folder,vision_asset['files'])
    report_phase('loading the frozen vision encoder')
    processor = SiglipImageProcessorPil.from_pretrained(str(model_folder),local_files_only=True)
    processor_validation=validate_siglip_processor(processor,json.loads((model_folder/'preprocessor_config.json').read_bytes()))
    print('SigLIP processor values verified.',json.dumps(processor_validation,sort_keys=True),flush=True)
    config = SiglipConfig.from_json_file(str(model_folder/'config.json'))
    assert (config.vision_config.hidden_size,config.vision_config.image_size,config.vision_config.patch_size) == (768,224,16)
    config.vision_config._attn_implementation = 'eager'
    model = SiglipVisionModel(config.vision_config)
    assert model.config._attn_implementation == 'eager'
    original = load_file(str(model_folder/'model.safetensors'),device='cpu')
    vision_state = {key.removeprefix('vision_model.'):value for key,value in original.items() if key.startswith('vision_model.')}
    assert vision_state and len(vision_state) == len(model.state_dict())
    model.load_state_dict(vision_state,strict=True)
    del original,vision_state
    model.requires_grad_(False)
    model.eval()
    model.to(device='cuda:0',dtype=torch.float16)
    assert all(p.dtype == torch.float16 and p.device.type == 'cuda' and not p.requires_grad for p in model.parameters())
    extractor = FinalNormExtractor(model)
    gc.collect()
    files = {row['file_name']:row for row in vision_asset['files']}
    recipe = {'schema':'atml-pa3-task2-siglip-feature-recipe-v1','dataset_identity':DATASET_SHA,
        'freeze_receipt_sha256':FREEZE_SHA,'input_shards_manifest_sha256':SHARDS_SHA,
        'model_manifest_sha256':MODELS_SHA,'vision_revision':vision_asset['revision'],
        'vision_archive_sha256':SIGLIP_ARCHIVE,'weight_file_sha256':files['model.safetensors']['sha256'],
        'processor_file_sha256':files['preprocessor_config.json']['sha256'],
        'feature_shape_per_image':[196,768],'feature_dtype':'float16','encoder_dtype':'float16',
        'image_conversion':'Decode original JPEG, then PIL RGB conversion; all source EXIF orientations are 1',
        'processor_class':'SiglipImageProcessorPil','processor_backend':'PIL bicubic resize to 224 x 224; pinned original rescale/normalize',
        'spatial_order':'Original encoder row-major 14 x 14 positions; no pooling or patch removal',
        'final_norm':'Use returned last_hidden_state, equal to one post_layernorm call; no extra normalization',
        'attention':'eager','inference_batch_size':32,'shard_image_limit':256,'npy_version':[2,0],
        'deterministic_algorithms':True,'tf32':False,'fp16_reduced_precision_reduction':False,
        'environment':environment,'implementation_sha256':{name:file_sha(inspect.getsourcefile(cls)) for name,cls in
            [('SiglipVisionModel',SiglipVisionModel),('SiglipImageProcessorPil',SiglipImageProcessorPil),('PilBackend',PilBackend)]}}
    store = FeatureStore(dataset_dir,LOCAL_ROOT,recipe)
    with bounded(120,'Feature recipe binding'):
        # One immutable binding prevents mixing resumed features from different
        # software versions, numerical backends or GPU types.
        immutable_bytes(dataset_dir/'features/siglip196_recipe.json',encoded(recipe))
        immutable_bytes(store.durable_dir/'recipe.json',encoded(recipe))

    def local_input(shard):
        relative = Path(shard['tar_relative_path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        destination = LOCAL_ROOT/'gpu_input_archives'/Path(relative).name
        with bounded(90,'Original image archive staging'):
            stage_file(dataset_dir/relative,destination,shard['tar_sha256'])
        return destination

    def rgb_image(record,content):
        with Image.open(io.BytesIO(content)) as image:
            image.load()
            metadata = record['decoded_metadata']
            assert image.size == (metadata['width'],metadata['height']) and image.mode == metadata['mode']
            assert image.getexif().get(274,1) == 1
            return image.convert('RGB')

    # Only training originals are used for numerical/interface diagnostics.
    report_phase('checking training-only vision probes')
    training = [row for row in images if row['source_split'] == 'train2014']
    chosen = training[:2]
    gray = next((row for row in training if row['decoded_metadata']['mode']=='L'),None)
    if gray is not None and gray not in chosen:
        chosen.append(gray)
    selected = {(row['source_split'],row['image_id']):row for row in chosen}
    probe_images = {}
    for shard in shards:
        if not any((row['source_split'],row['image_id']) in selected for row in shard['images']):
            continue
        path = local_input(shard)
        for row,content in image_bytes_from_archive(path,shard['images']):
            key = (row['source_split'],row['image_id'])
            if key in selected:
                probe_images[key] = rgb_image(row,content)
    ordered_probe = [probe_images[(r['source_split'],r['image_id'])] for r in chosen]
    cpu_pixels = processor(images=ordered_probe,return_tensors='pt')['pixel_values']
    repeated_pixels = processor(images=ordered_probe,return_tensors='pt')['pixel_values']
    assert tuple(cpu_pixels.shape) == (len(chosen),3,224,224) and torch.equal(cpu_pixels,repeated_pixels)
    assert torch.isfinite(cpu_pixels).all() and cpu_pixels.min() >= -1 and cpu_pixels.max() <= 1
    if gray is not None:
        gray_index = chosen.index(gray)
        assert torch.equal(cpu_pixels[gray_index,0],cpu_pixels[gray_index,1]) and torch.equal(cpu_pixels[gray_index,1],cpu_pixels[gray_index,2])
    pixels = cpu_pixels.to('cuda:0',dtype=torch.float16)
    embeddings_seen = []
    hook = model.embeddings.register_forward_hook(lambda module,inputs,output:embeddings_seen.append(output.detach().clone()))
    try:
        first = extractor(pixels)
    finally:
        hook.remove()
    assert len(embeddings_seen) == 1
    with torch.inference_mode():
        expected_embeddings = model.embeddings.patch_embedding(pixels).flatten(2).transpose(1,2)
        expected_embeddings += model.embeddings.position_embedding(model.embeddings.position_ids)
    assert torch.equal(expected_embeddings,embeddings_seen[0])
    second = extractor(pixels)
    torch.cuda.synchronize()
    assert torch.equal(first,second), 'Repeated frozen inference changed its output.'
    probe = {'status':'passed','training_image_ids':[r['image_id'] for r in chosen],
        'gray_training_image_checked':gray is not None,'pixel_repeat_exact':True,'encoder_repeat_exact':True,
        'row_major_position_order_checked':True,'final_norm_calls_per_forward':1,
        'returned_patches_equal_final_norm':True,'shape':list(first.shape),'dtype':str(first.dtype),
        'probe_feature_sha256':hashlib.sha256(first.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        'encoder_parameters':sum(p.numel() for p in model.parameters()),'all_encoder_parameters_frozen':True}
    with bounded(120,'Probe evidence save'):
        immutable_bytes(store.durable_dir/'probe.json',encoded(probe))
    del first,second,pixels,cpu_pixels,repeated_pixels,expected_embeddings,embeddings_seen,ordered_probe,probe_images
    gc.collect()
    torch.cuda.empty_cache()
    initialization_peak_allocated = torch.cuda.max_memory_allocated()
    initialization_peak_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    print('Vision interface passed. Final norm exactly once, row-major 196 patches, deterministic preprocessing and frozen FP16 inference.',flush=True)
    report_phase('caching verified feature archives')
    activities = Counter()
    compute_seconds = preprocessing_seconds = staging_seconds = save_seconds = 0.0
    computed_images = 0
    for shard in shards:
        index,rows = shard['index'],shard['images']
        with bounded(120,'Existing feature receipt verification'):
            previous = store.reuse(index,rows)
        if previous is not None:
            receipts.append(previous)
            activities['reused'] += 1
            print(f'Feature archives verified {len(receipts)}/37 | images {sum(r["image_count"] for r in receipts)}/9260 | reused',flush=True)
            continue
        if time.monotonic() >= args.soft_deadline:
            print('Reached the bounded session limit. Completed feature archives are saved for the next run.',flush=True)
            break
        stamp = time.monotonic()
        path = local_input(shard)
        staging_seconds += time.monotonic()-stamp
        result_array = np.empty((len(rows),196,768),dtype='<f2')
        batch,offset = [],0
        def infer_batch():
            global preprocessing_seconds,compute_seconds,offset,computed_images
            begin = time.monotonic()
            batch_pixels = processor(images=batch,return_tensors='pt')['pixel_values'].to('cuda:0',dtype=torch.float16)
            torch.cuda.synchronize()
            preprocessing_seconds += time.monotonic()-begin
            start_event,end_event = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start_event.record()
            features = extractor(batch_pixels)
            end_event.record()
            end_event.synchronize()
            compute_seconds += start_event.elapsed_time(end_event)/1000
            result_array[offset:offset+len(batch)] = features.detach().cpu().numpy()
            offset += len(batch)
            computed_images += len(batch)
        for row,content in image_bytes_from_archive(path,rows):
            batch.append(rgb_image(row,content))
            if len(batch) == 32:
                infer_batch()
                batch.clear()
        if batch:
            infer_batch()
            batch.clear()
        assert offset == len(rows)
        stamp = time.monotonic()
        with bounded(120,'Feature shard save and durable readback'):
            receipt = store.save(index,rows,result_array)
        save_seconds += time.monotonic()-stamp
        del result_array
        receipts.append(receipt)
        activities['computed'] += 1
        print(f'Feature archives verified {len(receipts)}/37 | images {sum(r["image_count"] for r in receipts)}/9260 | saved | elapsed {time.monotonic()-started:.1f}s',flush=True)
    complete = len(receipts) == 37
    feature_count = sum(row['image_count'] for row in receipts)
    feature_manifest_path = feature_manifest_sha = None
    if complete:
        assert [image for row in receipts for image in row['images']] == images and feature_count == 9260
        manifest = {'schema':'atml-pa3-task2-features-complete-v1','status':'passed','dataset_identity':DATASET_SHA,
            'recipe_sha256':store.recipe_sha,'recipe':recipe,'probe':probe,'image_count':9260,'shard_count':37,'shards':receipts}
        feature_manifest_path = store.durable_dir/'features_complete.json'
        with bounded(120,'Complete feature manifest save'):
            immutable_bytes(feature_manifest_path,encoded(manifest))
            feature_manifest_sha = file_sha(feature_manifest_path)
            immutable_bytes(RUN_ROOT/'manifests/active_task2_features.json',encoded({
                'schema':'atml-pa3-task2-active-features-v1','dataset_identity':DATASET_SHA,'recipe_sha256':store.recipe_sha,
                'manifest_path':str(feature_manifest_path),'manifest_sha256':feature_manifest_sha,'image_count':9260,'shard_count':37}))
    result = {'block':'11','status':'passed' if complete else 'checkpointed','dataset_identity':DATASET_SHA,
        'feature_cache_complete':complete,'images_verified':feature_count,'shards_verified':len(receipts),
        'run_activities':dict(activities),'recipe_sha256':store.recipe_sha,
        'feature_manifest_path':str(feature_manifest_path) if complete else None,'feature_manifest_sha256':feature_manifest_sha,
        'feature_bytes':sum(row['feature_bytes'] for row in receipts),'probe':probe,'environment':environment,
        'package_setup':package_setup,'cache_package_sha256':args.package_sha,
        'elapsed_seconds':round(time.monotonic()-started,3),'computed_images_this_run':computed_images,
        'compute_seconds_cuda_events':round(compute_seconds,3),'preprocessing_and_transfer_seconds':round(preprocessing_seconds,3),
        'input_archive_staging_seconds':round(staging_seconds,3),'save_and_verification_seconds':round(save_seconds,3),
        'initialization_peak_gpu_allocated_bytes':initialization_peak_allocated,
        'cache_peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_gpu_allocated_bytes':max(initialization_peak_allocated,torch.cuda.max_memory_allocated()),
        'peak_gpu_reserved_bytes':max(initialization_peak_reserved,torch.cuda.max_memory_reserved()),
        'gpu_used':True,'qwen_loaded':False,'training_started':False,'gpu_release_requested_by_controller_on_exit':True}
    with bounded(120,'Session result save'):
        immutable_bytes(RUN_ROOT/f'logs/block_11_{attempt}.json',encoded(result))
    print('BLOCK 11 PASSED' if complete else 'BLOCK 11 CHECKPOINTED',flush=True)
    print(json.dumps(result,indent=2),flush=True)
except BaseException:
    failure = {'block':'11','status':'failed','dataset_identity':DATASET_SHA,'traceback':traceback.format_exc(),
        'phase':phase,
        'verified_shard_results_this_run':len(receipts),'cache_package_sha256':args.package_sha,'training_started':False,
        'gpu_release_requested_by_controller_on_exit':True}
    print(json.dumps(failure,indent=2),flush=True)
    try:
        from task2.session_io import bounded_io,save_json
        with bounded_io(30,'Failure diagnostic save'):
            save_json(RUN_ROOT/f'logs/block_11_failed_{attempt}.json',failure)
    except BaseException as diagnostic_error:
        print('Failure diagnostic could not be saved:',repr(diagnostic_error),flush=True)
    raise
finally:
    if extractor is not None:
        extractor.close()
    if model is not None:
        del model
    gc.collect()
