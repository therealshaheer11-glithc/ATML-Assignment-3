# Repair 1: obtain and verify only the COCO archive; CPU only.
# Run in the same notebook. Do not rerun Blocks 1 or 2.
import urllib.request, zipfile, time, traceback

needed = ('ROOT', 'LOCAL', 'save_record', 'file_sha', 'copy_checked')
if any(name not in globals() for name in needed):
    raise RuntimeError('Notebook state changed. Stop and share this message.')
if os.environ.get('COLAB_GPU', '0') not in ('', '0'):
    raise RuntimeError('This repair needs accelerator None.')

EXPECTED_BYTES = 252872794
EXPECTED_SHA = '031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
CHECKSUM_SOURCE = ('https://raw.githubusercontent.com/tensorflow/datasets/master/'
                   'tensorflow_datasets/url_checksums/coco_captions.txt')
CANDIDATES = [
    'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip',
    'https://images.cocodataset.org/annotations/annotations_trainval2014.zip',
]
repair_id = datetime.now(ZoneInfo('Asia/Karachi')).strftime('%Y%m%d_%H%M%S_') + uuid.uuid4().hex[:8]
repair = {'attempt': repair_id, 'status': 'started', 'gpu_used': False,
          'original_failure': {'http_status': 404, 'url':
              'https://storage.googleapis.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'},
          'checksum_source': CHECKSUM_SOURCE, 'expected_bytes': EXPECTED_BYTES,
          'expected_sha256': EXPECTED_SHA, 'probes': [], 'downloaded_this_attempt': False}
local_zip = LOCAL / 'annotation_archives/coco2014.zip'
durable_zip = ROOT / 'raw_annotation_archives/coco2014.zip'
local_zip.parent.mkdir(parents=True, exist_ok=True)
source_lock = ROOT / 'configs/coco2014_source.json'

try:
    if source_lock.exists():
        source_record = json.loads(source_lock.read_text())
        selected = source_record['url']
        if selected not in CANDIDATES:
            raise RuntimeError('Unexpected saved COCO source.')
    else:
        selected = None
        for url in CANDIDATES:
            print('Checking HTTPS source:', url, flush=True)
            probe_record = {'url': url}
            try:
                request = urllib.request.Request(url, headers={'Range': 'bytes=0-3'})
                with urllib.request.urlopen(request, timeout=20) as response:
                    magic = response.read(4)
                    probe_record.update({'http_status': response.status,
                                         'resolved_url': response.geturl(),
                                         'zip_magic': magic.hex()})
                    if response.geturl().startswith('https://') and magic == b'PK\x03\x04':
                        selected = url
                repair['probes'].append(probe_record)
                if selected:
                    break
            except Exception as exc:
                probe_record['error'] = f'{type(exc).__name__}: {exc}'
                repair['probes'].append(probe_record)
        if selected is None:
            raise RuntimeError('Neither HTTPS source returned a ZIP header; see saved diagnostics.')
    repair['selected_url'] = selected

    if durable_zip.exists():
        copy_checked(durable_zip, local_zip)

    if not local_zip.exists():
        partial = local_zip.with_name(f'coco2014_repair_{repair_id}.zip.part')
        print('Downloading COCO annotations; expected size 252,872,794 bytes...', flush=True)
        received, last_update = 0, time.monotonic()
        with urllib.request.urlopen(selected, timeout=60) as response:
            if not response.geturl().startswith('https://'):
                raise RuntimeError('Unexpected non-HTTPS redirect.')
            with partial.open('xb') as out:
                while True:
                    chunk = response.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    received += len(chunk)
                    if received > EXPECTED_BYTES:
                        raise IOError('Response exceeded expected archive size.')
                    if time.monotonic() - last_update > 10:
                        print(f'  {received / 2**20:.1f} / {EXPECTED_BYTES / 2**20:.1f} MiB', flush=True)
                        last_update = time.monotonic()
        if received != EXPECTED_BYTES or file_sha(partial) != EXPECTED_SHA:
            raise IOError('Size/checksum mismatch. Partial file preserved; not accepted.')
        partial.rename(local_zip)
        repair['downloaded_this_attempt'] = True

    if local_zip.stat().st_size != EXPECTED_BYTES or file_sha(local_zip) != EXPECTED_SHA:
        raise IOError('Cached archive does not match the published checksum; preserving it.')
    with zipfile.ZipFile(local_zip) as z:
        bad = z.testzip()
        if bad:
            raise IOError(f'Archive CRC failed: {bad}')
        names = {Path(name).name for name in z.namelist()}
        required = {'captions_train2014.json', 'captions_val2014.json',
                    'instances_train2014.json', 'instances_val2014.json'}
        if not required.issubset(names):
            raise IOError('Required COCO 2014 annotation files are missing.')

    copy_checked(local_zip, durable_zip)
    if file_sha(durable_zip) != EXPECTED_SHA:
        raise IOError('Drive-copy checksum mismatch.')
    source_record = {'url': selected, 'archive_key': 'coco2014',
                     'expected_bytes': EXPECTED_BYTES, 'expected_sha256': EXPECTED_SHA,
                     'checksum_source': CHECKSUM_SOURCE}
    save_record('configs/coco2014_source.json', source_record)
    repair.update({'status': 'passed', 'sha256': EXPECTED_SHA,
                   'zip_crc_passed': True, 'required_members_present': True,
                   'drive_copy_readback_verified': True, 'drive_archive': str(durable_zip),
                   'remaining_annotation_archives_not_processed_by_this_repair': 6})
except Exception:
    repair['status'] = 'failed'
    repair['traceback'] = traceback.format_exc()
    save_record(f'logs/block_03_repair_{repair_id}.json', repair)
    print(json.dumps(repair, indent=2), flush=True)
    raise

log_path = f'logs/block_03_repair_{repair_id}.json'
digest = save_record(log_path, repair)
print('\nCOCO REPAIR COMPLETE — published checksum, ZIP CRC, and Drive readback passed.')
print(json.dumps(repair, indent=2))
print('Saved log:', ROOT / log_path)
print('Log SHA256:', digest)
print('Keep accelerator None. Paste the output; do not rerun the original Block 3.')
