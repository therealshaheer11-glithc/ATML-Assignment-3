"""Reconstruct the original Block 7 archive using a separately obtained COCO image."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'reconstructed/Task_2_Block_07_CPU_Preflight.zip')
    args = parser.parse_args()
    spec = json.loads((ROOT / 'workflow/block07_archive.json').read_bytes())
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w') as z:
        for row in spec['members']:
            content = (args.image if row.get('external_image') else ROOT / row['source']).read_bytes()
            if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
                raise ValueError('Source differs from the original: ' + row['name'])
            info = zipfile.ZipInfo(row['name'], tuple(row['date_time']))
            info.compress_type = row['compress_type']
            info.external_attr = row['external_attr']
            info.create_system = row['create_system']
            z.writestr(info, content)
    content = archive.getvalue()
    actual = hashlib.sha256(content).hexdigest()
    if actual != spec['original_sha256']:
        raise ValueError('Reconstructed ZIP differs; check ZIP/zlib version. No output written.')
    output = args.output.resolve()
    if ROOT in output.parents and output.parts[len(ROOT.parts)] != 'reconstructed':
        raise ValueError('Keep the image-containing ZIP in reconstructed/ or outside this package.')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.read_bytes() != content:
            raise FileExistsError('Different existing output preserved: ' + str(output))
    else:
        output.write_bytes(content)
    print(json.dumps({'status': 'passed', 'archive': str(output), 'sha256': actual,
                      'note': 'Contains an original image; keep out of GitHub.'}, indent=2))


if __name__ == '__main__':
    main()
