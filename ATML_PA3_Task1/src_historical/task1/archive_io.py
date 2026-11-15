"""Verified archive helpers extracted unchanged from the previously used restorer.

No GPU controller or legacy completion logic is imported here.
"""
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time
import uuid
from pathlib import Path,PurePosixPath



def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def safe_relative(value):
    p=PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or not p.parts or str(p)!=value:
        raise RuntimeError('Unsafe archive/manifest path: '+value)
    return p


def restore(archive_path,destination,entries,heartbeat=lambda phase:None):
    """No extractall: validate each header, path, link and streamed file hash."""
    destination=Path(destination)
    if destination.is_symlink():raise RuntimeError('Linked extraction root')
    destination.mkdir(parents=True,exist_ok=True)
    expected={entry['path']:entry for entry in entries}
    if len(expected)!=len(entries):raise RuntimeError('Duplicate inventory paths')
    for rel,entry in expected.items():
        safe_relative(rel)
        if entry.get('kind','file')=='symlink':
            target=Path(entry['target'])
            resolved=(target if target.is_absolute() else (destination/rel).parent/target).resolve()
            if not resolved.is_relative_to(destination.resolve()):raise RuntimeError('External software symlink: '+rel)
    seen=set()
    with tarfile.open(archive_path,'r:*') as archive:
        for number,member in enumerate(archive,1):
            safe_relative(member.name)
            if member.name not in expected or member.name in seen:raise RuntimeError('Unexpected/duplicate archive entry')
            seen.add(member.name);entry=expected[member.name];kind=entry.get('kind','file')
            target=destination/member.name
            if not target.parent.resolve().is_relative_to(destination.resolve()):raise RuntimeError('Archive parent escapes root')
            target.parent.mkdir(parents=True,exist_ok=True)
            if kind=='directory':
                if not member.isdir() or target.is_symlink() or (target.exists() and not target.is_dir()):raise RuntimeError('Directory mismatch')
                target.mkdir(exist_ok=True)
            elif kind=='symlink':
                if not member.issym() or member.linkname!=entry['target']:raise RuntimeError('Symlink mismatch')
                if target.is_symlink():
                    if os.readlink(target)!=entry['target']:raise RuntimeError('Existing software link differs')
                elif target.exists():raise RuntimeError('Existing non-link at software symlink path')
                else:target.symlink_to(entry['target'])
            elif kind=='file':
                if not member.isfile() or member.size!=entry['bytes']:raise RuntimeError('File type/size mismatch')
                if target.is_symlink():raise RuntimeError('Unexpected linked output')
                if target.exists():
                    if not target.is_file() or digest(target)!=entry['sha256']:raise RuntimeError('Existing file differs: '+str(target))
                else:
                    pending=target.with_name(target.name+'.pending_'+uuid.uuid4().hex);h=hashlib.sha256()
                    with archive.extractfile(member) as stream,pending.open('xb') as out:
                        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block);out.write(block)
                    if h.hexdigest()!=entry['sha256']:raise IOError('Extracted content hash mismatch')
                    pending.chmod(member.mode&0o777)
                    if target.exists():raise RuntimeError('Concurrent extraction detected')
                    pending.rename(target)
            else:raise RuntimeError('Unsupported inventory type')
            if number%250==0:heartbeat('restore '+str(number)+'/'+str(len(entries)))
    if seen!=set(expected):raise RuntimeError('Incomplete archive coverage')
    heartbeat('archive fully restored and checked')

