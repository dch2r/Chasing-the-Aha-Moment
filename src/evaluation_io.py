"""Durable evaluation logs; no GPU dependencies."""
import json,os
from pathlib import Path

def read_rows(path, allowed_ids):
    path=Path(path)
    if not path.exists(): return {}
    data=path.read_bytes(); lines=data.splitlines(keepends=True); rows={}; offset=0
    for i,line in enumerate(lines):
        try: row=json.loads(line)
        except (ValueError, UnicodeDecodeError):
            # Recover only an unfinished final append, preserving the original bytes.
            if i!=len(lines)-1 or line.endswith(b'\n'): raise
            backup=path.with_name(path.name+'.interrupted')
            if backup.exists(): raise RuntimeError('An interrupted log backup already exists; inspect it before recovery.')
            backup.write_bytes(data)
            with path.open('r+b') as f:f.truncate(offset)
            break
        key=(row['label'],row['source_id'])
        if key[0] not in ('base','tuned') or key[1] not in allowed_ids or key in rows:
            raise ValueError('Unknown or duplicate evaluation row')
        rows[key]=row;offset+=len(line)
    # A complete JSON object without its newline can also be a valid final write.
    if rows and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as f:f.write(b'\n');f.flush();os.fsync(f.fileno())
    return rows

def append_row(path,row):
    with Path(path).open('a') as f:
        f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
