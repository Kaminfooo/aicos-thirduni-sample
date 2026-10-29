#!/usr/bin/env python3
"""Create a consistent AICOS operational backup without private signing keys.

The SQLite database is copied with the online backup API. Evidence media and
selected public configuration are archived with SHA-256 checksums. Private keys
and .env secrets are excluded by design.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sqlite3, tarfile, tempfile, time
from pathlib import Path


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--db',default=os.environ.get('DB_PATH','aicos.db'))
    ap.add_argument('--snapshots',default=os.environ.get('SNAPSHOT_DIR','snapshots'))
    ap.add_argument('--out',default='')
    args=ap.parse_args()
    db=Path(args.db); snaps=Path(args.snapshots)
    if not db.is_file(): raise SystemExit(f'database not found: {db}')
    stamp=time.strftime('%Y%m%d-%H%M%S',time.gmtime())
    out=Path(args.out or f'AICOS-BACKUP-{stamp}.tar.gz')
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)/'AICOS-BACKUP'; root.mkdir()
        copied=root/'aicos.db'
        src=sqlite3.connect(str(db)); dst=sqlite3.connect(str(copied))
        src.backup(dst); dst.close(); src.close()
        manifest={'created_at':time.time(),'database':'aicos.db','files':{}}
        manifest['files']['aicos.db']=sha(copied)
        if snaps.is_dir():
            mdir=root/'evidence-media'; mdir.mkdir()
            for p in snaps.iterdir():
                if p.is_file() and p.suffix.lower() in {'.jpg','.gif'}:
                    target=mdir/p.name; target.write_bytes(p.read_bytes())
                    manifest['files'][f'evidence-media/{p.name}']=sha(target)
        (root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        (root/'README.txt').write_text('AICOS operational backup. Private signing keys and .env credentials are intentionally excluded.\n',encoding='utf-8')
        with tarfile.open(out,'w:gz') as tf: tf.add(root,arcname='AICOS-BACKUP')
    print(out)
if __name__=='__main__': main()
