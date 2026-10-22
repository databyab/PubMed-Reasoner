"""Download verified NLM MeSH XML files and preserve provenance."""
import hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import requests

# Verified NLM release pattern for annual XML descriptor/qualifier archives; server returns errors explicitly.
BASE="https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/"

def download_mesh(year: int, output_dir: str | Path):
    dest=Path(output_dir); dest.mkdir(parents=True, exist_ok=True)
    files={"descriptor":f"desc{year}.xml", "qualifier":f"qual{year}.xml"}
    manifest={"release_year":year,"downloaded_at":datetime.now(timezone.utc).isoformat(),"files":{}}
    targets=[dest/name for name in files.values()]
    existing=[str(p) for p in targets if p.exists()]
    if existing: raise FileExistsError(f"Refusing to overwrite existing MeSH release files: {', '.join(existing)}")
    try:
        for kind, name in files.items():
            url=BASE+name; target=dest/name; temp=target.with_suffix(target.suffix+".part")
            response=requests.get(url, timeout=90, stream=True); response.raise_for_status()
            digest=hashlib.sha256()
            with temp.open("wb") as f:
                for chunk in response.iter_content(1024*1024):
                    if chunk: f.write(chunk); digest.update(chunk)
            manifest["files"][kind]={"filename":name,"url":url,"sha256":digest.hexdigest()}
        # Verify XML before publishing either completed file.
        import xml.etree.ElementTree as ET
        for name in files.values(): ET.parse(dest/(name+".part"))
        for name in files.values(): (dest/(name+".part")).replace(dest/name)
    except Exception:
        for name in files.values():
            part=dest/(name+".part")
            if part.exists(): part.unlink()
        raise
    p=dest/"source_manifest.json"; tmp=p.with_suffix(".tmp"); tmp.write_text(json.dumps(manifest,indent=2),encoding="utf8"); tmp.replace(p)
    return manifest

