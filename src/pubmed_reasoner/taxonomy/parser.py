"""Parse NLM MeSH XML descriptor and qualifier records."""
import xml.etree.ElementTree as ET
from pathlib import Path
from .models import MeshDescriptor

def _text(node, path):
    found = node.find(path)
    return "".join(found.itertext()).strip() if found is not None else ""

def parse_mesh_xml(path: str | Path, kind: str = "descriptor") -> list[MeshDescriptor]:
    root = ET.parse(path).getroot()
    output = []
    for rec in root.findall(".//DescriptorRecord" if kind == "descriptor" else ".//QualifierRecord"):
        id_key, name_key = ("DescriptorUI", "DescriptorName/String") if kind == "descriptor" else ("QualifierUI", "QualifierName/String")
        ident, name = _text(rec, id_key), _text(rec, name_key)
        if not ident or not name: continue
        terms, trees = [], []
        for term in rec.findall(".//Term/String"):
            value = "".join(term.itertext()).strip()
            if value and value != name: terms.append(value)
        trees = [x.text.strip() for x in rec.findall(".//TreeNumber") if x.text and x.text.strip()]
        quals = [x.text.strip() for x in rec.findall(".//AllowableQualifier/QualifierReferredTo/QualifierName/String") if x.text]
        status = _text(rec, "RecordStatus") or None
        output.append(MeshDescriptor(descriptor_id=ident, name=name, entry_terms=list(dict.fromkeys(terms)), tree_numbers=trees, qualifiers=quals, status=status))
    return output

