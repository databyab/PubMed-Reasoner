from .models import MeshDescriptor
def validate_descriptors(items: list[MeshDescriptor]) -> dict:
    ids=[d.descriptor_id for d in items]
    return {"count":len(items),"missing_ids":sum(not x for x in ids),"missing_names":sum(not d.name for d in items),"duplicate_ids":len(ids)-len(set(ids)),"valid":bool(items) and len(ids)==len(set(ids)) and all(ids) and all(d.name for d in items)}

