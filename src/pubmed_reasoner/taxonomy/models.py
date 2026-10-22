from pydantic import BaseModel, Field
class MeshDescriptor(BaseModel):
    descriptor_id: str
    name: str
    entry_terms: list[str] = Field(default_factory=list)
    tree_numbers: list[str] = Field(default_factory=list)
    qualifiers: list[str] = Field(default_factory=list)
    status: str | None = None

