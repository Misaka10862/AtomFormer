"""Representation choices shared by AtomFormer training, RL, and evaluation."""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RepresentationSpec:
    name: str
    target_key: str
    assembly_mode: str


_SPECS = {
    "rooted": RepresentationSpec(
        name="rooted",
        target_key="rooted_decoration_smiles",
        assembly_mode="assemble_single_attachment_rooted",
    ),
    "plain": RepresentationSpec(
        name="plain",
        target_key="plain_decoration_smiles",
        assembly_mode="assemble_single_attachment_plain",
    ),
}


def get_representation_spec(name: Optional[str] = None) -> RepresentationSpec:
    """Return a validated representation specification."""

    key = name or "rooted"
    try:
        return _SPECS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown representation {key!r}; expected rooted or plain.") from exc


def assemble_with_representation(
    representation: str,
    scaffold_smiles: str,
    decoration_smiles: str,
):
    """Assemble a generated decoration using the selected public routine."""

    from .assembly import assemble_single_attachment_plain, assemble_single_attachment_rooted

    spec = get_representation_spec(representation)
    if spec.assembly_mode == "assemble_single_attachment_plain":
        return assemble_single_attachment_plain(scaffold_smiles, decoration_smiles)
    return assemble_single_attachment_rooted(scaffold_smiles, decoration_smiles)
