from __future__ import annotations

import re


def normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")


def resolve_label(label: str, aliases: dict[str, str]) -> str:
    normalized = normalize_label(label)
    return aliases.get(normalized, normalized)


def build_alias_map(ontology: dict, class_map: dict[str, int]) -> dict[str, str]:
    if not isinstance(ontology, dict) or not isinstance(ontology.get("classes"), list):
        raise ValueError("Ontology must contain a classes list")
    normalized_classes = {normalize_label(name): class_id for name, class_id in class_map.items()}
    aliases: dict[str, str] = {}
    seen: set[str] = set()
    for item in ontology["classes"]:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise ValueError("Ontology classes must have a name")
        name = normalize_label(item["name"])
        if not name or name in seen or name not in normalized_classes:
            raise ValueError("Ontology classes do not match the detector class map")
        if item.get("id") != normalized_classes[name]:
            raise ValueError(f"Ontology class ID differs for {name}")
        seen.add(name)
        values = item.get("automatic_aliases", [])
        if not isinstance(values, list):
            raise ValueError(f"Invalid automatic aliases for {name}")
        for value in [name, *values]:
            if not isinstance(value, str) or not normalize_label(value):
                raise ValueError(f"Invalid automatic alias for {name}")
            normalized = normalize_label(value)
            if normalized in aliases and aliases[normalized] != name:
                raise ValueError(f"Ontology alias conflict: {normalized}")
            aliases[normalized] = name
    if seen != set(normalized_classes):
        raise ValueError("Ontology classes do not match the detector class map")
    return aliases
