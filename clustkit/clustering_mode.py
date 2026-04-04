"""Threshold-aware clustering mode defaults."""


def resolve_clustering_mode(
    clustering_mode: str,
    threshold: float,
    sketch_size: int | None = None,
    sensitivity: str | None = None,
) -> tuple[int, str]:
    """Resolve sketch size and sensitivity for a clustering mode."""
    mode = clustering_mode.lower()
    if mode not in {"balanced", "accurate", "fast"}:
        raise ValueError(
            "Clustering mode must be 'balanced', 'accurate', or 'fast'."
        )

    if mode == "balanced":
        resolved_sketch = 128
        resolved_sensitivity = "medium"
    elif mode == "accurate":
        resolved_sensitivity = "high"
        resolved_sketch = 128 if threshold <= 0.35 else 256
    else:
        resolved_sketch = 128
        resolved_sensitivity = "low"

    if sketch_size is not None:
        resolved_sketch = sketch_size
    if sensitivity is not None:
        resolved_sensitivity = sensitivity

    return resolved_sketch, resolved_sensitivity
