"""CLI helper tests."""

from clustkit.clustering_mode import resolve_clustering_mode


def test_balanced_mode_defaults():
    sketch_size, sensitivity = resolve_clustering_mode(
        "balanced", 0.4, None, None
    )
    assert sketch_size == 128
    assert sensitivity == "medium"


def test_accurate_mode_is_threshold_aware():
    low_t = resolve_clustering_mode("accurate", 0.3, None, None)
    high_t = resolve_clustering_mode("accurate", 0.4, None, None)

    assert low_t == (128, "high")
    assert high_t == (256, "high")


def test_fast_mode_defaults():
    sketch_size, sensitivity = resolve_clustering_mode("fast", 0.5, None, None)
    assert sketch_size == 128
    assert sensitivity == "low"


def test_explicit_values_override_mode_defaults():
    sketch_size, sensitivity = resolve_clustering_mode(
        "accurate", 0.4, 64, "medium"
    )
    assert sketch_size == 64
    assert sensitivity == "medium"
