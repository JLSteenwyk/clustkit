.. _change_log:


Change log
==========

^^^^^

**0.1.0**

- Initial release
- Six-phase pipeline: MinHash sketching, LSH, banded Smith-Waterman alignment, graph construction, Leiden community detection, representative selection
- Clustering methods: Leiden (default), connected components, greedy
- Clustering mode presets: balanced, accurate, fast
- GPU-accelerated Smith-Waterman alignment via CuPy (optional)
- C/OpenMP alignment kernel with multi-threaded execution
- Output formats: TSV and CD-HIT-compatible
- Representative selection: longest, centroid, most_connected

^^^^^
