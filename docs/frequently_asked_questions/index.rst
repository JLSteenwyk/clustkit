.. _faq:


FAQ
===

**What types of sequences does ClustKIT support?**

ClustKIT currently supports protein sequences in FASTA or FASTQ format.

|

**How does ClustKIT differ from CD-HIT or MMseqs2?**

CD-HIT and MMseqs2 use greedy, centroid-based clustering: a sequence either joins an
existing cluster or becomes a new centroid. This works well at high identity thresholds
but loses sensitivity at low thresholds (30-50%) because results depend on sequence
processing order. ClustKIT builds a sparse similarity graph from Smith-Waterman alignments
and partitions it with Leiden community detection, which optimizes a global modularity
objective. This produces well-connected clusters regardless of input order.

|

**When should I use Leiden vs. connected components vs. greedy?**

Leiden (default) is recommended for low-to-medium identity thresholds (t <= 0.5) where it
significantly outperforms greedy methods. At high identity thresholds (t >= 0.7), all three
methods perform comparably. Connected components is the fastest but may merge distinct
families through transitive connections. Greedy is familiar if you are transitioning from
CD-HIT.

|

**Do I need a GPU?**

No. GPU acceleration is optional and provides the greatest benefit for large datasets at
low identity thresholds, where the alignment phase dominates runtime. For most use cases,
multi-threaded CPU execution is sufficient.

|

**How much memory does ClustKIT require?**

Memory usage depends primarily on the number of candidate pairs after LSH filtering, not
the raw number of sequences. For a dataset of 500,000 sequences at t = 0.7, peak memory
is typically under 8 GB with default settings.

|

**I am having trouble installing ClustKIT, what should I do?**

Please install ClustKIT using a virtual environment as directed in the installation instructions.
If you are still running into issues after installing in a virtual environment, please open an
issue on `GitHub <https://github.com/JLSteenwyk/ClustKIT/issues>`_ or contact the developer
via `email <https://jlsteenwyk.com/contact.html>`_.

^^^^^
