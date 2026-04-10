.. _tutorials:


Tutorials
=========

^^^^^

The following tutorials demonstrate common ClustKIT workflows using real protein sequence data.

|

Tutorial 1: Basic clustering
-----------------------------

This tutorial walks through a basic clustering run at 50% identity.

.. code-block:: shell

	# Cluster proteins at 50% identity using 8 threads
	clustkit -i proteins.fasta -o results_50/ -t 0.5 --threads 8

This produces three output files:

- ``results_50/clusters.tsv`` -- Cluster assignments
- ``results_50/representatives.fasta`` -- Representative sequences
- ``results_50/run_info.json`` -- Run statistics

The ``clusters.tsv`` file is tab-separated with three columns:

.. code-block:: none

	sequence_id	cluster_id	is_representative
	seq_001	0	True
	seq_042	0	False
	seq_007	1	True
	...

|

Tutorial 2: Low-identity clustering
------------------------------------

Clustering at low identity thresholds (30-50%) is where ClustKIT provides the
greatest advantage over greedy methods like CD-HIT and Linclust. The Leiden
community detection algorithm produces well-connected clusters even when
sequence similarity is sparse.

.. code-block:: shell

	# Cluster at 30% identity with accurate mode
	clustkit -i proteins.fasta -o results_30/ -t 0.3 \
		--clustering-mode accurate --threads 8

For maximum sensitivity at low thresholds, use ``--clustering-mode accurate``.
This increases the sketch size and LSH sensitivity to find more candidate pairs.

|

Tutorial 3: GPU-accelerated clustering
---------------------------------------

For large datasets, GPU acceleration can significantly reduce alignment time.
This requires the ``clustkit[gpu]`` installation.

.. code-block:: shell

	# Install with GPU support
	pip install clustkit[gpu]

	# Cluster using a single GPU
	clustkit -i large_proteins.fasta -o results_gpu/ -t 0.3 \
		--device 0 --threads 8

The ``--device`` option accepts:

- ``cpu`` -- CPU only (default)
- ``auto`` -- Automatically benchmark CPU vs. GPU and pick the fastest
- ``0``, ``1``, etc. -- A specific GPU device ID

GPU acceleration provides the greatest speedup at low identity thresholds where
the alignment phase dominates runtime. At t = 0.3, a single GPU provides ~1.3x
speedup over 8 CPU threads, and dual GPUs provide ~2.7x speedup.

|

Tutorial 4: Comparing clustering methods
-----------------------------------------

ClustKIT supports three clustering methods. This tutorial compares them on the
same input.

.. code-block:: shell

	# Leiden community detection (default)
	clustkit -i proteins.fasta -o results_leiden/ -t 0.5 \
		--cluster-method leiden --threads 8

	# Connected components
	clustkit -i proteins.fasta -o results_cc/ -t 0.5 \
		--cluster-method connected --threads 8

	# Greedy centroid-based
	clustkit -i proteins.fasta -o results_greedy/ -t 0.5 \
		--cluster-method greedy --threads 8

**When to use each method:**

- **Leiden** (default): Best for low-to-medium identity thresholds (t <= 0.5).
  Optimizes a global modularity objective and avoids the chain-extension artifacts
  of greedy methods.
- **Connected components**: Useful when every pair above threshold should be in
  the same cluster. Fast, but may merge distinct families through transitive
  connections.
- **Greedy**: Familiar centroid-based approach similar to CD-HIT. Works well at
  high identity thresholds (t >= 0.7) where clusters are dense.

|

Tutorial 5: CD-HIT-compatible output
-------------------------------------

If your downstream pipeline expects CD-HIT-format ``.clstr`` files, use the
``--format cdhit`` option:

.. code-block:: shell

	clustkit -i proteins.fasta -o results_cdhit/ -t 0.5 \
		--format cdhit --threads 8

This produces output in the standard CD-HIT cluster format for compatibility
with existing workflows.

|

Tutorial 6: Choosing representative sequences
----------------------------------------------

ClustKIT offers three strategies for selecting representative sequences from
each cluster:

.. code-block:: shell

	# Longest sequence (default)
	clustkit -i proteins.fasta -o results/ --representative longest

	# Centroid (highest average similarity to cluster members)
	clustkit -i proteins.fasta -o results/ --representative centroid

	# Most connected (most edges in the similarity graph)
	clustkit -i proteins.fasta -o results/ --representative most_connected

The ``longest`` strategy is the default and matches the behavior of CD-HIT and
VSEARCH. The ``centroid`` strategy may be preferred when the representative
should be maximally similar to all cluster members. The ``most_connected``
strategy selects the sequence with the most pairwise relationships above
threshold.

^^^^^
