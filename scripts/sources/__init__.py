"""Per-source-type loader helpers for the batch standardization pipeline.

Submodules:

* ``github`` — GitHub URL parsing (the actual ``git clone`` invocation
  still lives in ``run_batch_standardization.discover_source_files``
  because it interleaves with the cache-marker bookkeeping).
* ``osf`` — Open Science Framework JSON API traversal and file download.

Web-dataset materialization stays in ``run_batch_standardization`` for
now because it depends on so many ``http_utils`` primitives that the
indirection would not buy clarity.
"""
