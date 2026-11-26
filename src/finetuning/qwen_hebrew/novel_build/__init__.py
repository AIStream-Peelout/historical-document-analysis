"""Row builder for the novel-only training set: one page row per manuscript side of never-trained documents.

The novelty ledger decides what is novel; this package only turns novel candidates into trainable rows and
reports a funnel. Stage files are JSON lines of the dataclasses in :mod:`.schema`.
"""
