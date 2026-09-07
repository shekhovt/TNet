# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.

# --- What this file is -----------------------------------------------------------------------
# The test suite for the energy package. It is CPU arithmetic on shapes: no data, no checkpoint,
# no GPU, and about half a minute end to end.

"""Tests for the energy package.

Runnable on their own -- ``pytest TNet/energy_ECML/tests`` -- because this computation has no data,
no checkpoints and no GPU.  A few seconds, so it can be run on every edit.
"""
