# SPDX-License-Identifier: CC BY-NC-SA 4.0
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. Making code open source is essential for scientific
# transparency and reproducibility. Providing access to the code allows researchers to
# verify, replicate, and expand upon results, which supports scientific progress. Current
# top-tier conferences encourage opensource code for the purpose of reproducibility that
# is de-facto an established expectation of the research community. Opensource code as
# part of publication should not affect the performance of an invention both from a
# commercial and IP point of view.
import os
import os, getpass, tempfile
user = getpass.getuser()
local_cache = os.path.join(tempfile.gettempdir(), f"{user}/torch_inductor_cache")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = local_cache

import torch

import torch._inductor.config as inductor_config
from torch._dynamo.decorators import mark_unbacked
import torch._dynamo

import logging
from contextlib import contextmanager

@contextmanager
def dynamo_debug(enable=True):
    if enable:
        torch._logging.set_logs(
                    dynamo=logging.WARNING,
                    inductor=logging.WARNING,
                    guards=False,        # show guard logs
                    output_code=True    # log generated backend code
                )
    try:
        yield
    finally:
        torch._logging.set_logs(
            dynamo=logging.WARNING,
            inductor=logging.WARNING,
            guards=False,
            output_code=False
        )

torch._logging.set_logs(
            dynamo=logging.WARNING,
            inductor=logging.WARNING,
            guards=False,        # show guard logs
            output_code=False    # log generated backend code
        )

TORCHINDUCTOR_BENCHMARK_GEMM=1

compile_args = dict(fullgraph=True, dynamic = True, backend = "inductor", options={'group_fusion':True, 'force_same_precision':False, 'disable_cpp_codegen':False, 'trace.graph_diagram':True, "triton.cudagraphs": False})

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.inline_inbuilt_nn_modules = True
torch._dynamo.config.force_parameter_static_shapes = False
torch._dynamo.config.cache_size_limit = 20
torch._dynamo.config.guard_nn_modules = False
