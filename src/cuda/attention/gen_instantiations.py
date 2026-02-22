#!/usr/bin/env python3

# This file is run to generate the kernel instantiations for the attention kernels
import itertools
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# map from python to c++ types
DTYPE_MAP = {
    "fp16": "cute::half_t",
    "bf16": "cute::bfloat16_t",
}

BOOL_MAP = {
    False: "false",
    True: "true",
}

SINGLE_MHA_KERNEL_TEMPLATE = """
#include "llmx/cuda/attention/device/single_mha_kernel.cuh"  // IWYU pragma: export

namespace llmx {{

template void single_mha_attention_launcher</*ELEMENT=*/{ELEMENT},
                                                  /*HEAD_DIM=*/{HEAD_DIM},
                                                  /*EVEN_K=*/{EVEN_K},
                                                  /*ALIBI=*/{ALIBI},
                                                  /*SOFT_CAP=*/{SOFT_CAP},
                                                  /*LOCAL=*/{LOCAL}>(
    const MHAParams &, // mha_params
    cudaStream_t);     // stream

}}  // namespace llmx
"""


@dataclass
class SingleMHAKernel:
    dtype: str
    head_dim: int
    even_k: bool
    alibi: bool
    softcap: bool
    local: bool

    @property
    def template(self) -> str:
        return SINGLE_MHA_KERNEL_TEMPLATE.format(
            ELEMENT=DTYPE_MAP[self.dtype],
            HEAD_DIM=self.head_dim,
            EVEN_K=BOOL_MAP[self.even_k],
            ALIBI=BOOL_MAP[self.alibi],
            SOFT_CAP=BOOL_MAP[self.softcap],
            LOCAL=BOOL_MAP[self.local],
        )

    @property
    def filename(self) -> str:
        def to_str(val: bool) -> str:
            return "1" if val else "0"

        return f"single_mha_{self.dtype}_hd{self.head_dim}_ek{to_str(self.even_k)}_al{to_str(self.alibi)}_sc{to_str(self.softcap)}_lc{to_str(self.local)}.cu"


def gen_single_mha_kernels() -> Iterator[SingleMHAKernel]:
    for dtype, head_dim, even_k, alibi, soft_cap, local in itertools.product(
        ["fp16", "bf16"],  # dtype
        [64, 128, 256],    # head_dim
        [False, True],     # even_k
        [False, True],     # alibi
        [False, True],     # soft_cap
        [False, True],     # local
    ):
        yield SingleMHAKernel(
            dtype=dtype,
            head_dim=head_dim,
            even_k=even_k,
            alibi=alibi,
            softcap=soft_cap,
            local=local,
        )


if __name__ == "__main__":
    output_dir = Path(__file__).parent / "gensrc"
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for kernel in gen_single_mha_kernels():
        (output_dir / kernel.filename).write_text(kernel.template)
