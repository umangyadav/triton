from pathlib import Path

import pytest
import triton
import triton.language as tl
from triton import knobs
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from triton._internal_testing import is_hip

if not is_hip():
    pytest.skip(allow_module_level=True)

ATTN_FWD_TTIR = str(Path(__file__).parent / "attn_fwd.ttir")


def test_llvm_fn_attrs_set_single_llir_attribute():
    target = triton.runtime.driver.active.get_current_target()

    baseline = triton.compile(ATTN_FWD_TTIR, target=target)
    with_attrs = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"},
    )

    assert '"amdgpu-sched-strategy"="iterative-ilp"' not in baseline.asm["llir"]
    assert '"amdgpu-sched-strategy"="iterative-ilp"' in with_attrs.asm["llir"]


def test_llvm_fn_attrs_set_multiple_llir_attributes():
    target = triton.runtime.driver.active.get_current_target()
    llvm_fn_attrs = ",".join([
        "amdgpu-sched-strategy=iterative-ilp",
        "triton-test-attr=enabled",
        "triton-bare-attr",
    ])

    with_attrs = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": llvm_fn_attrs},
    )

    assert '"amdgpu-sched-strategy"="iterative-ilp"' in with_attrs.asm["llir"]
    assert '"triton-test-attr"="enabled"' in with_attrs.asm["llir"]
    assert '"triton-bare-attr"' in with_attrs.asm["llir"]
    assert '"triton-bare-attr"=' not in with_attrs.asm["llir"]


def test_llvm_fn_attrs_change_amdgcn():
    target = triton.runtime.driver.active.get_current_target()

    baseline = triton.compile(ATTN_FWD_TTIR, target=target)
    with_sched_attr = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_fn_attrs": "amdgpu-sched-strategy=iterative-ilp"},
    )

    assert baseline.asm["amdgcn"] != with_sched_attr.asm["amdgcn"]


def test_llvm_fn_attrs_override_compiler_attributes(tmp_path):
    target = triton.runtime.driver.active.get_current_target()

    with knobs.amd.scope(), knobs.cache.scope():
        knobs.cache.dir = str(tmp_path)
        knobs.amd.use_expert_scheduling = True
        # Expert scheduling mode is applied after linking device libraries.
        baseline = triton.compile(ATTN_FWD_TTIR, target=target)
        with_attrs = triton.compile(
            ATTN_FWD_TTIR,
            target=target,
            options={"llvm_fn_attrs": "amdgpu-expert-scheduling-mode=false"},
        )

    assert '"amdgpu-expert-scheduling-mode"="true"' in baseline.asm["llir"]
    assert '"amdgpu-expert-scheduling-mode"="false"' in with_attrs.asm["llir"]


@triton.jit
def _clamp_kernel(X, Y, USE_ASM: tl.constexpr):
    offs = tl.arange(0, 64)
    x = tl.load(X + offs)
    if USE_ASM:
        x = tl.inline_asm_elementwise("v_min_f32 $0, $1, $2", "=v,v,v", [x, 1.0], dtype=tl.float32, is_pure=True,
                                      pack=1)
    else:
        x = tl.minimum(x, 1.0)
    tl.store(Y + offs, x * 2.0)


def test_expert_scheduling_inline_asm_disabled(tmp_path):
    # Expert scheduling does not track VGPRs written by inline assembly, so it
    # stays off by default for kernels with inline asm on gfx1250.
    target = GPUTarget("hip", "gfx1250", 32)
    signature = {"X": "*fp32", "Y": "*fp32", "USE_ASM": "constexpr"}

    def compile_llir(use_asm):
        src = ASTSource(fn=_clamp_kernel, signature=signature, constexprs={"USE_ASM": use_asm})
        return triton.compile(src, target=target).asm["llir"]

    with knobs.amd.scope(), knobs.cache.scope():
        knobs.cache.dir = str(tmp_path)
        knobs.amd.use_expert_scheduling = None
        assert '"amdgpu-expert-scheduling-mode"="true"' in compile_llir(False)
        assert "amdgpu-expert-scheduling-mode" not in compile_llir(True)
        knobs.amd.use_expert_scheduling = True
        assert '"amdgpu-expert-scheduling-mode"="true"' in compile_llir(True)


def test_llvm_cl_opts_set_bare_option():
    target = triton.runtime.driver.active.get_current_target()

    baseline = triton.compile(ATTN_FWD_TTIR, target=target)
    bare = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_cl_opts": "enable-gvn-hoist"},
    )

    assert baseline.asm["llir"] != bare.asm["llir"]


def test_llvm_cl_opts_set_option_value():
    target = triton.runtime.driver.active.get_current_target()

    # -misched takes precedence over the scheduler the target or a function
    # attribute selects, so it changes the schedule on every architecture.
    baseline = triton.compile(ATTN_FWD_TTIR, target=target)
    ilp_max = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_cl_opts": "misched=ilpmax"},
    )
    ilp_min = triton.compile(
        ATTN_FWD_TTIR,
        target=target,
        options={"llvm_cl_opts": "misched=ilpmin"},
    )

    assert baseline.asm["amdgcn"] != ilp_max.asm["amdgcn"]
    assert baseline.asm["amdgcn"] != ilp_min.asm["amdgcn"]
    assert ilp_max.asm["amdgcn"] != ilp_min.asm["amdgcn"]
