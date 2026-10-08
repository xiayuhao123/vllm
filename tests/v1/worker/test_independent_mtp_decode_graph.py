# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.autoregressive import independent_cudagraph as icg
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as ars


def _config():
    return SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=False),
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            enforce_eager=False,
            enable_adaptive_verification=False,
            uses_dynamic_speculative_decoding=lambda: False,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["Qwen3_5MTP"])
            ),
        ),
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 512],
            max_cudagraph_capture_size=512,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=512, max_num_batched_tokens=8192),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        lora_config=None,
        kv_transfer_config=None,
        watermark_config=None,
        offload_config=SimpleNamespace(
            uva=SimpleNamespace(cpu_offload_gb=0),
            prefetch=SimpleNamespace(offload_group_size=0),
        ),
    )


@pytest.fixture(autouse=True)
def _sm89(monkeypatch):
    monkeypatch.setattr(icg.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(icg.current_platform, "get_device_capability", lambda: (8, 9))
    monkeypatch.setattr(icg.envs, "VLLM_ENABLE_SM89_MTP_DECODE_GRAPH", True)


def _resolve(
    config, backends=None, support=AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
):
    return icg.make_independent_mtp_decode_config(
        config,
        backend_names={"FLASHINFER"} if backends is None else backends,
        min_cg_support=support,
    )


def test_private_draft_config_keeps_target_piecewise():
    target = _config()
    draft, reason = _resolve(target)
    assert draft is not None, reason
    assert draft is not target
    assert draft.compilation_config is not target.compilation_config
    assert target.compilation_config.cudagraph_mode == CUDAGraphMode.PIECEWISE
    assert target.compilation_config.cudagraph_capture_sizes == [
        1,
        2,
        4,
        8,
        16,
        32,
        512,
    ]
    assert draft.compilation_config.cudagraph_mode == CUDAGraphMode.FULL_DECODE_ONLY
    assert draft.compilation_config.cudagraph_capture_sizes == list(range(1, 9))
    assert draft.compilation_config.max_cudagraph_capture_size == 8
    assert draft.model_config is target.model_config


@pytest.mark.parametrize(
    "field,value",
    [
        ("speculative_config.method", "dflash"),
        ("speculative_config.num_speculative_tokens", 1),
        ("speculative_config.enforce_eager", True),
        ("speculative_config.enable_adaptive_verification", True),
        ("model_config.enforce_eager", True),
        ("compilation_config.cudagraph_mode", CUDAGraphMode.NONE),
        ("compilation_config.cudagraph_mode", CUDAGraphMode.FULL_AND_PIECEWISE),
        ("compilation_config.cudagraph_capture_sizes", []),
        ("compilation_config.max_cudagraph_capture_size", 0),
        ("parallel_config.tensor_parallel_size", 4),
        ("parallel_config.pipeline_parallel_size", 2),
        ("parallel_config.data_parallel_size", 2),
        ("parallel_config.decode_context_parallel_size", 2),
        ("parallel_config.prefill_context_parallel_size", 2),
        ("parallel_config.enable_dbo", True),
        ("lora_config", object()),
        ("kv_transfer_config", object()),
        ("watermark_config", object()),
        ("offload_config.uva.cpu_offload_gb", 1),
        ("offload_config.prefetch.offload_group_size", 1),
        (
            "speculative_config.draft_model_config.hf_config.architectures",
            ["Qwen3_5MoeMTP"],
        ),
        ("speculative_config.uses_dynamic_speculative_decoding", lambda: True),
    ],
)
def test_ineligible_configs_keep_the_original_path(field, value):
    config = _config()
    owner = config
    parts = field.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    setattr(owner, parts[-1], value)
    draft, reason = _resolve(config)
    assert draft is None
    assert reason


@pytest.mark.parametrize("capability", [(8, 6), (9, 0), None])
def test_other_gpus_keep_original_path(monkeypatch, capability):
    monkeypatch.setattr(
        icg.current_platform, "get_device_capability", lambda: capability
    )
    assert _resolve(_config())[0] is None


def test_flag_off_keeps_original_path(monkeypatch):
    monkeypatch.setattr(icg.envs, "VLLM_ENABLE_SM89_MTP_DECODE_GRAPH", False)
    assert _resolve(_config())[0] is None


@pytest.mark.parametrize("backends", [{"FLASH_ATTN"}, set(), {"FLASHINFER", "MAMBA"}])
def test_other_backends_keep_original_path(backends):
    assert _resolve(_config(), backends)[0] is None


def test_no_graph_attention_support_keeps_original_path():
    assert _resolve(_config(), support=AttentionCGSupport.NEVER)[0] is None


def test_scheduler_and_capture_limits_are_respected():
    config = _config()
    config.scheduler_config.max_num_seqs = 4
    config.compilation_config.max_cudagraph_capture_size = 3
    draft, _ = _resolve(config)
    assert draft.compilation_config.cudagraph_capture_sizes == [1, 2, 3]


@pytest.mark.parametrize(
    "requests,expected", [(4, CUDAGraphMode.FULL), (3, CUDAGraphMode.NONE)]
)
def test_padded_draft_graph_falls_back_to_actual_batch(requests, expected):
    desc = BatchExecutionDescriptor(CUDAGraphMode.FULL, 4, 4, uniform_token_count=1)
    result = icg.exact_draft_graph_batch(desc, requests)
    assert result.cg_mode == expected
    assert result.num_reqs == requests
    assert result.num_tokens == requests


@pytest.mark.parametrize("fixed,prefills", [(False, 0), (True, 1)])
def test_non_graph_or_prefill_metadata_is_rejected_before_replay(fixed, prefills):
    metadata = SimpleNamespace(
        num_prefills=prefills,
        decode=SimpleNamespace(wrapper=SimpleNamespace(is_cuda_graph_enabled=fixed)),
    )
    with pytest.raises(RuntimeError, match="graph-enabled.*decode"):
        icg.check_independent_decode_metadata({"draft.attn": metadata})


def test_fixed_decode_metadata_is_valid():
    metadata = SimpleNamespace(
        num_prefills=0,
        decode=SimpleNamespace(wrapper=SimpleNamespace(is_cuda_graph_enabled=True)),
    )
    icg.check_independent_decode_metadata({"draft.attn": metadata})


class _Speculator(ars.AutoRegressiveSpeculator):
    def load_draft_model(self, target_model, target_attn_layer_names):
        raise NotImplementedError


def _speculator():
    speculator = object.__new__(_Speculator)
    speculator.vllm_config = _config()
    speculator.device = torch.device("cpu")
    speculator.num_speculative_steps = 3
    speculator.draft_attn_layer_names = {"draft.attn"}
    speculator._independent_decode_config = None
    return speculator


def _group(graph_enabled=False):
    return SimpleNamespace(
        backend=SimpleNamespace(get_name=lambda: "FLASHINFER"),
        supports_draft_decode_metadata_update=False,
        graph_enabled=graph_enabled,
    )


def test_reinitialization_rebuilds_only_draft_builders_with_private_config(monkeypatch):
    speculator = _speculator()
    target_groups = [[_group()]]
    private_configs = []

    def normal_set_attn(self, model_state, kv_config, blocks, inputs, target):
        assert self.attn_vllm_config is self.vllm_config
        self.attn_groups = [[_group()]]
        self.attn_cg_support = SimpleNamespace(
            min_cg_support=AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
        )

    def init_draft(kv_config, config, device, *, active_layer_names):
        assert active_layer_names == {"draft.attn"}
        private_configs.append(config)
        return [[_group(True)]], speculator.attn_cg_support, []

    monkeypatch.setattr(ars.DraftModelSpeculator, "set_attn", normal_set_attn)
    monkeypatch.setattr(ars, "init_attn_backend", init_draft)
    for _ in range(2):
        speculator.set_attn(None, None, None, None, target_groups)
        assert speculator.attn_groups[0][0].graph_enabled
        assert not speculator.use_fused_multi_step_decode
    assert private_configs[0] is not private_configs[1]
    assert speculator.attn_vllm_config is private_configs[1]
    assert not target_groups[0][0].graph_enabled
    assert (
        speculator.vllm_config.compilation_config.cudagraph_mode
        == CUDAGraphMode.PIECEWISE
    )


@pytest.mark.parametrize("independent", [False, True])
def test_prefill_and_decode_managers_use_their_own_config(monkeypatch, independent):
    speculator = _speculator()
    if independent:
        speculator._independent_decode_config, _ = _resolve(speculator.vllm_config)

    def manager(config, device, mode, decode_query_len):
        return SimpleNamespace(config=config, mode=mode, query_len=decode_query_len)

    monkeypatch.setattr(ars, "SpeculatorCudaGraphManager", manager)
    speculator.init_cudagraph_manager(CUDAGraphMode.PIECEWISE)
    prefill = speculator.prefill_cudagraph_manager
    decode = speculator.decode_cudagraph_manager
    assert prefill.config is speculator.vllm_config
    assert prefill.mode == CUDAGraphMode.PIECEWISE
    assert prefill.query_len == 4
    assert decode.query_len == 1
    assert decode.mode == (
        CUDAGraphMode.FULL_DECODE_ONLY if independent else CUDAGraphMode.NONE
    )
    assert decode.config is (
        speculator._independent_decode_config if independent else speculator.vllm_config
    )


@pytest.mark.parametrize("mode", [CUDAGraphMode.FULL, CUDAGraphMode.NONE])
def test_each_draft_step_updates_metadata_before_graph_or_eager_execution(
    monkeypatch, mode
):
    speculator = _speculator()
    speculator._independent_decode_config, _ = _resolve(speculator.vllm_config)
    speculator.current_draft_step = torch.tensor(0)
    speculator.idx_mapping = torch.arange(4)
    speculator.input_buffers = SimpleNamespace(
        positions=torch.arange(4), query_start_loc=torch.arange(5)
    )
    lengths = torch.tensor([16, 32, 64, 128])
    events = []
    speculator.block_tables = SimpleNamespace(compute_slot_mappings=lambda *args: None)
    speculator.kv_cache_config = None
    monkeypatch.setattr(ars, "build_slot_mappings_by_layer", lambda *args: {})

    def metadata(**kwargs):
        events.append(("metadata", kwargs["step"], lengths.clone()))
        return {
            "draft.attn": SimpleNamespace(
                num_prefills=0,
                decode=SimpleNamespace(
                    wrapper=SimpleNamespace(is_cuda_graph_enabled=True)
                ),
            )
        }

    def execute(*args, **kwargs):
        events.append(
            ("execute", speculator.current_draft_step.item(), lengths.clone())
        )
        if speculator.current_draft_step.item() < speculator.num_speculative_steps - 1:
            lengths.add_(1)

    speculator._build_uniform_attn_metadata = metadata
    speculator._generate_draft = execute
    speculator.decode_cudagraph_manager = SimpleNamespace(run_fullgraph=execute)
    desc = BatchExecutionDescriptor(mode, 4, 4, uniform_token_count=1)
    speculator._multi_step_decode(4, False, desc, None, lengths)
    assert [(event[0], event[1]) for event in events] == [
        ("metadata", 1),
        ("execute", 1),
        ("metadata", 2),
        ("execute", 2),
    ]
    assert torch.equal(events[2][2], events[0][2] + 1)
