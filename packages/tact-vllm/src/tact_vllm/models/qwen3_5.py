# SPDX-License-Identifier: Apache-2.0
# Constructor bodies adapted from vLLM 0.30.0; upstream Qwen/vLLM/HuggingFace copyright.
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.models import qwen3_5 as native

from tact_vllm.runtime.buffers import TactRuntimeBuffers, load_manifest
from tact_vllm.runtime.ops import apply_site


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class TactQwen3_5Model(native.Qwen3_5Model):
    def __init__(self, *, vllm_config: native.VllmConfig, prefix: str = ""):
        native.nn.Module.__init__(self)
        config: native.Qwen3_5TextConfig | native.Qwen3_5MoeTextConfig = (
            vllm_config.model_config.hf_text_config
        )
        parallel_config = vllm_config.parallel_config
        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts
        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        self.embed_tokens = native.VocabParallelEmbedding(self.vocab_size, config.hidden_size)

        def get_layer(prefix: str):
            return native.Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[native.extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = native.make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.is_fused_shared_expert_enabled = native.is_model_fused_shared_expert_compatible(
            self.layers, native.Qwen3NextSparseMoeBlock, "mlp"
        )
        self.make_empty_intermediate_tensors = native.make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if native.get_pp_group().is_last_rank:
            self.norm = native.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = native.PPMissingLayer()
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        manifest = load_manifest()
        if (
            manifest["layer_count"] != config.num_hidden_layers
            or manifest["hidden_size"] != config.hidden_size
        ):
            raise ValueError("TACT manifest/backbone shape mismatch")
        if self.use_sequence_parallel:
            raise ValueError("TACT token sequence parallelism is not implemented")
        self.tact = TactRuntimeBuffers(manifest, device=self.embed_tokens.weight.device)
        if self.tact.consumer_check:
            for block, site in self.tact.decoder_sites.items():
                target = (self.layers[block + 1].input_layernorm
                          if block + 1 < len(self.layers) else self.norm)
                target.register_forward_pre_hook(
                    self.tact.consumer_probes[site].norm_hook, with_kwargs=True)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        if intermediate_tensors is not None or self.aux_hidden_state_layers:
            raise ValueError("TACT PP and auxiliary hidden outputs are unsupported")
        hidden = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        for layer_index, layer in enumerate(self.layers):
            hidden, residual = layer(positions=positions, hidden_states=hidden, residual=residual)
            if layer_index in self.tact.decoder_sites:
                residual = apply_site(
                    hidden, residual, self.tact, self.tact.decoder_sites[layer_index]
                )
        hidden, _ = self.norm(hidden, residual)
        if self.tact.final_site is not None:
            hidden = apply_site(hidden, None, self.tact, self.tact.final_site)
        return hidden


class TactCausalConstructor:
    def __init__(self, *, vllm_config: native.VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3.5 currently does not support 'all' prefix caching, please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config
        native.nn.Module.__init__(self)
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = TactQwen3_5Model(
            vllm_config=vllm_config, prefix=native.maybe_prefix(prefix, "model")
        )
        if native.get_pp_group().is_last_rank:
            self.lm_head = native.ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=self.quant_config,
                prefix=native.maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
        else:
            self.lm_head = native.PPMissingLayer()
        self.logits_processor = native.LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors


class TactQwen3_5ForCausalLM(TactCausalConstructor, native.Qwen3_5ForCausalLM):
    pass


class TactQwen3_5MoeForCausalLM(TactCausalConstructor, native.Qwen3_5MoeForCausalLM):
    def __init__(self, *, vllm_config, prefix=""):
        TactCausalConstructor.__init__(self, vllm_config=vllm_config, prefix=prefix)
        self.set_moe_parameters()


@native.MULTIMODAL_REGISTRY.register_processor(
    native.Qwen3VLMultiModalProcessor,
    info=native.Qwen3_5ProcessingInfo,
    dummy_inputs=native.Qwen3VLDummyInputsBuilder,
)
class TactQwen3_5ForConditionalGeneration(native.Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config: native.VllmConfig, prefix: str = "model"):
        native.nn.Module.__init__(self)
        config: native.Qwen3_5Config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = multimodal_config.is_multimodal_pruning_enabled()
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = native.cached_tokenizer_from_config(vllm_config.model_config)
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = native.Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-06),
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "visual"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = TactQwen3_5ForCausalLM(
                vllm_config=vllm_config, prefix=native.maybe_prefix(prefix, "language_model")
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors


@native.MULTIMODAL_REGISTRY.register_processor(
    native.Qwen3VLMultiModalProcessor,
    info=native.Qwen3_5MoeProcessingInfo,
    dummy_inputs=native.Qwen3VLDummyInputsBuilder,
)
class TactQwen3_5MoeForConditionalGeneration(native.Qwen3_5MoeForConditionalGeneration):
    def __init__(self, *, vllm_config: native.VllmConfig, prefix: str = "model"):
        native.nn.Module.__init__(self)
        config: native.Qwen3_5MoeConfig = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = multimodal_config.is_multimodal_pruning_enabled()
        self.video_pruning_rate = self.multimodal_config.video_pruning_rate
        self._tokenizer = native.cached_tokenizer_from_config(vllm_config.model_config)
        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        )
        self.visual_dim = config.vision_config.out_hidden_size
        self.multiscale_dim = self.visual_dim * self.deepstack_num_level
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = native.Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-06),
                quant_config=quant_config,
                prefix=native.maybe_prefix(prefix, "visual"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = TactQwen3_5MoeForCausalLM(
                vllm_config=vllm_config, prefix=native.maybe_prefix(prefix, "language_model")
            )
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors
        self.set_moe_parameters()
