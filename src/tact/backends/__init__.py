"""Serving adapters sharing one runner and the native HTTP protocol."""


def process_manager(cfg, directory):
    if cfg["serving"]["backend"] == "easysteer":
        from tact.backends.easysteer import EasySteerProcessManager

        return EasySteerProcessManager(cfg, directory)
    if cfg["serving"]["backend"] == "registry_vllm":
        from tact.backends.registry_vllm import RegistryProcessManager

        return RegistryProcessManager(cfg, directory)
    from tact.backends.native_vllm import BackendProcessManager

    return BackendProcessManager(cfg, directory)


def model_config(cfg):
    from tact.backends.native_vllm import model_config as native_config

    result = native_config(cfg)
    if cfg["serving"]["backend"] == "registry_vllm":
        from tact.backends.registry_vllm import request_fields

        result["model_kwargs"]["extra_body"]["vllm_xargs"] = request_fields(cfg)
    if cfg["serving"]["backend"] == "easysteer":
        from tact.backends.easysteer import request_fields

        result["model_kwargs"]["extra_body"]["steering"] = request_fields(cfg)
    return result
