import torch

from app.core.logging import get_logger
from app.ml.identifier import CatReIDModel

logger = get_logger(__name__)


def export_to_onnx(checkpoint_path: str, output_path: str, embedding_dim: int = 512):
    """Export PyTorch model to ONNX format."""
    model = CatReIDModel(embedding_dim=embedding_dim)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=17,
    )
    logger.info("Exported ONNX model to %s", output_path)


def export_to_tensorrt(onnx_path: str, engine_path: str, fp16: bool = True):
    """Export ONNX model to TensorRT engine."""
    try:
        import tensorrt as trt

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("TensorRT parse error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("TensorRT FP16 enabled")

        engine = builder.build_serialized_network(network, config)
        with open(engine_path, "wb") as f:
            f.write(engine)
        logger.info("Exported TensorRT engine to %s", engine_path)
    except ImportError:
        logger.warning("TensorRT not available — skipping TRT export")
