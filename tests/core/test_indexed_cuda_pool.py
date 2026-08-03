"""Indexed CUDA labels must remain in the GPU admission pool."""
from unittest.mock import MagicMock

from muse.cli_impl.load_director import LoadDirector
from muse.core.memory_probe import resolve_memory_pool


def test_indexed_cuda_labels_use_cuda_pool():
    for device in ("cuda:0", "cuda:1", "CUDA:12"):
        assert resolve_memory_pool(
            device,
            gpu_free_gb=None,
            cuda_available=False,
        ) == "cuda"


def test_malformed_cuda_indices_do_not_claim_cuda_pool():
    for device in ("cuda:", "cuda:-1", "cuda:gpu"):
        assert resolve_memory_pool(
            device,
            gpu_free_gb=8.0,
            cuda_available=True,
        ) == "cpu"


def test_load_director_delegates_indexed_cuda_to_cuda_pool():
    probe = MagicMock()
    probe.gpu_free_gb.return_value = 8.0
    probe.cpu_free_gb.return_value = 64.0
    director = LoadDirector(
        enable_fn=MagicMock(return_value=9001),
        disable_fn=MagicMock(),
        memory_probe=probe,
    )

    assert director._resolve_pool_device("cuda:0") == "cuda"
    assert director._resolve_pool_device("cuda:7") == "cuda"
