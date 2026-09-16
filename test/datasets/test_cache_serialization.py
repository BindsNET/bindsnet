import numpy as np
import torch


class TestSpokenMNISTCache:
    """
    Pins the serialization contract of the ``SpokenMNIST`` processed-data cache.

    ``SpokenMNIST`` writes its processed data with
    ``torch.save((audio, labels), ...)`` and reads it back with
    ``torch.load(..., weights_only=True)``. ``weights_only=True`` refuses to
    execute code while loading, which is only usable because this cache holds
    nothing but tensors. These tests pin that property, so that if the cached
    payload ever gains a non-tensor object the failure shows up here rather
    than as a broken dataset load for a user.

    The real loader needs a download, so these tests exercise the same
    save/load pair on the same payload shape instead: ``audio`` is a list of
    2-D float tensors (filter banks, one per utterance) and ``labels`` is a
    1-D float tensor, per ``SpokenMNIST.process_data``.
    """

    def test_cache_round_trips_under_weights_only(self, tmp_path):
        audio = [torch.rand(7, 13), torch.rand(11, 13)]
        labels = torch.Tensor([3.0, 8.0])

        path = str(tmp_path / "audio.pt")
        torch.save((audio, labels), open(path, "wb"))

        _audio, _labels = torch.load(open(path, "rb"), weights_only=True)

        assert len(_audio) == len(audio)
        for loaded, original in zip(_audio, audio):
            assert torch.equal(loaded, original)
        assert torch.equal(_labels, labels)

    def test_numpy_derived_filter_banks_round_trip(self, tmp_path):
        """
        ``process_data`` builds filter banks through NumPy before they become
        tensors. This pins that the converted result still loads under
        ``weights_only=True``.
        """
        audio = [torch.Tensor(np.random.rand(5, 13).astype(np.float32))]
        labels = torch.Tensor([1.0])

        path = str(tmp_path / "audio.pt")
        torch.save((audio, labels), open(path, "wb"))

        _audio, _labels = torch.load(open(path, "rb"), weights_only=True)

        assert torch.equal(_audio[0], audio[0])
        assert torch.equal(_labels, labels)
