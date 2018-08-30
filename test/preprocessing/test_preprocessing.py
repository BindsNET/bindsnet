import os
import torch

from bindsnet.preprocessing import NumentaPreprocessor


class TestPreprocessing:
    def test_numenta_preprocessor(self):
        csvfile = './test/preprocessing/test.csv'
        processedfile = './test/preprocessing/data.p'

        def test1():
            enc = NumentaPreprocessor()
            val = enc.process(csvfile, use_cache=True, cachedfile=processedfile)
            assert os.path.exists(processedfile)
            os.remove(processedfile)
            assert val is not None

        def test2():
            enc = NumentaPreprocessor()

            val = enc.process(csvfile, use_cache=False, cachedfile=processedfile)
            assert val is not None
            assert not os.path.exists(processedfile)

        def test3():
            enc = NumentaPreprocessor()
            v1 = enc.process(csvfile, use_cache=False, cachedfile=processedfile)
            v2 = enc.process(csvfile, use_cache=False, cachedfile=processedfile)
            assert torch.eq(v1, v2).all()
            assert len(v1) == 30

        def test4():
            enc = NumentaPreprocessor(n=5000, w=35)
            val = enc.process(csvfile, False, processedfile)
            assert val.shape[1] == 5000
            assert sum(sum(val)) == 35 * 30

        test1(), test2(), test3(), test4()
