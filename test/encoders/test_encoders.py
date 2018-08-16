from bindsnet.encoders import *


class TestEncoders:
    """
    Tests encoders.
    """

    def test_numenta_encoder(self):
        csvfile = './test/encoders/test.csv'
        encodingfile = './test/encoders/encoding.p'

        def test1():
            enc = NumentaEncoder(csvfile, save=True, encodingfile=encodingfile)
            val = enc.get_encoding()
            assert os.path.exists(encodingfile)
            os.remove(encodingfile)
            assert val is not None

        def test2():
            enc = NumentaEncoder(csvfile, save=False, encodingfile=encodingfile)
            val = enc.get_encoding()
            assert val is not None
            assert not os.path.exists(encodingfile)

        def test3():
            enc1 = NumentaEncoder(csvfile, save=False, encodingfile=encodingfile)
            enc2 = NumentaEncoder(csvfile, save=False, encodingfile=encodingfile)
            v1, v2 = enc1.get_encoding(), enc2.get_encoding()
            assert torch.all(torch.eq(v1, v2))
            assert len(v1) == 30

        def test4():
            enc = NumentaEncoder(csvfile, save=False, encodingfile=encodingfile, n=5000, w=35)
            val = enc.get_encoding()
            assert val.shape[1] == 5000
            assert sum(sum(val)) == 35 * 30

        test1(), test2(), test3(), test4()
