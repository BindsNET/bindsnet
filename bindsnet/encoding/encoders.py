from . import encodings


class Encoder:
    """
    Base class for spike encodings transforms.

    - Calls self.enc from the subclass and passes whatever arguments were
      provided. self.enc must be callable with torch.Tensor, *args, **kwargs
    """

    def __init__(self, *args, **kwargs) -> None:
        self.enc_args = args
        self.enc_kwargs = kwargs

    def __call__(self, img):
        return self.enc(img, *self.enc_args, **self.enc_kwargs)


class NullEncoder(Encoder):
    """
    Pass through of the datum that was input.
    
    WARNING - this is not a real encoder into spikes. Be careful with
    the usage of this class.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, img):
        return img


class SingleEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, sparsity: float = 0.3, **kwargs):
        """
        Creates a callable SingleEncoder which encodes as defined in
        :code:`bindsnet.encodings.single`

        :param time: Length of single spike train per input variable.
        :param dt: Simulation time step.
        :param sparsity: Sparsity of the input representation. 0 for no spike and 1 for all spike.
        """
        super().__init__(time, dt=dt, sparsity=sparsity, **kwargs)

        self.enc = encodings.single


class RepeatEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        """
        Creates a callable RepeatEncoder which encodes as defined in
        :code:`bindsnet.encodings.repeat`

        :param time: Length of repeat spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = encodings.repeat


class BernoulliEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        """
        Creates a callable BernoulliEncoder which encodes as defined in
        :code:`bindsnet.encodings.bernoulli`

        :param time: Length of Bernoulli spike train per input variable.
        :param dt: Simulation time step.

        Keyword arguments:

        :param float max_prob: Maximum probability of spike per Bernoulli trial.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = encodings.bernoulli


class PoissonEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        """
        Creates a callable PoissonEncoder which encodes as defined in
        :code:`bindsnet.encodings.poisson`

        :param time: Length of Poisson spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = encodings.poisson


class RankOrderEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        """
        Creates a callable RankOrderEncoder which encodes as defined in
        :code:`bindsnet.encodings.rank_order`

        :param time: Length of RankOrder spike train per input variable.
        :param dt: Simulation time step.
        """
        super().__init__(time, dt=dt, **kwargs)

        self.enc = encodings.rank_order
