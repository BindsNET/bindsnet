import torch

from ..utils import im2col_indices


def post_pre(conn, **kwargs):
    '''
    Simple STDP rule involving both pre- and post-synaptic spiking activity.
    
    Inputs:
        
        | :code:`conn` (:code:`bindsnet.network.topology.AbstractConnection`):
        An instance of class :code:`AbstractAbstractConnectionConnection`.
    '''
    if not 'kernel_size' in conn.__dict__:
        x_source, x_target = conn.source.x.unsqueeze(-1), conn.target.x.unsqueeze(0)
        s_source, s_target = conn.source.s.float().unsqueeze(-1), conn.target.s.float().unsqueeze(0)
        
        # Post-synaptic.
        conn.w += conn.nu_post * x_source * s_target
        # Pre-synaptic.
        conn.w -= conn.nu_pre * s_source * x_target

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
    else:
        out_channels, _, kernel_height, kernel_width = conn.w.size()
        padding, stride = conn.padding, conn.stride
        
        x_source = im2col_indices(conn.source.x,
                                  kernel_height,
                                  kernel_width,
                                  padding=padding,
                                  stride=stride)

        x_target = conn.target.x.permute(1, 2, 3, 0).reshape(out_channels,
                                                             -1)
        s_source = im2col_indices(conn.source.s,
                                  kernel_height,
                                  kernel_width,
                                  padding=padding,
                                  stride=stride).float()

        s_target = conn.target.s.permute(1, 2, 3, 0).reshape(out_channels,
                                                             -1).float()
        
        # Post-synaptic.
        post = s_target @ x_source.t()
        conn.w += conn.nu_post * post.view(conn.w.size())
        
        # Pre-synaptic.
        pre = x_target @ s_source.t()
        conn.w -= conn.nu_pre * pre.view(conn.w.size())

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)

def hebbian(conn, **kwargs):
    '''
    Simple Hebbian learning rule. Pre- and post-synaptic updates are both positive.
    
    Inputs:
        
        | :code:`conn` (:code:`bindsnet.network.topology.AbstractConnection`):
        An instance of class :code:`AbstractConnection`.
    '''
    if not 'kernel_size' in conn.__dict__:
        # Post-synaptic.
        conn.w += conn.nu_post * conn.source.x.unsqueeze(-1) * conn.target.s.float().unsqueeze(0)
        # Pre-synaptic.
        conn.w += conn.nu_pre * conn.source.s.float().unsqueeze(-1) * conn.target.x.unsqueeze(0)

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
    else:
        out_channels, _, kernel_height, kernel_width = conn.w.size()
        padding, stride = conn.padding, conn.stride
        
        x_source = im2col_indices(conn.source.x,
                                  kernel_height,
                                  kernel_width,
                                  padding=padding,
                                  stride=stride)

        x_target = conn.target.x.permute(1, 2, 3, 0).reshape(out_channels,
                                                             -1)
        s_source = im2col_indices(conn.source.s,
                                  kernel_height,
                                  kernel_width,
                                  padding=padding,
                                  stride=stride).float()

        s_target = conn.target.s.permute(1, 2, 3, 0).reshape(out_channels,
                                                             -1).float()
        
        # Post-synaptic.
        post = (x_source @ s_target.t()).view(conn.w.size())
        if post.max() > 0:
            post = post / post.max()
        
        conn.w += conn.nu_post * post
        
        # Pre-synaptic.
        pre = (s_source @ x_target.t()).view(conn.w.size())
        if pre.max() > 0:
            pre = pre / pre.max()
        
        conn.w += conn.nu_pre * pre

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)


def m_stdp(conn, **kwargs):
    '''
    Reward-modulated STDP. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    
    Inputs:
        
        | :code:`conn` (:code:`bindsnet.network.topology.AbstractConnection`):
        An instance of class :code:`AbstractConnection`.
    '''
    # Parse keyword arguments.
    try:
        reward = kwargs['reward']
    except KeyError:
        raise KeyError('function m_stdp requires a reward kwarg')

    a_plus = kwargs.get('a_plus', 1)
    a_minus = kwargs.get('a_plus', -1)
    
    if not 'kernel_size' in conn.__dict__:
        # Get P^+ and P^- values (function of firing traces).
        p_plus = a_plus * conn.source.x.unsqueeze(-1)
        p_minus = a_minus * conn.target.x.unsqueeze(0)

        # Get pre- and post-synaptic spiking neurons.
        pre_fire = conn.source.s.float().unsqueeze(-1)
        post_fire = conn.target.s.float().unsqueeze(0)

        # Calculate point eligibility value.
        eligibility = p_plus * post_fire + pre_fire * p_minus

        # Compute weight update.
        conn.w += conn.nu * reward * eligibility

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
    else:
        out_channels, _, kernel_height, kernel_width = conn.w.size()
        padding, stride = conn.padding, conn.stride
        
        p_plus = a_plus * im2col_indices(conn.source.x,
                                         kernel_height,
                                         kernel_width,
                                         padding=padding,
                                         stride=stride)
        
        p_minus = a_minus * conn.target.x.permute(1, 2, 3, 0).reshape(out_channels,
                                                                      -1)
        pre_fire = im2col_indices(conn.source.s,
                                  kernel_height,
                                  kernel_width,
                                  padding=padding,
                                  stride=stride).float()

        post_fire = conn.target.s.permute(1, 2, 3, 0).reshape(out_channels,
                                                              -1).float()
        
        # Post-synaptic.
        post = (p_plus @ post_fire.t()).view(conn.w.size())
        if post.max() > 0:
            post = post / post.max()
        
        # Pre-synaptic.
        pre = (pre_fire @ p_minus.t()).view(conn.w.size())
        if pre.max() > 0:
            pre = pre / pre.max()

        # Calculate point eligibility value.
        eligibility = post + pre
        
        # Compute weight update.
        conn.w += conn.nu * reward * eligibility
            
        # Bound weights.
        conn.w = torch.clamp(conn.w,
                             conn.wmin,
                             conn.wmax)


def m_stdp_et(conn, **kwargs):
    '''
    Reward-modulated STDP with eligibility trace. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    
    Inputs:
        
        | :code:`conn` (:code:`bindsnet.network.topology.AbstractConnection`):
        An instance of class :code:`AbstractConnection`.
        
        | Keyword arguments:
            
            | :code:`a_plus` (:code:`int`): Learning rate (positive).
            | :code:`a_minus` (:code:`int`): Learning rate (negative).
    '''
    if not 'kernel_size' in conn.__dict__:
        # Parse keyword arguments.
        try:
            reward = kwargs['reward']
        except KeyError:
            raise KeyError('function m_stdp_et requires a reward kwarg')

        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_plus', -1)

        # Get P^+ and P^- values (function of firing traces).
        conn.p_plus = -(conn.tc_plus * conn.p_plus) + a_plus * conn.source.x.unsqueeze(-1)
        conn.p_minus = -(conn.tc_minus * conn.p_minus) + a_minus * conn.target.x.unsqueeze(0)

        # Get pre- and post-synaptic spiking neurons.
        pre_fire = conn.source.s.float().unsqueeze(-1)
        post_fire = conn.target.s.float().unsqueeze(0)

        # Calculate value of eligibility trace.
        conn.e_trace += -(conn.tc_e_trace * conn.e_trace) + \
                        conn.p_plus * post_fire + pre_fire * conn.p_minus

        # Compute weight update.
        conn.w += conn.nu * reward * conn.e_trace

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
    else:
        out_channels, _, kernel_height, kernel_width = conn.w.size()
        padding, stride = conn.padding, conn.stride
        
        p_plus = a_plus * im2col_indices(conn.source.x,
                                         kernel_height,
                                         kernel_width,
                                         padding=padding,
                                         stride=stride)

        p_minus = a_minus * conn.target.x.permute(1, 2, 3, 0).reshape(out_channels,
                                                                      -1)
        pre_fire = im2col_indices(conn.source.s,
                                  kernel_height,
                                  kernel_width,
                                  padding=padding,
                                  stride=stride).float()

        post_fire = conn.target.s.permute(1, 2, 3, 0).reshape(out_channels,
                                                              -1).float()
        
        # Post-synaptic.
        post = (p_plus @ post_fire.t()).view(conn.w.size())
        if post.max() > 0:
            post = post / post.max()
        
        # Pre-synaptic.
        pre = (pre_fire @ p_minus.t()).view(conn.w.size())
        if pre.max() > 0:
            pre = pre / pre.max()

        # Calculate point eligibility value.
        conn.e_trace += -(conn.tc_e_trace * conn.e_trace) + (post + pre)
        
        # Compute weight update.
        conn.w += conn.nu * reward * conn.e_trace
            
        # Bound weights.
        conn.w = torch.clamp(conn.w,
                             conn.wmin,
                             conn.wmax)
