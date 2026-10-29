"""MulticompartmentConnection pipeline bookkeeping: saving and loading feature state,
sub-features, and pipeline edits."""

import pytest
import torch

from bindsnet.learning import MCC_learning
from bindsnet.network import Network
from bindsnet.network import topology_features as tf
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection


def _net(pipeline, n_in=6, n_out=4, batch=1):
    net = Network(dt=1.0, batch_size=batch)
    net.add_layer(Input(n=n_in, traces=True), "in")
    net.add_layer(LIFNodes(n=n_out, traces=True, thresh=-60.0), "out")
    conn = MulticompartmentConnection(
        net.layers["in"], net.layers["out"], device="cpu", pipeline=pipeline
    )
    net.add_connection(conn, "in", "out")
    return net, conn


def _spikes(steps, n=6, batch=1, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(steps, batch, n, generator=g) < 0.5


def test_duplicate_feature_names_refuse_to_save():
    """Two features named "w" would share one state_dict key.

    Caught: saving without the name check (the Bias value overwrote the Weight's).
    """
    _, conn = _net(
        [tf.Weight("w", torch.ones(6, 4)), tf.Bias("w", torch.full((6, 4), 7.0))]
    )
    with pytest.raises(ValueError, match="unique"):
        conn.state_dict()


def test_load_rejects_wrong_shape():
    """A feature value of another shape is an error, as for a normal Parameter.

    Caught: loading with a plain copy_ (a [1, 4] row was broadcast over [6, 4]).
    """
    _, conn = _net([tf.Weight("w", torch.rand(6, 4))])
    state = conn.state_dict()
    state["feature.w"] = torch.ones(1, 4)
    with pytest.raises(RuntimeError, match="size mismatch for feature.w"):
        conn.load_state_dict(state)


def test_adaptation_feature_state_round_trip():
    """The adaptation features' mask, spike buffer and counter are saved and loaded,
    so a reloaded copy continues exactly.

    Caught: saving only ``value`` (the copy's mask and counter restarted).
    """

    def build():
        torch.manual_seed(0)
        ann = [torch.rand(4, 4) - 0.5, torch.rand(1, 4) - 0.5]
        adapt = tf.AdaptationBaseSynapsHistory(
            "a", value=torch.rand(6, 4) - 0.5, ann_values=ann
        )
        return _net([tf.Weight("w", torch.rand(6, 4) * 4, range=[0, 10]), adapt])

    net, _ = build()
    x = _spikes(80)
    net.run(inputs={"in": x[:40]}, time=40)
    copy, _ = build()
    copy.load_state_dict(net.state_dict())
    net.run(inputs={"in": x[40:]}, time=40)
    copy.run(inputs={"in": x[40:]}, time=40)
    a, b = net.connections[("in", "out")], copy.connections[("in", "out")]
    assert torch.equal(a.feature_index["a"].mask, b.feature_index["a"].mask)
    assert torch.equal(a.feature_index["a"].value, b.feature_index["a"].value)
    assert torch.equal(net.layers["out"].v, copy.layers["out"].v)


def test_updating_parent_learns_once_per_step():
    """With an Updating sub-feature in the pipeline, its parent learns when that
    entry runs, not a second time in the connection's update.

    Caught: the connection's update also updating the driven parent (the weight
    change doubled).
    """
    x = _spikes(20)

    def change(with_updating):
        torch.manual_seed(0)
        w = tf.Weight(
            "w",
            torch.rand(6, 4) * 4,
            range=[0, 10],
            nu=(1e-2, 1e-2),
            learning_rule=MCC_learning.PostPre,
        )
        pipeline = [w, tf.Updating("u", parent_feature=w)] if with_updating else [w]
        net, conn = _net(pipeline)
        w0 = w.value.clone()
        net.run(inputs={"in": x}, time=20)
        return (w.value - w0).sum()

    once, via_updating = change(False), change(True)
    assert once != 0
    # Updating runs during compute, so the schedule differs slightly; the change
    # must be that of one update per step, not two.
    assert (via_updating - once).abs() < (via_updating - 2 * once).abs()


def test_sub_feature_parent_must_be_in_pipeline():
    """A sub-feature runs a method of its parent, which must be primed in the same
    pipeline; otherwise the constructor says so.

    Caught: building it (Network.run later raised an unclear TypeError).
    """
    w = tf.Weight(
        "w", torch.rand(6, 4), nu=(1e-2, 1e-2), learning_rule=MCC_learning.PostPre
    )
    with pytest.raises(ValueError, match="must be in the same pipeline"):
        _net([tf.Weight("v", torch.rand(6, 4)), tf.Updating("u", parent_feature=w)])


def test_insert_pipeline_inserts_at_index():
    """insert_pipeline(feature, index) puts the feature at ``index``.

    Caught: list.insert called with its arguments swapped (TypeError).
    """
    w = tf.Weight("w", torch.rand(6, 4))
    _, conn = _net([w])
    bias = tf.Bias("b", torch.rand(6, 4))
    conn.insert_pipeline(bias, 0)
    assert conn.pipeline[0] is bias and conn.feature_index["b"] is bias
