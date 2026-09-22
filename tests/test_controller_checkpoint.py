import torch

from vae_motion.models import (
    PoseVAEController,
    PoseVAEPolicy,
    load_model,
    read_config,
    save_model,
)

OBS = 6
ACT = 3


def build_controller():
    torch.manual_seed(0)
    return PoseVAEController(OBS, ACT)


def build_policy():
    torch.manual_seed(0)
    return PoseVAEPolicy(PoseVAEController(OBS, ACT))


def test_controller_is_constructed_from_dimensions_not_an_env():
    controller = build_controller()
    assert controller.observation_dim == OBS
    assert controller.action_dim == ACT


def test_controller_round_trip_preserves_output(tmp_path):
    controller = build_controller()
    path = str(tmp_path / "c.safetensors")
    save_model(controller, path)

    reloaded = load_model(path)
    reloaded.eval()
    controller.eval()

    obs = torch.randn(2, OBS)
    with torch.no_grad():
        assert torch.equal(controller(obs), reloaded(obs))


def test_policy_round_trip_preserves_actor_and_critic(tmp_path):
    policy = build_policy()
    path = str(tmp_path / "p.safetensors")
    save_model(policy, path)

    reloaded = load_model(path)
    policy.eval()
    reloaded.eval()

    obs = torch.randn(2, OBS)
    with torch.no_grad():
        assert torch.equal(policy.actor(obs), reloaded.actor(obs))
        assert torch.equal(policy.critic(obs), reloaded.critic(obs))


def test_policy_round_trip_preserves_all_tensors(tmp_path):
    policy = build_policy()
    path = str(tmp_path / "p.safetensors")
    save_model(policy, path)
    reloaded = load_model(path)

    original = policy.state_dict()
    restored = reloaded.state_dict()
    assert set(original) == set(restored)
    # dist.logstd._bias is a parameter too and must survive.
    assert "dist.logstd._bias" in original
    for name in original:
        assert torch.equal(original[name], restored[name]), name


def test_policy_config_records_dimensions(tmp_path):
    path = str(tmp_path / "p.safetensors")
    save_model(build_policy(), path)
    config = read_config(path)
    assert config["class"] == "PoseVAEPolicy"
    assert config["observation_dim"] == OBS
    assert config["action_dim"] == ACT


def test_frame_skip_travels_in_the_config(tmp_path):
    # Replaces the old `actor_critic.env_info = {...}` attribute hack.
    path = str(tmp_path / "p.safetensors")
    save_model(build_policy(), path, extra_config={"frame_skip": 2})
    assert read_config(path)["frame_skip"] == 2


def test_frame_skip_absent_when_not_supplied(tmp_path):
    path = str(tmp_path / "p.safetensors")
    save_model(build_policy(), path)
    assert read_config(path).get("frame_skip", 1) == 1
