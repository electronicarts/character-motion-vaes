import os
import re

import gym
import pytest

import environments  # noqa: F401  (registers the env ids)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REGISTERED = [
    "RandomWalkEnv-v0",
    "TargetEnv-v0",
    "JoystickEnv-v0",
    "PathFollowEnv-v0",
    "HumanMazeEnv-v0",
]


@pytest.mark.parametrize("env_id", REGISTERED)
def test_env_id_resolves_without_namespace_prefix(env_id):
    # Importing `environments` is what registers the ids; gym.make is then
    # given the bare id.
    assert gym.spec(env_id) is not None


@pytest.mark.parametrize("env_id", REGISTERED)
def test_namespaced_id_is_not_in_the_registry(env_id):
    # gym 0.22 changed "prefix:" from an import hint into a namespace lookup.
    # environments/__init__.py registers without a namespace, so the prefixed
    # form is absent from the registry and gym.make raises NameNotFound.
    # (gym.spec still honours the legacy form, so it cannot detect this.)
    assert "environments:{}".format(env_id) not in gym.envs.registry.env_specs
    assert env_id in gym.envs.registry.env_specs


def test_no_source_file_passes_a_namespaced_id_to_gym_make():
    offenders = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "docs", "tests"}]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            with open(path, encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    if re.search(r'"\{\}:\{\}"\.format|"environments:', line):
                        offenders.append(
                            "%s:%d" % (os.path.relpath(path, REPO_ROOT), number)
                        )
    assert offenders == [], "namespaced env id built at: {}".format(offenders)
