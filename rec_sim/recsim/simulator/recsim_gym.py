# coding=utf-8
# Modernized RecSim Gymnasium Wrapper

import collections
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from recsim.simulator import environment


def _dummy_metrics_aggregator(responses, metrics, info):
    return metrics


def _dummy_metrics_writer(metrics, add_summary_fn):
    return


class RecSimGymEnv(gym.Env):
    """
    Modern Gymnasium wrapper for RecSim environments.

    Supports:
      - SingleUserEnvironment
      - MultiUserEnvironment
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        raw_environment,
        reward_aggregator,
        metrics_aggregator=_dummy_metrics_aggregator,
        metrics_writer=_dummy_metrics_writer,
    ):
        super().__init__()

        self._environment = raw_environment
        self._reward_aggregator = reward_aggregator
        self._metrics_aggregator = metrics_aggregator
        self._metrics_writer = metrics_writer

        self._metrics = collections.defaultdict(float)

        # Build Gymnasium spaces
        self._action_space = self._build_action_space()
        self._observation_space = self._build_observation_space()

    # ------------------------------------------------------------------
    # Gymnasium Required Properties
    # ------------------------------------------------------------------
    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_action_space(self):
        """Creates a MultiDiscrete action or Tuple of such for multi-user."""
        single_action = spaces.MultiDiscrete(
            np.full(self._environment.slate_size, self._environment.num_candidates)
        )

        if isinstance(self._environment, environment.MultiUserEnvironment):
            return spaces.Tuple([single_action] * self._environment.num_users)

        return single_action

    def _build_observation_space(self):
        """Creates observation dict space: user/doc/response."""

        # user observation space
        if isinstance(self._environment, environment.MultiUserEnvironment):
            base_user = self._environment.user_model[0].observation_space()
            base_resp = self._environment.user_model[0].response_space()

            user_obs_space = spaces.Tuple(
                [base_user] * self._environment.num_users
            )
            resp_obs_space = spaces.Tuple(
                [base_resp] * self._environment.num_users
            )
        else:  # single user
            user_obs_space = self._environment.user_model.observation_space()
            resp_obs_space = self._environment.user_model.response_space()

        # document observation space
        doc_obs_space = self._environment.candidate_set.observation_space()

        return spaces.Dict(
            {
                "user": user_obs_space,
                "doc": doc_obs_space,
                "response": resp_obs_space,
            }
        )

    # ------------------------------------------------------------------
    # Gymnasium API: step(), reset()
    # ------------------------------------------------------------------
    def step(self, action):
        user_obs, doc_obs, responses, done = self._environment.step(action)

        # Convert responses to observation format
        if isinstance(self._environment, environment.MultiUserEnvironment):
            all_responses = tuple(
                tuple(resp.create_observation() for resp in per_user)
                for per_user in responses
            )
        else:
            all_responses = tuple(resp.create_observation() for resp in responses)

        obs = {
            "user": user_obs,
            "doc": doc_obs,
            "response": all_responses,
        }

        reward = self._reward_aggregator(responses)
        info = self.extract_env_info()

        terminated = done
        truncated = False  # RecSim has no truncation concept

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        user_obs, doc_obs = self._environment.reset()

        obs = {
            "user": user_obs,
            "doc": doc_obs,
            "response": None,
        }
        return obs, {}

    # ------------------------------------------------------------------
    # Extra: metrics, info, seeding
    # ------------------------------------------------------------------
    def extract_env_info(self):
        return {"env": self._environment}

    def reset_metrics(self):
        self._metrics = collections.defaultdict(float)

    def update_metrics(self, responses, info=None):
        self._metrics = self._metrics_aggregator(
            responses, self._metrics, info
        )

    def write_metrics(self, add_summary_fn):
        self._metrics_writer(self._metrics, add_summary_fn)

    def render(self, mode="human"):
        raise NotImplementedError

    def close(self):
        pass
