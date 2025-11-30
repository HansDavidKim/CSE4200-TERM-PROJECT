# coding=utf-8
# Copyright 2019 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Abstract classes that encode a user's state and dynamics.

This version has been updated to use gymnasium instead of gym,
and to rely on Python 3's abc.ABC instead of six.add_metaclass.
"""

import abc
from gymnasium import spaces
import numpy as np


class AbstractResponse(abc.ABC):
  """Abstract class to model a user response."""

  @staticmethod
  @abc.abstractmethod
  def response_space():
    """Gymnasium space that defines how a single response is represented."""

  @abc.abstractmethod
  def create_observation(self):
    """Creates a tensor-like observation of this response.

    Returns:
      A numpy array (or array-like) representation of the response.
      If you use PyTorch, you can wrap this in torch.tensor(...) at the agent
      or environment boundary.
    """


class AbstractUserState(abc.ABC):
  """Abstract class to represent a user's state."""

  # Number of features to represent the user's interests.
  NUM_FEATURES = None

  @abc.abstractmethod
  def create_observation(self):
    """Generates obs of underlying state to simulate partial observability.

    Returns:
      obs: A float array (numpy) of the observed user features.
    """

  @staticmethod
  @abc.abstractmethod
  def observation_space():
    """Gymnasium.space object that defines how user states are represented.

    Typically this will be a spaces.Box(...) describing the feature vector.
    """


class AbstractUserSampler(abc.ABC):
  """Abstract class to sample users."""

  def __init__(self, user_ctor, seed=0):
    """Creates a new user state sampler.

    User states of the type user_ctor are sampled.

    Args:
      user_ctor: A class/constructor for the type of user states that will be
        sampled.
      seed: An integer for a random seed.
    """
    self._user_ctor = user_ctor
    self._seed = seed
    self.reset_sampler()

  def reset_sampler(self):
    """Resets the internal random number generator."""
    self._rng = np.random.RandomState(self._seed)

  @abc.abstractmethod
  def sample_user(self):
    """Creates a new instantiation of this user's hidden state parameters.

    Returns:
      An instance of AbstractUserState (or a subclass).
    """

  def get_user_ctor(self):
    """Returns the constructor/class of the user states that will be sampled."""
    return self._user_ctor


class AbstractUserModel(abc.ABC):
  """Abstract class to represent an encoding of a user's dynamics."""

  def __init__(self, response_model_ctor, user_sampler, slate_size):
    """Initializes a new user model.

    Args:
      response_model_ctor: A class/constructor representing the type of
        responses this model will generate. The class must implement
        AbstractResponse.
      user_sampler: An instance of AbstractUserSampler that can generate
        initial user states from an initial state distribution.
      slate_size: integer number of documents that can be served to the user at
        any interaction.
    """
    if not response_model_ctor:
      raise TypeError('response_model_ctor is a required callable')

    self._user_sampler = user_sampler
    self._user_state = self._user_sampler.sample_user()
    self._response_model_ctor = response_model_ctor
    self._slate_size = slate_size

  # Transition model
  @abc.abstractmethod
  def update_state(self, slate_documents, responses):
    """Updates the user's state based on the slate and document selected.

    Args:
      slate_documents: A list of AbstractDocuments for items in the slate.
      responses: A list of AbstractResponses for each item in the slate.

    Side effects:
      Updates the user's hidden state (self._user_state).
    """

  def reset(self):
    """Resets the user to a freshly sampled state."""
    self._user_state = self._user_sampler.sample_user()

  def reset_sampler(self):
    """Resets the sampler RNG."""
    self._user_sampler.reset_sampler()

  @abc.abstractmethod
  def is_terminal(self):
    """Returns a boolean indicating whether this session is over."""

  # Choice model
  @abc.abstractmethod
  def simulate_response(self, documents):
    """Simulates the user's response to a slate of documents.

    This could involve simulating models of attention, as well as random
    sampling for selection from scored documents.

    Args:
      documents: a list of AbstractDocuments

    Returns:
      responses: a list of AbstractResponse objects for each slate item
    """

  def response_space(self):
    """Gymnasium space describing the response for a whole slate.

    Internally builds a Tuple of single-response spaces, repeated slate_size.
    """
    res_space = self._response_model_ctor.response_space()
    return spaces.Tuple(tuple([res_space] * self._slate_size))

  def get_response_model_ctor(self):
    """Returns a constructor for the type of response this model will create."""
    return self._response_model_ctor

  def observation_space(self):
    """A Gymnasium space that describes possible user observations."""
    return self._user_state.observation_space()

  def create_observation(self):
    """Emits observation about the user's state."""
    return self._user_state.create_observation()
