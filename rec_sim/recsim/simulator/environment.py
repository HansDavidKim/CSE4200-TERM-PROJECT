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

"""Refactored RecSim environment (gym→gymnasium, Python3 ABC cleanup).

This version removes six, Python2 metaclass patterns, and
keeps full compatibility with the user/document models.
"""

import abc
import collections
import itertools

from recsim import document


class AbstractEnvironment(abc.ABC):
    """Abstract class representing the recommender system environment.

    The agent interacts with this environment, receiving observations of
    user state and candidate documents, and produces an action (slate).
    """

    def __init__(
        self,
        user_model,
        document_sampler,
        num_candidates,
        slate_size,
        resample_documents=True,
    ):
        self._user_model = user_model
        self._document_sampler = document_sampler
        self._slate_size = slate_size
        self._num_candidates = num_candidates
        self._resample_documents = resample_documents

        # Create a candidate set.
        self._do_resample_documents()

        if slate_size > num_candidates:
            raise ValueError(
                f"Slate size {slate_size} cannot exceed number of candidates {num_candidates}"
            )

    def _do_resample_documents(self):
        """Resample candidate documents (content creator stub)."""
        self._candidate_set = document.CandidateSet()
        for _ in range(self._num_candidates):
            self._candidate_set.add_document(self._document_sampler.sample_document())

    @abc.abstractmethod
    def reset(self):
        """Resets environment → returns (user_obs, doc_obs)."""

    @abc.abstractmethod
    def reset_sampler(self):
        """Resets document/user samplers."""

    @property
    def num_candidates(self):
        return self._num_candidates

    @property
    def slate_size(self):
        return self._slate_size

    @property
    def candidate_set(self):
        return self._candidate_set

    @property
    def user_model(self):
        return self._user_model

    @abc.abstractmethod
    def step(self, slate):
        """Step environment with action `slate`.

        Returns:
            user_obs,
            doc_obs,
            responses,
            done
        """


# -------------------------------------------------------------------
# Single-user environment
# -------------------------------------------------------------------
class SingleUserEnvironment(AbstractEnvironment):
    """Environment for a single user session."""

    def reset(self):
        """Reset → (user_obs, doc_obs)"""
        self._user_model.reset()
        user_obs = self._user_model.create_observation()

        if self._resample_documents:
            self._do_resample_documents()

        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation()
        )
        return user_obs, self._current_documents

    def reset_sampler(self):
        self._document_sampler.reset_sampler()
        self._user_model.reset_sampler()

    def step(self, slate):
        """Executes an action and returns results."""

        if len(slate) > self._slate_size:
            raise ValueError(
                f"Slate size too large: expected {self._slate_size}, got {len(slate)}"
            )

        # map indices to document objects
        doc_ids = list(self._current_documents.keys())
        mapped_slate = [doc_ids[x] for x in slate]
        documents = self._candidate_set.get_documents(mapped_slate)

        # simulate user response
        responses = self._user_model.simulate_response(documents)

        # update user state
        self._user_model.update_state(documents, responses)

        # update document states
        self._document_sampler.update_state(documents, responses)

        # next state
        user_obs = self._user_model.create_observation()
        done = self._user_model.is_terminal()

        if self._resample_documents:
            self._do_resample_documents()

        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation()
        )

        return user_obs, self._current_documents, responses, done


Environment = SingleUserEnvironment  # backward compatibility alias


# -------------------------------------------------------------------
# Multi-user environment
# -------------------------------------------------------------------
class MultiUserEnvironment(AbstractEnvironment):
    """Environment that simulates multiple users simultaneously."""

    @property
    def num_users(self):
        return len(self.user_model)

    def reset(self):
        for um in self.user_model:
            um.reset()

        user_obs = [um.create_observation() for um in self.user_model]

        if self._resample_documents:
            self._do_resample_documents()

        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation()
        )

        return user_obs, self._current_documents

    def reset_sampler(self):
        self._document_sampler.reset_sampler()
        for um in self.user_model:
            um.reset_sampler()

    def step(self, slates):
        if len(slates) != self.num_users:
            raise ValueError(
                f"Expected {self.num_users} slates, got {len(slates)}"
            )

        all_user_obs, all_documents, all_responses = [], [], []

        for um, slate in zip(self.user_model, slates):
            if len(slate) > self._slate_size:
                raise ValueError(
                    f"Slate too large: expected {self._slate_size}, got {len(slate)}"
                )

            doc_ids = list(self._current_documents.keys())
            mapped_slate = [doc_ids[x] for x in slate]
            documents = self._candidate_set.get_documents(mapped_slate)

            if um.is_terminal():
                responses = []
            else:
                responses = um.simulate_response(documents)
                um.update_state(documents, responses)

            all_user_obs.append(um.create_observation())
            all_documents.append(documents)
            all_responses.append(responses)

        # flatten for document sampler
        def flatten(x):
            return list(itertools.chain(*x))

        self._document_sampler.update_state(
            flatten(all_documents), flatten(all_responses)
        )

        done = all(um.is_terminal() for um in self.user_model)

        if self._resample_documents:
            self._do_resample_documents()

        self._current_documents = collections.OrderedDict(
            self._candidate_set.create_observation()
        )

        return all_user_obs, self._current_documents, all_responses, done
