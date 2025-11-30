# coding=utf-8
# Modernized RecSim choice model (Python3, no-six, cleaner ABC patterns)

import abc
import numpy as np


def softmax(vector):
    """Numerically stable softmax."""
    vector = np.array(vector)
    v = vector - np.max(vector)
    exp_v = np.exp(v)
    return exp_v / np.sum(exp_v)


# ======================================================================
# Abstract base
# ======================================================================
class AbstractChoiceModel(abc.ABC):
    """Abstract base class for user choice models."""

    _scores = None          # numpy array of per-document scores
    _score_no_click = None  # scalar

    @abc.abstractmethod
    def score_documents(self, user_state, doc_obs):
        """Compute unnormalized scores for documents in the slate.

        Args:
            user_state: AbstractUserState instance.
            doc_obs: list/array of document observations in the slate.

        Sets:
            self._scores: numpy array of document scores.
            self._score_no_click: scalar representing no-click score.
        """

    @property
    def scores(self):
        return self._scores

    @property
    def score_no_click(self):
        return self._score_no_click

    @abc.abstractmethod
    def choose_item(self):
        """Return the index of chosen document, or None if no click."""
        raise NotImplementedError


# ======================================================================
# Helper class: choice models that normalize scores into probabilities
# ======================================================================
class NormalizableChoiceModel(AbstractChoiceModel):
    """A choice model where document selection probabilities are normalized."""

    @staticmethod
    def _score_documents_helper(user_state, doc_obs):
        """Compute raw scores with user_state.score_document(doc)."""
        return np.array([user_state.score_document(doc) for doc in doc_obs])

    def choose_item(self):
        """Sample index according to normalized (scores + no-click) probs."""
        all_scores = np.append(self._scores, self._score_no_click)
        probs = all_scores / np.sum(all_scores)

        idx = np.random.choice(len(probs), p=probs)

        # last index corresponds to 'no click'
        if idx == len(probs) - 1:
            return None
        return idx


# ======================================================================
# 1) Multinomial Logit: softmax(logits)
# ======================================================================
class MultinomialLogitChoiceModel(NormalizableChoiceModel):
    """Choice model using softmax(logits) probability.

    choice_features:
        - no_click_mass: logit for no-click option (default: -inf)
    """

    def __init__(self, choice_features):
        self._no_click_mass = choice_features.get("no_click_mass", -float("inf"))

    def score_documents(self, user_state, doc_obs):
        logits = self._score_documents_helper(user_state, doc_obs)
        logits = np.append(logits, self._no_click_mass)

        # stable softmax
        probs = softmax(logits)

        self._scores = probs[:-1]
        self._score_no_click = probs[-1]


# ======================================================================
# 2) Multinomial Proportional Model: positive scores only
# ======================================================================
class MultinomialProportionalChoiceModel(NormalizableChoiceModel):
    """Choice model with proportional probabilities:
        p(i) = (score_i - min_normalizer) / sum(...)
    """

    def __init__(self, choice_features):
        self._min_normalizer = choice_features.get("min_normalizer")
        self._no_click_mass = choice_features.get("no_click_mass", 0.0)

    def score_documents(self, user_state, doc_obs):
        scores = self._score_documents_helper(user_state, doc_obs)
        all_scores = np.append(scores, self._no_click_mass)

        # subtract minimum to ensure non-negative
        all_scores = all_scores - self._min_normalizer

        if np.any(all_scores < 0):
            raise ValueError("Normalized scores must be non-negative.")

        self._scores = all_scores[:-1]
        self._score_no_click = all_scores[-1]


# ======================================================================
# Cascade Models (Examine in order until a click occurs)
# ======================================================================
class CascadeChoiceModel(NormalizableChoiceModel):
    """Base for cascade choice models.

    Attributes:
        attention_prob: probability user examines next item
        score_scaling: scaling factor to convert score to probability
    """

    def __init__(self, choice_features):
        self._attention_prob = choice_features.get("attention_prob", 1.0)
        self._score_scaling = choice_features.get("score_scaling")

        if not (0.0 <= self._attention_prob <= 1.0):
            raise ValueError("attention_prob must be in [0, 1]")

        if self._score_scaling <= 0.0:
            raise ValueError("score_scaling must be positive")

    def _positional_normalization(self, scores):
        """Normalize scores as cascade click probabilities."""
        no_click_prob = 1.0
        click_probs = np.zeros_like(scores)

        for i in range(len(scores)):
            scaled = self._score_scaling * scores[i]
            if scaled > 1.0:
                raise ValueError(
                    f"score_scaling makes probability > 1: original={scores[i]}"
                )

            click_probs[i] = no_click_prob * self._attention_prob * scaled
            no_click_prob *= (1.0 - self._attention_prob * scaled)

        self._scores = click_probs
        self._score_no_click = no_click_prob


class ExponentialCascadeChoiceModel(CascadeChoiceModel):
    """Cascade model where score → exp(score)."""

    def score_documents(self, user_state, doc_obs):
        scores = self._score_documents_helper(user_state, doc_obs)
        scores = np.exp(scores)
        self._positional_normalization(scores)


class ProportionalCascadeChoiceModel(CascadeChoiceModel):
    """Cascade model where score → score - min_normalizer."""

    def __init__(self, choice_features):
        self._min_normalizer = choice_features.get("min_normalizer")
        super().__init__(choice_features)

    def score_documents(self, user_state, doc_obs):
        scores = self._score_documents_helper(user_state, doc_obs)
        scores = scores - self._min_normalizer

        if np.any(scores < 0):
            raise ValueError("Normalized scores must be non-negative.")

        self._positional_normalization(scores)
