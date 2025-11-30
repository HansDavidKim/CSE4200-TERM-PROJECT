# coding=utf-8
# Copyright 2019 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Classes to represent and interface with documents in RecSim."""

import abc
from gymnasium import spaces
import numpy as np


class CandidateSet:
    """Collection of AbstractDocuments indexed by document ID."""

    def __init__(self):
        self._documents = {}

    def size(self):
        return len(self._documents)

    def get_all_documents(self):
        return self.get_documents(self._documents.keys())

    def get_documents(self, document_ids):
        """Get documents by document IDs."""
        return [self._documents[int(doc_id)] for doc_id in document_ids]

    def add_document(self, document):
        self._documents[document.doc_id()] = document

    def remove_document(self, document):
        del self._documents[document.doc_id()]

    def create_observation(self):
        """Return dict of observable document features."""
        return {
            str(doc_id): doc.create_observation()
            for doc_id, doc in self._documents.items()
        }

    def observation_space(self):
        """Return gymnasium Dict space."""
        return spaces.Dict({
            str(doc_id): doc.observation_space()
            for doc_id, doc in self._documents.items()
        })


class AbstractDocumentSampler(abc.ABC):
    """Abstract class to sample documents."""

    def __init__(self, doc_ctor, seed=0):
        self._doc_ctor = doc_ctor
        self._seed = seed
        self.reset_sampler()

    def reset_sampler(self):
        self._rng = np.random.RandomState(self._seed)

    @abc.abstractmethod
    def sample_document(self):
        """Return an instance of AbstractDocument."""

    def get_doc_ctor(self):
        return self._doc_ctor

    @property
    def num_clusters(self):
        return 0

    def update_state(self, documents, responses):
        """Optional: update document states."""
        pass


class AbstractDocument(abc.ABC):
    """Base class representing a document with observable features."""

    NUM_FEATURES = None

    def __init__(self, doc_id):
        self._doc_id = doc_id  # unique integer id

    def doc_id(self):
        return self._doc_id

    @abc.abstractmethod
    def create_observation(self):
        """Return a numpy float array representing document features."""

    @classmethod
    @abc.abstractmethod
    def observation_space(cls):
        """Return gymnasium space representing document feature shapes."""
