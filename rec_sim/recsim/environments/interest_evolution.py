# coding=utf-8
# Modernized RecSim - Interest Evolution Environment
# Python3 only, gymnasium, improved clarity

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from recsim import choice_model
from recsim import document
from recsim import user
from recsim import utils
from recsim.simulator import environment
from recsim.simulator import recsim_gym


# ============================================================
# Response Model
# ============================================================
class IEvResponse(user.AbstractResponse):
    """User response to a video."""

    MIN_QUALITY_SCORE = -100
    MAX_QUALITY_SCORE = 100

    def __init__(
        self,
        clicked=False,
        watch_time=0.0,
        liked=False,
        quality=0.0,
        cluster_id=0,
        click_doc_id=-1,
    ):
        self.clicked = clicked
        self.watch_time = watch_time
        self.liked = liked
        self.quality = quality
        self.cluster_id = cluster_id
        self.click_doc_id = click_doc_id  # 선택된 문서 ID (고정 아이템 id)

    def create_observation(self):
        return {
            "click": int(self.clicked),
            "watch_time": float(self.watch_time),
            "liked": int(self.liked),
            "quality": float(self.quality),
            "cluster_id": int(self.cluster_id),
            "click_doc_id": int(self.click_doc_id),
        }

    @classmethod
    def response_space(cls):
        return spaces.Dict(
            {
                "click": spaces.Discrete(2),
                "watch_time": spaces.Box(
                    low=0.0,
                    high=IEvVideo.MAX_VIDEO_LENGTH,
                    shape=(),
                    dtype=np.float32,
                ),
                "liked": spaces.Discrete(2),
                "quality": spaces.Box(
                    low=cls.MIN_QUALITY_SCORE,
                    high=cls.MAX_QUALITY_SCORE,
                    shape=(),
                    dtype=np.float32,
                ),
                "cluster_id": spaces.Discrete(IEvVideo.NUM_FEATURES),
                # click_doc_id: -1 = no click, 0 ~ num_candidates-1 = 아이템 id
                "click_doc_id": spaces.Box(
                    low=-1,
                    high=10000,
                    shape=(),
                    dtype=np.int32,
                ),
            }
        )


# ============================================================
# Document Model
# ============================================================
class IEvVideo(document.AbstractDocument):
    MAX_VIDEO_LENGTH = 100.0
    NUM_FEATURES = 20

    def __init__(
        self,
        doc_id,
        features,
        cluster_id=None,
        video_length=None,
        quality=None,
    ):
        self.features = np.array(features, dtype=np.float32)
        self.cluster_id = cluster_id
        self.video_length = video_length
        self.quality = quality
        super().__init__(doc_id)

    def create_observation(self):
        return self.features

    @classmethod
    def observation_space(cls):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(cls.NUM_FEATURES,),
            dtype=np.float32,
        )


# ============================================================
# Document Samplers
# ============================================================
class IEvVideoSampler(document.AbstractDocumentSampler):
    """Random video generator (일반 버전, 지금은 create_environment에서 사용 안 함)."""

    def __init__(
        self,
        doc_ctor=IEvVideo,
        num_candidates=20,  # item 수 추가
        min_feature_value=-1.0,
        max_feature_value=1.0,
        video_length_mean=4.3,
        video_length_std=1.0,
        **kwargs
    ):
        super().__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        self._num_candidates = num_candidates
        self._min_feature = min_feature_value
        self._max_feature = max_feature_value
        self._len_mean = video_length_mean
        self._len_std = video_length_std

    def sample_document(self):
        features = self._rng.uniform(
            self._min_feature,
            self._max_feature,
            self.get_doc_ctor().NUM_FEATURES,
        )

        video_length = min(
            self._rng.normal(self._len_mean, self._len_std),
            self.get_doc_ctor().MAX_VIDEO_LENGTH,
        )

        # item 개수 = candidate 수로 고정
        doc_id = self._doc_count % self._num_candidates

        doc = self._doc_ctor(
            doc_id=doc_id,
            features=features,
            cluster_id=None,
            video_length=video_length,
            quality=1.0,
        )
        self._doc_count += 1
        return doc


class UtilityModelVideoSampler(document.AbstractDocumentSampler):
    """Videos sampled for utility model experiments."""

    def __init__(
        self,
        doc_ctor=IEvVideo,
        num_candidates=20,        # ★ 추가: 고정 아이템 수
        min_utility=-3.0,
        max_utility=3.0,
        video_length=4.0,
        **kwargs
    ):
        super().__init__(doc_ctor, **kwargs)
        self._doc_count = 0
        self._num_clusters = self.get_doc_ctor().NUM_FEATURES
        self._min_u = min_utility
        self._max_u = max_utility
        self._video_len = video_length
        self._num_candidates = num_candidates  # ★ 저장

        trashy = np.linspace(self._min_u, 0, int(self._num_clusters * 0.7))
        nutritious = np.linspace(0, self._max_u, int(self._num_clusters * 0.3))
        self.cluster_means = np.concatenate([trashy, nutritious])

    def sample_document(self):
        cid = self._rng.randint(0, self._num_clusters)
        features = np.zeros(self._num_clusters, dtype=np.float32)
        features[cid] = 1.0
        quality = self._rng.normal(self.cluster_means[cid], 0.1)

        # doc_id를 0 ~ num_candidates-1 범위로 고정
        # 원래는 doc_count += 1 되면 item 증가
        doc_id = self._doc_count % self._num_candidates

        doc = self._doc_ctor(
            doc_id=doc_id,
            features=features,
            cluster_id=cid,
            video_length=self._video_len,
            quality=quality,
        )
        self._doc_count += 1
        return doc


# ============================================================
# User State
# ============================================================
class IEvUserState(user.AbstractUserState):

    NUM_FEATURES = 20

    def __init__(
        self,
        user_interests,
        time_budget=None,
        score_scaling=None,
        attention_prob=None,
        no_click_mass=None,
        keep_interact_prob=None,
        min_doc_utility=None,
        user_update_alpha=None,
        watched_videos=None,
        impressed_videos=None,
        liked_videos=None,
        step_penalty=None,
        min_normalizer=None,
        user_quality_factor=None,
        document_quality_factor=None,
    ):
        self.user_interests = np.array(user_interests, dtype=np.float32)
        self.time_budget = float(time_budget)
        self.keep_interact_prob = keep_interact_prob
        self.min_doc_utility = min_doc_utility

        self.choice_features = {
            "score_scaling": score_scaling,
            "attention_prob": attention_prob,
            "no_click_mass": no_click_mass,
            "min_normalizer": min_normalizer,
        }

        self.user_update_alpha = user_update_alpha
        self.step_penalty = step_penalty
        self.user_quality_factor = user_quality_factor
        self.document_quality_factor = document_quality_factor

        self.watched_videos = watched_videos or set()
        self.impressed_videos = impressed_videos or set()
        self.liked_videos = liked_videos or set()

    def score_document(self, doc_obs):
        return float(np.dot(self.user_interests, doc_obs))

    def create_observation(self):
        return self.user_interests

    @classmethod
    def observation_space(cls):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(cls.NUM_FEATURES,),
            dtype=np.float32,
        )


# ============================================================
# User Sampler
# ============================================================
class UtilityModelUserSampler(user.AbstractUserSampler):

    def __init__(
        self,
        user_ctor=IEvUserState,
        document_quality_factor=1.0,
        no_click_mass=1.0,
        min_normalizer=-1.0,
        **kwargs
    ):
        self._no_click_mass = no_click_mass
        self._min_normalizer = min_normalizer
        self._doc_quality_factor = document_quality_factor
        super().__init__(user_ctor, **kwargs)

    def sample_user(self):
        interests = self._rng.uniform(
            -1.0, 1.0, self.get_user_ctor().NUM_FEATURES
        )

        utility_norm = 1.0 / 3.4
        alpha = 0.9 * utility_norm

        return self._user_ctor(
            user_interests=interests,
            time_budget=200.0,
            no_click_mass=self._no_click_mass,
            step_penalty=0.5,
            score_scaling=0.05,
            attention_prob=0.65,
            min_normalizer=self._min_normalizer,
            user_quality_factor=0.0,
            document_quality_factor=self._doc_quality_factor,
            user_update_alpha=alpha,
        )


# ============================================================
# User Model
# ============================================================
class IEvUserModel(user.AbstractUserModel):
    """User dynamics model for interest evolution."""

    def __init__(
        self,
        slate_size,
        choice_model_ctor,
        response_model_ctor=IEvResponse,
        user_state_ctor=IEvUserState,
        no_click_mass=1.0,
        seed=0,
        alpha_x_intercept=1.0,
        alpha_y_intercept=0.3,
    ):
        super().__init__(
            response_model_ctor,
            UtilityModelUserSampler(
                user_ctor=user_state_ctor,
                no_click_mass=no_click_mass,
                seed=seed,
            ),
            slate_size,
        )

        self.choice_model = choice_model_ctor(self._user_state.choice_features)
        self._alpha_x = alpha_x_intercept
        self._alpha_y = alpha_y_intercept

    def is_terminal(self):
        return self._user_state.time_budget <= 0

    def simulate_response(self, documents):
        """한 스텝에서 slate에 대해 응답 생성 + click_doc_id 세팅."""
        responses = [self._response_model_ctor() for _ in documents]

        doc_obs = [d.create_observation() for d in documents]
        self.choice_model.score_documents(self._user_state, doc_obs)
        selected_idx = self.choice_model.choose_item()

        for i, resp in enumerate(responses):
            resp.quality = documents[i].quality
            resp.cluster_id = documents[i].cluster_id

        if selected_idx is not None:
            clicked_doc = documents[selected_idx]
            responses[selected_idx].clicked = True
            responses[selected_idx].click_doc_id = clicked_doc.doc_id()
            responses[selected_idx].watch_time = min(
                self._user_state.time_budget, clicked_doc.video_length
            )

        return responses

    def update_state(self, slate_docs, responses):
        """사용자 상태 업데이트 (관심도 + time_budget)."""
        user_state = self._user_state

        def compute_alpha(x):
            return (-self._alpha_y / self._alpha_x) * np.abs(x) + self._alpha_y

        for doc, response in zip(slate_docs, responses):
            if response.clicked:
                # score_documents는 self.choice_model.scores를 세팅하고, 반환값은 None
                self.choice_model.score_documents(
                    user_state, [doc.create_observation()]
                )
                expected_utility = self.choice_model.scores[0]

                mask = doc.features
                target = doc.features - user_state.user_interests
                alpha = compute_alpha(user_state.user_interests)

                # Use RNG from user_sampler
                if self._user_sampler._rng.random() < np.dot(
                    (user_state.user_interests + 1.0) * 0.5, mask
                ):
                    user_state.user_interests += alpha * mask * target
                else:
                    user_state.user_interests -= alpha * mask * target

                user_state.user_interests = np.clip(
                    user_state.user_interests, -1.0, 1.0
                )

                received = (
                    user_state.user_quality_factor * expected_utility
                    + user_state.document_quality_factor * doc.quality
                )

                user_state.time_budget -= response.watch_time
                user_state.time_budget += (
                    user_state.user_update_alpha
                    * response.watch_time
                    * received
                )
                return

        # no click
        user_state.time_budget -= user_state.step_penalty


# ============================================================
# Reward
# ============================================================
def clicked_watchtime_reward(responses):
    return sum(r.watch_time for r in responses if r.clicked)


# ============================================================
# Environment Factory
# ============================================================
def create_environment(env_config):

    user_model = IEvUserModel(
        slate_size=env_config["slate_size"],
        choice_model_ctor=choice_model.MultinomialProportionalChoiceModel,
        response_model_ctor=IEvResponse,
        user_state_ctor=IEvUserState,
        seed=env_config["seed"],
        alpha_y_intercept=0.3 * env_config.get("drift_scale", 1.0), # Scale drift
    )

    doc_sampler = UtilityModelVideoSampler(
        doc_ctor=IEvVideo,
        num_candidates=env_config["num_candidates"],  # ★ 여기서 전달
        seed=env_config["seed"],
    )

    recsim_env = environment.Environment(
        user_model=user_model,
        document_sampler=doc_sampler,
        num_candidates=env_config["num_candidates"],
        slate_size=env_config["slate_size"],
        resample_documents=env_config["resample_documents"],
    )

    return recsim_gym.RecSimGymEnv(
        recsim_env,
        reward_aggregator=clicked_watchtime_reward,
        metrics_aggregator=utils.aggregate_video_cluster_metrics,
        metrics_writer=utils.write_video_cluster_metrics,
    )
