import json
import csv
import numpy as np
from tqdm import tqdm

from recsim.environments.interest_evolution import create_environment


def convert_doc_obs(doc_obs):
    """OrderedDict → dict of lists 변환."""
    return {int(k): v.tolist() for k, v in doc_obs.items()}


def run_single_episode(env, steps=100):
    """한 명의 유저에 대해 steps 만큼 시뮬레이션"""
    obs = env.reset()

    slate_size = env._environment.slate_size
    num_candidates = env._environment.num_candidates

    episode_data = []

    for t in range(steps):
        # action: 무작위 slate 선택
        action = np.random.choice(num_candidates, size=slate_size, replace=False)

        # gymnasium step() → 5개 값 반환
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # user obs (numpy → list)
        user_obs = (
            obs["user"].tolist() if isinstance(obs["user"], np.ndarray)
            else obs["user"]
        )

        # response 그대로 유지
        responses = obs["response"]

        episode_data.append({
            "step": t,
            "action": action.tolist(),
            "user": user_obs,
            "response": responses,
            "reward": reward
        })

        if done:
            break

    return episode_data


def simulate_users_json(
    slate_size=5,
    num_candidates=20,
    steps=20,
    num_users=100,
    file_name="sim_output.jsonl",
    global_seed=42,
    sim_seed=1,
):
    """
    전체 시뮬레이션 API — JSONL 저장 (CSV 포맷과 동일한 attribute)
    문서 feature 제거 버전
    """

    np.random.seed(global_seed)

    env_config = {
        "slate_size": slate_size,
        "num_candidates": num_candidates,
        "resample_documents": False,
        "seed": sim_seed,
    }

    env = create_environment(env_config)

    with open(file_name, "w", encoding="utf-8") as f:
        for user_id in tqdm(range(num_users), desc="Simulating Users (JSON)"):

            episode = run_single_episode(env, steps=steps)

            json_steps = []
            for step_data in episode:

                # CSV 구조와 동일하게 변환
                step_json = {
                    "step": step_data["step"],
                    "user": step_data["user"],  # 20 floats
                    "action": ",".join(map(str, step_data["action"])),
                    "reward": step_data["reward"],
                    "response": []
                }

                # responses : CSV와 완벽히 동일
                for resp in step_data["response"]:
                    step_json["response"].append({
                        "click": resp["click"],
                        "click_doc_id": resp["click_doc_id"],
                        "watch_time": float(resp["watch_time"]),
                        "liked": resp["liked"],
                        "quality": float(resp["quality"]),
                        "cluster_id": resp["cluster_id"],
                    })

                json_steps.append(step_json)

            record = {
                "user_id": user_id,
                "steps": json_steps
            }

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[JSON] 저장 완료: {file_name}")





def simulate_users_csv(
    slate_size=5,
    num_candidates=20,
    steps=100,
    num_users=100,
    file_name="sim_output.csv",
    global_seed=42,
    sim_seed=1,
    drift_scale=1.0,
):
    """문서 feature 제거 버전 CSV 저장 함수"""

    np.random.seed(global_seed)

    env_config = {
        "slate_size": slate_size,
        "num_candidates": num_candidates,
        "resample_documents": False,
        "seed": sim_seed,
        "drift_scale": drift_scale,
    }

    env = create_environment(env_config)

    # CSV header 생성
    user_cols = [f"user_{i}" for i in range(20)]
    action_cols = ["action"]
    reward_col = ["reward"]

    # response columns only (문서 feature 제거)
    resp_cols = []
    for i in range(slate_size):
        resp_cols += [
            f"resp_{i}_click",
            f"resp_{i}_click_doc_id",     # ← 추가
            f"resp_{i}_watch",
            f"resp_{i}_liked",
            f"resp_{i}_quality",
            f"resp_{i}_cluster",
        ]

    header = ["user_id", "step"] + user_cols + action_cols + reward_col + resp_cols

    with open(file_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for user_id in tqdm(range(num_users), desc="Simulating Users (CSV)"):
            episode = run_single_episode(env, steps=steps)

            for step_data in episode:
                row = []

                # 기본 정보
                row += [user_id, step_data["step"]]

                # user state
                row += step_data["user"]

                # action (slate 문서 id 5개 → "12,3,7,18,5")
                row += [",".join(map(str, step_data["action"]))]

                # reward
                row += [step_data["reward"]]

                # response flatten
                responses = step_data["response"]
                for resp in responses:
                    row += [
                        resp["click"],
                        resp["click_doc_id"], 
                        float(resp["watch_time"]),
                        resp["liked"],
                        float(resp["quality"]),
                        resp["cluster_id"],
                    ]

                writer.writerow(row)

    print(f"[CSV] 저장 완료: {file_name}")



