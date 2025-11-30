# coding=utf-8
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
r"""An executable class to run agents in the simulator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

from absl import flags
import gin.torch
import logging
from gymnasium import spaces
import numpy as np
import glob
from recsim.simulator import environment # type: ignore

from torch.utils.tensorboard import SummaryWriter
import torch
import json

flags.DEFINE_bool(
    'debug_mode', False,
    'If set to true, the agent will output in-episode statistics '
    'to Tensorboard. Disabled by default as this results in '
    'slower training.')
flags.DEFINE_string('agent_name', None, 'Name of the agent.')
flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string(
    'environment_name', 'interest_evolution',
    'The environment with which to run the experiment. Supported choices are '
    '{interest_evolution, interest_exploration}.')
flags.DEFINE_string(
    'episode_log_file', '',
    'Filename under base_dir to output simulated episodes in SequenceExample.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
    '"third_party/py/dopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "runner_lib.Runner.max_steps_per_episode=100')


FLAGS = flags.FLAGS


def load_gin_configs(gin_files, gin_bindings):
  """Loads gin configuration files.

  Args:
    gin_files: list, of paths to the gin configuration files for this
      experiment.
    gin_bindings: list, of gin parameter bindings to override the values in the
      config files.
  """
  gin.parse_config_files_and_bindings(
      gin_files, bindings=gin_bindings, skip_unknown=False)


@gin.configurable
class Runner(object):
  """Object that handles running experiments.

  Here we use the term 'experiment' to mean simulating interactions between the
  agent and the environment and reporting some statistics pertaining to these
  interactions.
  """

  _output_dir = None
  _checkpoint_dir = None
  _agent = None
  _checkpointer = None

  def __init__(self,
               base_dir,
               create_agent_fn,
               env,
               episode_log_file='',
               checkpoint_file_prefix='ckpt',
               max_steps_per_episode=27000):
    """Initializes the Runner object in charge of running a full experiment.

    Args:
      base_dir: str, the base directory to host all required sub-directories.
      create_agent_fn: A function that takes as args a Tensorflow session and an
        environment, and returns an agent.
      env: A Gym environment for running the experiments.
      episode_log_file: Path to output simulated episodes in tf.SequenceExample.
        Disable logging if episode_log_file is an empty string.
      checkpoint_file_prefix: str, the prefix to use for checkpoint files.
      max_steps_per_episode: int, maximum number of steps after which an episode
        terminates.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info('max_steps_per_episode = %s', max_steps_per_episode)

    if base_dir is None:
      raise ValueError('Missing base_dir.')

    self._base_dir = base_dir
    self._create_agent_fn = create_agent_fn
    self._env = env
    self._checkpoint_file_prefix = checkpoint_file_prefix
    self._max_steps_per_episode = max_steps_per_episode
    self._episode_log_file = episode_log_file
    self._episode_writer = None

  def _set_up(self, eval_mode):
    """Sets up the runner by creating and initializing the agent."""
    # Reset the tf default graph to avoid name collisions from previous runs
    # before doing anything else.
    self._summary_writer = SummaryWriter(log_dir=self._output_dir)
    if self._episode_log_file:
        self._episode_writer = open(
            os.path.join(self._output_dir, self._episode_log_file), 'wb')
    # Set up a session and initialize variables
    self._agent = self._create_agent_fn(
        self._env,
        summary_writer=self._summary_writer,
        eval_mode=eval_mode)
    # type check: env/agent must both be multi- or single-user
    if self._agent.multi_user and not isinstance(
        self._env.environment, environment.MultiUserEnvironment):
      raise ValueError('Multi-user agent requires multi-user environment.')
    if not self._agent.multi_user and isinstance(
        self._env.environment, environment.MultiUserEnvironment):
      raise ValueError('Single-user agent requires single-user environment.')

  def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
    """Reloads the latest checkpoint if it exists.

    This method will first create a `Checkpointer` object and then call
    `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
    checkpoint in self._checkpoint_dir, and what the largest file number is.
    If a valid checkpoint file is found, it will load the bundled data from this
    file and will pass it to the agent for it to reload its data.
    If the agent is able to successfully unbundle, this method will increase and
    return the iteration number keyed by 'current_iteration' and the step number
    keyed by 'total_steps' as the return values.

    Args:
      checkpoint_file_prefix: str, the checkpoint file prefix.
    Returns:
      start_iteration: The iteration number to be continued after the latest
        checkpoint.
      start_step: The step number to be continued after the latest checkpoint.
    """
    def get_latest_checkpoint_number(checkpoint_dir, prefix):
        files = glob.glob(os.path.join(checkpoint_dir, f"{prefix}_*.pth"))
        if not files:
            return -1
        versions = []
        for f in files:
            basename = os.path.basename(f)
            try:
                num = int(basename.replace(prefix + "_", "").replace(".pth", ""))
                versions.append(num)
            except ValueError:
                continue
        return max(versions) if versions else -1

    start_iteration = 0
    start_step = 0
    # Check if checkpoint exists.
    # Note that the existence of checkpoint 0 means that we have finished
    # iteration 0 (so we will start from iteration 1).
    latest_checkpoint_version = get_latest_checkpoint_number(
        self._checkpoint_dir, checkpoint_file_prefix)
    if latest_checkpoint_version >= 0:
      assert not self._episode_writer, 'Can only log episodes from scratch.'
      checkpoint_path = os.path.join(self._checkpoint_dir, f"{checkpoint_file_prefix}_{latest_checkpoint_version}.pth")
      experiment_data = torch.load(checkpoint_path, map_location='cpu')

      start_iteration = experiment_data.get('current_iteration', 0) + 1
      experiment_data.pop('current_iteration', None)

      start_step = experiment_data.get('total_steps', 0) + 1
      experiment_data.pop('total_steps', None)

      if self._agent.unbundle(self._checkpoint_dir, latest_checkpoint_version,
                              experiment_data):
        logging.info(
            'Reloaded checkpoint and will start from '
            'iteration %d', start_iteration)
    return start_iteration, start_step

  def _log_one_step(self, user_obs, doc_obs, slate, responses, reward,
                    is_terminal, episode_log):
    """Adds one step of agent-environment interaction into SequenceExample.

    Args:
      user_obs: An array of floats representing user state observations
      doc_obs: A list of observations of the documents
      slate: An array of indices to doc_obs
      responses: A list of observations of responses for items in the slate
      reward: A float for the reward returned after this step
      is_terminal: A boolean for whether a terminal state has been reached
      sequence_example: A SequenceExample proto for logging current episode
    """
    if self._episode_writer is None:
        return

    step_data = {}

    if isinstance(self._env.environment, environment.MultiUserEnvironment):
      step_data["users"] = []

      for i, (single_user,
              single_slate,
              single_user_responses,
              single_reward) in enumerate(zip(user_obs,
                                              slate,
                                              responses,
                                              reward)):
        user_space = list(self._env.observation_space.spaces['user'].spaces)[i]
        resp_space = self._env.observation_space.spaces['response'][i][0]
        user_flat = spaces.flatten(user_space, single_user)

        flattened_responses = []
        for response in single_user_responses:
            resp_flat = spaces.flatten(resp_space, response)
            flattened_responses.append(resp_flat)

        step_data["users"].append({
            "user": user_flat,
            "slate": single_slate,
            "reward": single_reward,
            "responses": flattened_responses,
        })
    else:  # single-user environment
        user_flat = spaces.flatten(self._env.observation_space.spaces['user'], user_obs)
        resp_space = self._env.observation_space.spaces['response'][0]

        flattened_responses = []
        for response in responses:
            resp_flat = spaces.flatten(resp_space, response)
            flattened_responses.append(resp_flat)

        step_data.update({
            "user": user_flat,
            "slate": slate,
            "reward": reward,
            "responses": flattened_responses,
        })

    doc_list = []
    for i, doc in enumerate(doc_obs.values()):
        doc_space = list(self._env.observation_space.spaces['doc'].spaces.values())[i]
        doc_list.append(spaces.flatten(doc_space, doc))
    step_data["docs"] = doc_list

    step_data["is_terminal"] = is_terminal
    episode_log.append(step_data)
  def _run_one_episode(self):
    """Executes a full trajectory of the agent interacting with the environment.

    Returns:
      The number of steps taken and the total reward.
    """
    step_number = 0
    total_reward = 0.

    start_time = time.time()

    episode_log = []
    observation = self._env.reset()
    action = self._agent.begin_episode(observation)

    # Keep interacting until we reach a terminal state.
    while True:
      last_observation = observation
      observation, reward, done, info = self._env.step(action)
      self._log_one_step(last_observation['user'], last_observation['doc'],
                         action, observation['response'], reward, done,
                         episode_log)
      # Update environment-specific metrics with responses to the slate.
      self._env.update_metrics(observation['response'], info)

      total_reward += reward
      step_number += 1

      if done:
        break
      elif step_number == self._max_steps_per_episode:
        # Stop the run loop once we reach the true end of episode.
        break
      else:
        action = self._agent.step(reward, observation)

    self._agent.end_episode(reward, observation)

    if self._episode_writer is not None:
        for step_data in episode_log:
            self._episode_writer.write(json.dumps(step_data) + "\n")
        self._episode_writer.flush()

    time_diff = time.time() - start_time

    self._update_episode_metrics(
        episode_length=step_number,
        episode_time=time_diff,
        episode_reward=total_reward)

    return step_number, total_reward

  def _initialize_metrics(self):
    """Initializes the metrics."""
    self._stats = {
        'episode_length': [],
        'episode_time': [],
        'episode_reward': [],
    }
    # Initialize environment-specific metrics.
    self._env.reset_metrics()

  def _update_episode_metrics(self, episode_length, episode_time,
                              episode_reward):
    """Updates the episode metrics with one episode."""

    self._stats['episode_length'].append(episode_length)
    self._stats['episode_time'].append(episode_time)
    self._stats['episode_reward'].append(episode_reward)

  def _write_metrics(self, step, suffix):
    """Writes the metrics to Tensorboard summaries."""

    def add_summary(tag, value):
      self._summary_writer.add_scalar(f"{tag}/{suffix}", value, step)

    num_steps = np.sum(self._stats['episode_length'])
    time_per_step = np.sum(self._stats['episode_time']) / num_steps

    add_summary('TimePerStep', time_per_step)
    add_summary('AverageEpisodeLength', np.mean(self._stats['episode_length']))
    add_summary('AverageEpisodeRewards', np.mean(self._stats['episode_reward']))
    add_summary('StdEpisodeRewards', np.std(self._stats['episode_reward']))

    # Environment-specific Tensorboard summaries.
    self._env.write_metrics(add_summary)

    self._summary_writer.flush()

  def _checkpoint_experiment(self, iteration, total_steps):
    """Checkpoints experiment data.

    Args:
      iteration: int, iteration number for checkpointing.
      total_steps: int, total number of steps for all iterations so far.
    """
    experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir,
                                                        iteration)
    if experiment_data:
      experiment_data['current_iteration'] = iteration
      experiment_data['total_steps'] = total_steps
      self._checkpointer.save_checkpoint(iteration, experiment_data)


@gin.configurable
class TrainRunner(Runner):
  """Object that handles running the training.

  See main.py for a simple example to train an agent.
  """

  def __init__(self, max_training_steps=250000, num_iterations=100,
               checkpoint_frequency=1, **kwargs):
    logging.info(
        'max_training_steps = %s, number_iterations = %s,'
        'checkpoint frequency = %s iterations.', max_training_steps,
        num_iterations, checkpoint_frequency)

    super(TrainRunner, self).__init__(**kwargs)
    self._max_training_steps = max_training_steps
    self._num_iterations = num_iterations
    self._checkpoint_frequency = checkpoint_frequency

    self._output_dir = os.path.join(self._base_dir, 'train')
    self._checkpoint_dir = os.path.join(self._output_dir, 'checkpoints')

    self._set_up(eval_mode=False)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning training...')
    start_iter, total_steps = self._initialize_checkpointer_and_maybe_resume(
        self._checkpoint_file_prefix)
    if self._num_iterations <= start_iter:
      logging.warning('num_iterations (%d) < start_iteration(%d)',
                         self._num_iterations, start_iter)
      return

    for iteration in range(start_iter, self._num_iterations):
      logging.info('Starting iteration %d', iteration)
      total_steps = self._run_train_phase(total_steps)
      if iteration % self._checkpoint_frequency == 0:
        self._checkpoint_experiment(iteration, total_steps)

  def _run_train_phase(self, total_steps):
    """Runs training phase and updates total_steps."""

    self._initialize_metrics()

    num_steps = 0

    while num_steps < self._max_training_steps:
      episode_length, _ = self._run_one_episode()
      num_steps += episode_length

    total_steps += num_steps
    self._write_metrics(total_steps, suffix='train')
    return total_steps


@gin.configurable
class EvalRunner(Runner):
  """Object that handles running the evaluation.

  See main.py for a simple example to evaluate an agent.
  """

  def __init__(self,
               max_eval_episodes=125000,
               test_mode=False,
               min_interval_secs=30,
               train_base_dir=None,
               **kwargs):
    logging.info('max_eval_episodes = %s', max_eval_episodes)
    super(EvalRunner, self).__init__(**kwargs)
    self._max_eval_episodes = max_eval_episodes
    self._test_mode = test_mode
    self._min_interval_secs = min_interval_secs

    self._output_dir = os.path.join(self._base_dir,
                                    'eval_%s' % max_eval_episodes)
    os.makedirs(self._output_dir, exist_ok=True)
    if train_base_dir is None:
      train_base_dir = self._base_dir
    self._checkpoint_dir = os.path.join(train_base_dir, 'train', 'checkpoints')

    self._set_up(eval_mode=True)

  def run_experiment(self):
    """Runs a full experiment, spread over multiple iterations."""
    logging.info('Beginning evaluation...')

    def get_latest_checkpoint_number(checkpoint_dir, prefix):
        files = glob.glob(os.path.join(checkpoint_dir, f"{prefix}_*.pth"))
        if not files:
            return -1
        versions = []
        for f in files:
            basename = os.path.basename(f)
            try:
                num = int(basename.replace(prefix + "_", "").replace(".pth", ""))
                versions.append(num)
            except ValueError:
                continue
        return max(versions) if versions else -1
    # Use the checkpointer class
    checkpoint_version = -1
    # Check new checkpoints in a loop.
    while True:
      # Check if checkpoint exists.
      # Note that the existence of checkpoint 0 means that we have finished
      # iteration 0 (so we will start from iteration 1).
      latest_checkpoint_version = get_latest_checkpoint_number(
          self._checkpoint_dir,self._checkpoint_file_prefix)
      # checkpoints_iterator already makes sure a new checkpoint exists.
      if latest_checkpoint_version <= checkpoint_version:
        time.sleep(self._min_interval_secs)
        continue
      checkpoint_version = latest_checkpoint_version
      checkpoint_path = os.path.join(self._checkpoint_dir, f"{self._checkpoint_file_prefix}_{checkpoint_version}.pth")
      experiment_data = torch.load(checkpoint_path, map_location='cpu')

      if hasattr(self._agent, 'load_state_dict') and 'agent_state_dict' in experiment_data:
          self._agent.load_state_dict(experiment_data['agent_state_dict'])
          logging.info(f"Loaded checkpoint version {checkpoint_version}")
      else:
          raise RuntimeError("Agent state dict not found in checkpoint")

      self._run_eval_phase(experiment_data['total_steps'])

      if self._test_mode:
        break

  def _run_eval_phase(self, total_steps):
    """Runs evaluation phase given model has been trained for total_steps."""

    self._env.reset_sampler()
    self._initialize_metrics()

    num_episodes = 0
    episode_rewards = []

    while num_episodes < self._max_eval_episodes:
      _, episode_reward = self._run_one_episode()
      episode_rewards.append(episode_reward)
      num_episodes += 1

    self._write_metrics(total_steps, suffix='eval')

    output_file = os.path.join(self._output_dir, 'returns_%s' % total_steps)
    logging.info(f'eval_file: {output_file}')
    with open(output_file, 'w') as f:
        f.write(str(episode_rewards))