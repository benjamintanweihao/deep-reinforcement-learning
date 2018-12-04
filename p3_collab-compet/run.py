import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm

from unityagents import UnityEnvironment
from collections import deque
from ddpg_agent import Agent

RANDOM = False
TRAIN = True
TEST = True


def init_environment_and_agent():
    env = UnityEnvironment(file_name='Tennis_Linux/Tennis.x86_64', no_graphics=False)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents: {}'.format(num_agents))

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action {}'.format(action_size))

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]

    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    # print('The state for the first agent looks like: {}'.format(state[0]))

    seed = random.randint(0, 1000)
    print('Using random seed: {}'.format(seed))
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed)

    return env, agent


def random_agent(env):
    """
    Random environment
    :param env:
    :return:
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    num_agents = len(env_info.agents)
    action_size = brain.vector_action_space_size

    for i in range(1, 6):                                      # play game for 5 episodes
        env_info = env.reset(train_mode=False)[brain_name]     # reset the environment
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        while True:
            actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
            actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)
            dones = env_info.local_done                        # see if episode finished
            scores += env_info.rewards                         # update the score (for each agent)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

    env.close()


def ddpg(env, agent, n_episodes=1000, max_t=1000, goal_score=0.5, learn_every=50, num_learn=10):
    brain_name = env.brain_names[0]
    total_scores_deque = deque(maxlen=100)
    total_scores = []

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        num_agents = len(env_info.agents)
        states = env_info.vector_observations
        scores = np.zeros(num_agents)

        agent.reset()

        for t in tqdm.tqdm(range(max_t), leave=False):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done

            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done)

            states = next_states
            scores += env_info.rewards

            if t % learn_every == 0:
                for _ in range(num_learn):
                    agent.step_learn(10)

            if np.any(dones):
                break

        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        total_scores_deque.append(mean_score)
        total_scores.append(mean_score)
        total_average_score = np.mean(total_scores_deque)

        print('\rEpisode {}\tAverage: {:.2f}\tMin: {:.2f}\tMax: {:.2f}'.format(
            i_episode, total_average_score, min_score, max_score))

        if total_average_score >= goal_score and i_episode >= 100:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('Woot! Solved after {} episodes. Total average score: {}'.format(
                i_episode, total_average_score))

            break

        if i_episode % 10 == 0:
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            print('Saving Episode: {}.'.format(i_episode))

    return total_scores


def ddpg_test(env, agent):
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth', map_location='cpu'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth', map_location='cpu'))

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    num_agents = len(env_info.agents)

    states = env_info.vector_observations
    scores = np.zeros(num_agents)

    while True:
        actions = agent.act(states, add_noise=False)
        env_info = env.step(actions)[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done

        scores += rewards
        states = next_states

        if np.any(dones):
            break

    print('Total Score: {}'.format(np.mean(scores)))

    env.close()

    return


if RANDOM:
    env_, agent_ = init_environment_and_agent()
    random_agent(env_)

if TRAIN:
    env_, agent_ = init_environment_and_agent()
    scores = ddpg(env_, agent_)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

if TEST:
    # Run the test
    ddpg_test(env_, agent_)
