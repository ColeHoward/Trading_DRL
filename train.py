import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math
from tqdm import tqdm
from evaluate import evaluate
import time


def train(model, env_generator, true_env, device, tracker, M):
    # set random seed
    random.seed(0)
    torch.manual_seed(0)
    gamma = .4
    opposite = lambda x: "long" if x == "short" else "short"
    batch_size = 64
    lr = 1e-4
    print('Using Device:', device)
    target_network_update_freq = 2000
    eps_start, eps_end, eps_decay = 1, .01, 10_000
    decay_eps = lambda iteration: eps_end + (eps_start - eps_end) * math.exp(-1 * iteration / eps_decay)
    tot_iter = 0
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=.000001)
    train_freq = 1
    alpha = .1
    num_epochs = 30
    start = time.time()

    for epoch in range(num_epochs):
        for env in env_generator:
            state = env.reset()
            done = False
            t = 0
            prev_action = None
            action_counts = {"long": 0, "short": 0}
            while not done:
                tot_iter += 1
                if random.random() < decay_eps(tot_iter):  # Epsilon greedy
                    action = random.choice(env.actionspace)
                else:
                    if random.random() < alpha and prev_action is not None:  # sticky actions
                        action = prev_action
                    else:
                        with torch.no_grad():
                            action_values = model.main_pred(state.to(device).unsqueeze(0))
                            action_idx = action_values.squeeze(0).argmax()
                            action = env.actionspace[action_idx]
                            action_counts[action] += 1
                prev_action = action

                # generate experience
                next_state, reward, done, opp = env.step(action)
                if not done:
                    experience = (state, action, reward, next_state, done)
                    opp_experience = (state, opposite(action), opp['reward'], opp['state'], done)
                    M.push(*experience)
                    M.push(*opp_experience)
                if t % 100 == 0:
                    tqdm.write(f"Epoch {epoch + 1}: {env} {action_counts}")
                if t == 100:
                    break
                t += 1
                state = next_state

                if len(M) >= batch_size and t % train_freq == 0:
                    states, actions, rewards, next_states, dones = M.sample(batch_size)
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    next_states = next_states.to(device)
                    dones = dones.to(device)

                    # Compute the target Q values for next states
                    next_state_values = model.tgt_pred(next_states)
                    max_next_state_values = next_state_values.max(1)[0]
                    y_i = rewards + gamma * max_next_state_values * (1 - dones)

                    # Compute the predicted Q values from the policy network for current states and actions
                    y_hat = model.main_pred(states.to(device)).gather(1, actions.unsqueeze(1)).squeeze(1)

                    # Compute loss
                    loss = F.smooth_l1_loss(y_hat, y_i)

                    # Optimization step
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                    optimizer.step()

                if tot_iter % target_network_update_freq == 0 and tot_iter > 0:
                    model.update_tgt()

            evaluate(env, model, tracker, epoch, False, device, true_env)
            evaluate(env, model, tracker, epoch, True, device, true_env)

    tracker.write_metrics(f'results/{true_env.ticker}_train.csv', True)
    tracker.write_metrics(f'results/{true_env.ticker}_test.csv', False)

    print('Duration:', time.time() - start)
