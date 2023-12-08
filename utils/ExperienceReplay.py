from collections import deque
import random
import torch



class ExperienceReplay:
    def __init__(self, capacity=6_000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)  # automatic FIFO
        self.action_map = {"long": 1, "short": 0}

    def push(self, state, action, reward, next_state, done) -> None:

        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batchSize) -> tuple:
        # ex. [(1, 2, 3), (5, 6, 7), (7, 8, 9)] -> [(1, 5, 7), (2, 6, 8), (3, 7, 9)]
        # see https://stackoverflow.com/a/19343/3343043 for details
        state, action, reward, nextState, done = zip(*random.sample(self.memory, batchSize))
        stack_state = torch.stack(state)
        stack_action = torch.Tensor(list(map(self.action_map.get, action))).long()
        stack_reward = torch.Tensor(reward)
        stack_nextState = torch.stack(nextState)
        stack_done = torch.Tensor(done)

        return stack_state, stack_action, stack_reward, stack_nextState, stack_done

    def __len__(self) -> int:
        return len(self.memory)

    def reset(self) -> None:
        self.memory = deque(maxlen=self.capacity)



def test_experience_replay():
    capacity = 10
    replay = ExperienceReplay(capacity=capacity)

    for i in range(15):
        state = torch.randn(4)
        action = torch.tensor(i)
        reward = torch.tensor(float(i))
        next_state = torch.randn(4)
        done = torch.tensor(i % 2)
        replay.push(state, action, reward, next_state, done)

    assert len(replay) == capacity, "Length of replay buffer is incorrect."

    batch_size = 5
    states, actions, rewards, next_states, dones = replay.sample(batch_size)

    assert states.shape == (batch_size, 4), "Shape of states is incorrect."
    assert actions.shape == (batch_size,), "Shape of actions is incorrect."
    assert rewards.shape == (batch_size,), "Shape of rewards is incorrect."
    assert next_states.shape == (batch_size, 4), "Shape of next_states is incorrect."
    assert dones.shape == (batch_size,), "Shape of dones is incorrect."

    print("PASSED: ExperienceReplay functions correctly.")


if __name__ == "__main__":
    test_experience_replay()
