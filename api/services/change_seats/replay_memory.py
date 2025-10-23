from collections import deque
import random

class ReplayMemory:

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """経験をメモリに追加する"""
        # (state, action, reward, next_state, done) のタプルとして保存
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """メモリからランダムに指定された数の経験を抜き出す"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """メモリの現在のサイズを返す"""
        return len(self.memory)