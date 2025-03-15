import collections
import typing
import interface
import numpy as np
import sandbox_rl.core.interfaces


class ReplayBuffer(interface.implements(sandbox_rl.core.interfaces.IReplayBuffer)):
    def __init__(self, max_size: int = 10000) -> None:
        self.buffer = collections.deque(maxlen=max_size)

    def store(self, data: typing.Tuple) -> None:
        self.buffer.append(data)

    def sample(self, batch_size: int) -> typing.List[typing.Tuple]:
        batch_size = min(batch_size, len(self))
        indices = np.random.choice(len(self), size=batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        return len(self.buffer)
