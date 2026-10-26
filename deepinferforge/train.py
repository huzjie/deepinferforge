# -*- coding: utf-8 -*-
"""Mock training: raise engine ``skill`` monotonically toward 1.0.

The signal is deliberately the "monotone-to-1" form (``skill += lr * (1 - skill)``)
rather than a reward-flip ``+1/-1`` update, which would drift downward when the
policy starts below random. This is the discipline the reference keeps so every
benchmark sweeps cleanly from a random baseline to the optimum.
"""


def train_engine(engine, steps=None, lr=None):
    cfg = engine.config.train
    steps = steps if steps is not None else cfg.max_steps
    lr = lr if lr is not None else cfg.learning_rate
    history = [engine.skill]
    for _ in range(steps):
        engine.skill += lr * (1.0 - engine.skill)
        history.append(engine.skill)
    return history
