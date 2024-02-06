import sys

import hydra
import numpy as np
import pygame
from furuta.rl.envs.furuta_sim import FurutaSim
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="configs", config_name="play.yaml")
def main(cfg: DictConfig):
    env = hydra.utils.instantiate(cfg.env, render_mode="human")
    if "wrappers" in cfg:
        for wrapper in cfg.wrappers:
            env = hydra.utils.instantiate(wrapper, env=env)
    pygame.init()
    env.reset()
    pressed_keys = set()
    while True:
        action = np.array([0.0])
        quit_ = False

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pressed_keys.add(event.key)
                if event.key == pygame.K_q:
                    quit_ = True
            elif event.type == pygame.KEYUP:
                pressed_keys.discard(event.key)
            if event.type == pygame.QUIT:
                quit_ = True

        if pygame.K_LEFT in pressed_keys:
            action -= cfg.max_dact
        elif pygame.K_RIGHT in pressed_keys:
            action += cfg.max_dact
        else:
            action = action - np.sign(action) * cfg.max_dact
        if quit_:
            break

        action = np.clip(action, -cfg.max_act, cfg.max_act)

        _, _, terminated, truncated, _ = env.step(action)
        env.render()
        if terminated or truncated:
            print("done")
            print(env._state)
            print(env.state_space.contains(env._state))
            print(env.state_space)
            env.reset()

    env.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
