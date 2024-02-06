import hydra
import numpy as np
from omegaconf import DictConfig


def sin_wave(step, frequency):
    return np.sin(2 * np.pi * frequency * step)


def square_wave(step, frequency):
    return np.sign(sin_wave(step, frequency)) * 1


def growing_wave(wave_func, step, frequency, max_steps):
    return wave_func(step, frequency) * (step / max_steps)


@hydra.main(version_base="1.3", config_path="configs", config_name="collect_data.yaml")
def main(cfg: DictConfig):
    env = hydra.utils.instantiate(cfg.env)
    for wrapper in cfg.wrappers:
        env = hydra.utils.instantiate(wrapper, env)

    for wave_func in [sin_wave, square_wave]:
        for freq in cfg.freqs:
            env.reset()
            step = 0
            while step < cfg.nb_steps_per_episode:
                step += 1

                action = np.array([growing_wave(wave_func, step, freq, cfg.nb_steps_per_episode)])
                _, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break

    env.close()


if __name__ == "__main__":
    main()
