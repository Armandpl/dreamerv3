import time

import numpy as np
from furuta.rl.envs.furuta_sim import FurutaSim
from mcap_protobuf.reader import read_protobuf_messages

if __name__ == "__main__":
    mcap_path = "../data/sim_mcap_dataset/ep241_20240204-160511.mcap"

    # TODO get dt from the mcap file
    dt = 0.02

    env = FurutaSim(render_mode="human")
    env.reset()

    for msg in read_protobuf_messages(mcap_path, log_time_order=True):
        p = msg.proto_msg
        state = np.array(
            [p.motor_angle, p.pendulum_angle, p.motor_angle_velocity, p.pendulum_angle_velocity]
        )
        env._state = state
        env.render()
        time.sleep(dt)

    env.close()
