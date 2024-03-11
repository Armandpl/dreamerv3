import torch


def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


def compute_lambda_returns(
    rewards: torch.FloatTensor,
    values: torch.FloatTensor,
    continues: torch.FloatTensor,
    gamma: float,
    return_lambda: float,
) -> torch.FloatTensor:
    """Values is the output from the critic, rewards are the predicted rewards continues are the
    predicted continue flags.

    Shape is (B, T, 1) Output shape is (B, T-1, 1)
    """
    horizon = rewards.shape[1] - 1
    # Eq. 7
    # lambda return shape should be (B, T, 1)
    # pred_rewards is (B, T, 1)
    # pred_values is (B, T, 1)
    # continues is (B, T, 1)
    # where T is IMAGINE_HORIZON + 1
    batch_size = rewards.shape[0]
    lambda_returns = torch.empty(batch_size, horizon, 1, device=rewards.device)

    # we compute the equation by developping it
    # eq. 7 Rt = rt + GAMMA*Ct ( (1 - LAMBDA) * Vt+1 + LAMBDA * Rt+1)
    # we first compute rt + GAMMA*Ct * ((1 - LAMBDA) * Vt+1)
    # then we go from the left to the right of the list/time dimension
    # and we compute Rt = interm[t] + Ct * GAMMA * LAMBDA * Rt+1
    # w/ the last RT+1 being the last nn values estimated
    # rt is the reward at t, Vt the output of the critic at t, Ct the output of the continue network at t
    # TODO should we offset the values to get Vt+1? -> YES
    # seems we have to offset continues too? -> yes ofc else it doesn't match the value
    interm = rewards[:, :-1] + gamma * continues[:, 1:] * values[:, 1:] * (
        1 - return_lambda
    )  # (B, T, 1)

    for t in reversed(range(horizon)):  # TODO can we use [::-1] synthax
        # don't have access to rewards after horizon so we use the estimation
        if t == (horizon - 1):
            Rt_plus_1 = values[:, -1]
        else:
            Rt_plus_1 = lambda_returns[:, t + 1]
        lambda_returns[:, t] = (
            interm[:, t] + continues[:, t + 1] * gamma * return_lambda * Rt_plus_1
        )

    return lambda_returns
