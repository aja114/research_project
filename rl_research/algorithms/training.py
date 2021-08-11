def train(env, agent_class, num_iter=100, logs=False):

    agent = agent_class(env)

    if logs:
        agent.train(num_iter)
    else:
        agent.train_without_logs(num_iter)

    return agent
