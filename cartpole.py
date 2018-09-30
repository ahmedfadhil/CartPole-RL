import gym
import numpy as np
from flask import Flask
import json

env = gym.make("cartpole-v1")


def play(env, policy):
    observation = env.reset()
    done = False
    observations = []
    score = 0

    for _ in range(5000):
        observations += [observation.tolist()]

        if done:
            break

    #     Pick an action according to the policy matrix

    outcome = np.dot(policy, observation)
    action = 1 if outcome > 0 else 0
    #      Make action, record reward
    observation, reward, done, info = env.step(action)
    score += reward
    return score, observation


policy = np.random.rand(1, 4) - 0.5

score, observations = play(env, policy)
print('Policy Score', score)

app = Flask(__name__, static_folder='.')


@app.route("/data")
def data():
    return json.dump(max[1])


@app.route("/")
def root():
    return app.send_static_file('./index.html')


app.run(host='0.0.0.0', port='3000')

max = (0, [], [])

for _ in range(100):
    policy = np.random.rand(1, 4)
    score, observations = play(env, policy)

    if score > max[0]:
        max = (score, observations, policy)

    print('Max score:', max[0])

scores = []
for _ in range(100):
    score, _ = play(env, max[2])
    scores += [score]

print('Average Score (100 trials)', np.mean(scores))
