import gym
import gym_chess
import random

def main():

    env = gym.make('Chess-v0')
    env.reset()
    print(env.render())

    
    done = False

    while not done:

        action = random.choice(list(env.legal_moves))
        
        obs, reward, done, info = env.step(action)
        print(env.render(mode='unicode'), end="\n\n")

    env.close()

if __name__ == "__main__":
    main()
