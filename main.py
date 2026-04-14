import random
import time

import gym


def main():

    env = gym.make("Chess-v0")
    board = env.reset()

    done = False

    start_time = time.perf_counter()

    while not done:
        legal_moves = env.legal_moves
        board_legal_moves = board.legal_moves
        action = random.choice(legal_moves)
        print(len(legal_moves), len(set(board_legal_moves)))

        board, reward, done, info = env.step(action)

        # tensor = board_to_tensor(board)
        if reward > 0:
            print(reward)

    end_time = time.perf_counter()
    print(env.render())
    env.close()
    print(f"Czas wykonania: {end_time - start_time:.6f} sekund")


if __name__ == "__main__":
    main()
