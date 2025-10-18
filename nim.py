import random
import copy

class Nim():
    """
    Nim game

    Piles is a list of pile sizes, current_player is 0 or 1, winner is None or 0/1.
    Removing the last object causes the player who took it to LOSE (per project spec).
    """

    def __init__(self, piles):
        self.piles = piles[:]               # list of ints
        self.player = 0                     # 0 or 1
        self.winner = None                  # None, 0, or 1

    @staticmethod
    def available_actions(piles):
        """
        Given a state (list of pile sizes), return a set of all legal (i, j)
        actions: remove j objects from pile i (1 <= j <= piles[i]).
        """
        actions = set()
        for i, count in enumerate(piles):
            for j in range(1, count + 1):
                actions.add((i, j))
        return actions

    @staticmethod
    def other_player(player):
        return 1 - player

    def switch_player(self):
        self.player = Nim.other_player(self.player)

    def move(self, action):
        """
        Applies action (i, j) to self.piles, then switches player.
        Sets self.winner if game is over.

        Return: None
        """
        i, j = action
        if self.winner is not None:
            raise Exception("Game already over")

        if j < 1 or i < 0 or i >= len(self.piles) or j > self.piles[i]:
            raise Exception("Invalid move")

        # perform move
        self.piles[i] -= j

        # If after move all piles are 0 => the player who just moved LOSES.
        if all(p == 0 for p in self.piles):
            self.winner = self.player  # the player who just moved
        else:
            # game continues
            self.switch_player()

class NimAI():
    """
    Q-learning AI for Nim.
    self.q maps ((state_tuple), action) -> float
    """

    def __init__(self, alpha=0.5, epsilon=0.1):
        # Q-values
        self.q = dict()
        # learning rate
        self.alpha = alpha
        # epsilon for epsilon-greedy
        self.epsilon = epsilon

    def get_q_value(self, state, action):
        """
        Return Q(state, action) if present, otherwise 0.
        state: list of ints
        action: tuple (i, j)
        """
        key = (tuple(state), action)
        return self.q.get(key, 0)

    def update_q_value(self, state, action, old_q, reward, future_rewards):
        """
        Update Q(state, action) according to:
          Q <- old_q + alpha * ( (reward + future_rewards) - old_q )
        """
        new_estimate = reward + future_rewards
        updated = old_q + self.alpha * (new_estimate - old_q)
        key = (tuple(state), action)
        self.q[key] = updated

    def best_future_reward(self, state):
        """
        Return max_a Q(state, a) over all available actions.
        If no available actions, return 0.
        Treat unknown (state,action) as 0.
        """
        actions = Nim.available_actions(state)
        if not actions:
            return 0

        best = None
        for action in actions:
            q_val = self.get_q_value(state, action)
            if best is None or q_val > best:
                best = q_val

        return best if best is not None else 0

    def choose_action(self, state, epsilon=True):
        """
        Choose an action for a given state.
        If epsilon=True: epsilon-greedy (with probability self.epsilon choose random available action).
        If epsilon=False: greedy (choose action with highest Q-value).
        If multiple best actions, break ties randomly.
        If no actions available, return None.
        """
        actions = list(Nim.available_actions(state))
        if not actions:
            return None

        # epsilon-greedy: randomly choose with probability epsilon
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)

        # Otherwise choose best action(s)
        best_actions = []
        best_q = None
        for action in actions:
            q_val = self.get_q_value(state, action)
            if (best_q is None) or (q_val > best_q):
                best_q = q_val
                best_actions = [action]
            elif q_val == best_q:
                best_actions.append(action)

        return random.choice(best_actions)

    def update(self, old_state, action, new_state, reward):
        """
        Provided in spec: helper to perform Q-update when the AI performs `action`
        in `old_state`, gets `reward`, and becomes `new_state`.
        """
        old_q = self.get_q_value(old_state, action)
        future_rewards = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, future_rewards)

#
# Optional helpers: train & play (adapted from CS50 distribution)
#
def train(n):
    """
    Train a NimAI by having it play n games against itself.
    Returns trained NimAI.
    """
    ai = NimAI()

    for i in range(n):
        game = Nim([1, 3, 5, 7])
        # Keep history of moves for each player so we can assign rewards at game end
        last = {0: None, 1: None}

        while game.winner is None:
            state = game.piles[:]
            action = ai.choose_action(state, epsilon=True)
            # store move for current player to update later
            last[game.player] = (state, action)

            # perform action
            game.move(action)

            # If the move ended the game, assign rewards:
            if game.winner is not None:
                # The player who made the last move loses (spec), so:
                loser = game.winner
                winner = Nim.other_player(loser)

                # Update the loser: reward -1
                s, a = last[loser]
                ai.update(s, a, game.piles[:], -1)

                # Update the winner: reward +1
                s, a = last[winner]
                ai.update(s, a, game.piles[:], 1)
            else:
                # intermediate move: reward 0 for previous player's move
                # update the player who just moved with immediate reward 0
                # (game.player just switched in move())
                prev_player = Nim.other_player(game.player)
                s, a = last[prev_player]
                ai.update(s, a, game.piles[:], 0)

    return ai

def play(ai):
    """
    Let a human play against the trained AI.
    (Simple textual UI)
    """
    game = Nim([1, 3, 5, 7])

    while game.winner is None:
        # show piles
        print("Piles:")
        for i, p in enumerate(game.piles):
            print(f"  Pile {i}: {p}")
        print()

        if game.player == 0:
            # Human's turn
            print("Your turn.")
            while True:
                pile = int(input("Choose a pile: "))
                count = int(input("How many to remove: "))
                try:
                    game.move((pile, count))
                    break
                except Exception as e:
                    print("Invalid move, try again.")
        else:
            # AI's turn
            print("AI's turn.")
            action = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {action[1]} from pile {action[0]}.")
            game.move(action)

    # Game over
    print()
    print("Game over.")
    print(f"Winner: {game.winner} (player who took the last object loses)")

if __name__ == "__main__":
    # quick demo: train a small number of games, then play
    print("Training 1000 games...")
    trained = train(1000)
    print("Done training.")
    play(trained)
