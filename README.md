# 🧠 Nim Q-Learning AI

A reinforcement learning project where an AI teaches itself to play the mathematical strategy game **Nim** using **Q-learning**.  
Developed in Python, this project demonstrates how an agent can learn optimal strategies entirely through self-play.

---

## 📖 Overview

In the game of **Nim**, players take turns removing objects from piles.  
On each turn, a player must remove **at least one object** from **exactly one pile**.  
The player forced to take the **last object loses**.

This project trains an AI to learn the best moves through **reinforcement learning**, without any hardcoded strategy.  
Over thousands of self-play games, the AI learns which actions lead to victory and which lead to defeat.

---

## 🧩 Key Concepts

- **Reinforcement Learning (Q-Learning):**
  - Learns an action-value function `Q(s, a)` for every state-action pair.
  - Updates Q-values after each move based on the reward and expected future rewards.

- **Epsilon-Greedy Action Selection:**
  - Balances exploration (trying new moves) with exploitation (choosing the best-known move).

- **Self-Play Training:**
  - The AI trains by playing thousands of games against itself to refine its strategy.

---
## ⚙️ Installation

Clone this repository and ensure you have **Python 3.8+** installed.

```bash
git clone https://github.com/<your-username>/nim-qlearning-ai.git
cd nim-qlearning-ai
(Optional) create a virtual environment:
````
bash
Copy code
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
## ▶️ Usage
1. Train the AI and Play
bash
Copy code
python play.py
You’ll see output similar to:

vbnet
Copy code
Playing training game 1
...
Playing training game 10000
Done training

Piles:
Pile 0: 1
Pile 1: 3
Pile 2: 5
Pile 3: 7

AI's Turn
AI chose to take 1 from pile 2.
Then you can play against the trained AI interactively in the terminal.

## ⚗️ How It Works
Concept	Description
State (s)	A tuple representing the number of objects in each pile, e.g. (1, 3, 5, 7)
Action (a)	A pair (pile, count) meaning “remove count objects from pile pile”
Reward (r)	+1 if AI wins, -1 if AI loses, 0 otherwise
Q-Learning Formula	Q(s, a) ← Q(s, a) + α * [(r + γ * max(Q(s′, a′))) − Q(s, a)]
Epsilon (ε)	Probability of choosing a random action to explore new possibilities

The AI gradually converges toward the optimal strategy by updating Q-values over time.

## 🧮 Example of Learned Strategy
After ~10,000 training games, the AI tends to:

Move into "safe" Nim-sum positions that force the opponent to lose.

Avoid leaving symmetric or losing configurations.

Prefer actions that maximize its long-term winning probability.

## 🧑‍💻 Technologies
Python 3

Standard Library Only

No third-party dependencies (except optional numpy or pandas for analysis)

## 🧰 Customization
You can adjust training parameters in nim.py:

python
Copy code
ai = NimAI(alpha=0.5, epsilon=0.1)
alpha: learning rate (how much new info overrides old knowledge)

epsilon: exploration rate (chance to try random moves)

## 🧪 Future Improvements
Add persistence (save and load trained Q-tables using pickle).

Add a GUI using tkinter or pygame.

Visualize Q-value evolution or training performance.

Extend to multi-agent reinforcement learning environments.

##🏆 Credits
Project inspired by CS50’s Introduction to Artificial Intelligence with Python (Harvard / edX).
Custom implementation and documentation by Lucas Gonzalez and HackingGPT.

## 📄 License
MIT License © 2025
Feel free to use, modify, and distribute with attribution.
