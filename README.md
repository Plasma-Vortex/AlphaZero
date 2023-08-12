# AlphaZero

This code implements the AlphaZero algorithm for Connect Four and Othello.

## To play against the AI

1. Clone the repo: `git clone https://github.com/howard36/AlphaZero.git`
2. Install dependencies.
    1. If using Pipenv: `pipenv install`, then `pipenv shell`
    2. Alternatively, install the packages listed in `Pipfile` with your preferred method.
3. Run `python Main.py`

## How good is it?

The AI can beat a perfect Connect Four player if it goes first (Connect Four is a win for the first player under optimal play). At Othello, it beats me every single game (haven't benchmarked against other Othello programs or players).
