# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        foods = newFood.asList()

        "*** YOUR CODE HERE ***"
        value = successorGameState.getScore()

        active_ghost_counter = 0
        scared_ghost_counter = 0
        for ghost_state in newGhostStates:
            if not ghost_state.scaredTimer:
                if ghost_state.getPosition() == newPos:
                    value -= 2000

                manhattan_distance = manhattanDistance(newPos, ghost_state.getPosition())
                if manhattan_distance <= 1:
                    active_ghost_counter += 1
            else:
                if ghost_state.getPosition() == newPos:
                    value += 2000
                manhattan_distance = manhattanDistance(newPos, ghost_state.getPosition())
                if manhattan_distance <= 1:
                    scared_ghost_counter += 1

        food_distance = 0
        for food in foods:
            if food == newPos:
                value += 100
            else:
                manhattan_distance = manhattanDistance(newPos, food)
                food_distance += manhattan_distance

        value -= active_ghost_counter * 200
        value += scared_ghost_counter * 100
        value -= food_distance * 0.1

        return value


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def minimax(self, game_state, agent_index=0, search_depth=0.0):
        state_value = None
        state_action = None

        num_agents = game_state.getNumAgents()
        if agent_index >= num_agents:
            agent_index = 0

        if game_state.isWin() or game_state.isLose() or search_depth / num_agents == self.depth:
            return self.evaluationFunction(game_state), None

        if not agent_index:
            for action in game_state.getLegalActions(agent_index):
                successor = game_state.generateSuccessor(agent_index, action)
                successor_value, _ = self.minimax(successor, agent_index + 1, search_depth + 1)

                if state_value is None:
                    state_value = successor_value
                else:
                    state_value = max(state_value, successor_value)

                if state_value == successor_value:
                    state_action = action

            return state_value, state_action

        else:
            for action in game_state.getLegalActions(agent_index):
                successor = game_state.generateSuccessor(agent_index, action)
                successor_value, _ = self.minimax(successor, agent_index + 1, search_depth + 1)

                if state_value is None:
                    state_value = successor_value
                else:
                    state_value = min(state_value, successor_value)

                if state_value == successor_value:
                    state_action = action

            return state_value, state_action

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, action = self.minimax(game_state=gameState)

        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def alpha_beta_minimax(self, game_state, alpha, beta, agent_index=0, search_depth=0):
        state_value = None
        state_action = None

        num_agents = game_state.getNumAgents()
        if agent_index >= num_agents:
            agent_index = 0

        if game_state.isWin() or game_state.isLose() or search_depth / num_agents == self.depth:
            return self.evaluationFunction(game_state), None

        if not agent_index:
            for action in game_state.getLegalActions(agent_index):
                successor = game_state.generateSuccessor(agent_index, action)
                successor_value, _ = self.alpha_beta_minimax(successor, alpha, beta, agent_index + 1, search_depth + 1)

                if state_value is None:
                    state_value = successor_value
                else:
                    state_value = max(state_value, successor_value)

                if state_value == successor_value:
                    state_action = action

                if state_value > beta:
                    return state_value, state_action

                alpha = max(alpha, state_value)

            return state_value, state_action

        else:
            for action in game_state.getLegalActions(agent_index):
                successor = game_state.generateSuccessor(agent_index, action)
                successor_value, _ = self.alpha_beta_minimax(successor, alpha, beta, agent_index + 1, search_depth + 1)

                if state_value is None:
                    state_value = successor_value
                else:
                    state_value = min(state_value, successor_value)

                if state_value == successor_value:
                    state_action = action

                if state_value < alpha:
                    return state_value, state_action

                beta = min(beta, state_value)

            return state_value, state_action

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, action = self.alpha_beta_minimax(gameState, -999999999, 999999999)

        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def expectimax(self, game_state, agent_index=0, search_depth=0.0):
        state_value = None
        state_action = None

        num_agents = game_state.getNumAgents()
        if agent_index >= num_agents:
            agent_index = 0

        if game_state.isWin() or game_state.isLose() or search_depth / num_agents == self.depth:
            return self.evaluationFunction(game_state), None

        if not agent_index:
            for action in game_state.getLegalActions(agent_index):
                successor = game_state.generateSuccessor(agent_index, action)
                successor_value, _ = self.expectimax(successor, agent_index + 1, search_depth + 1)

                if state_value is None:
                    state_value = successor_value
                else:
                    state_value = max(state_value, successor_value)

                if state_value == successor_value:
                    state_action = action

            return state_value, state_action

        else:
            state_value = 0
            actions = game_state.getLegalActions(agent_index)
            successor_probability = 1 / len(actions)

            for action in actions:
                successor = game_state.generateSuccessor(agent_index, action)
                successor_value, _ = self.expectimax(successor, agent_index + 1, search_depth + 1)

                state_value += successor_probability * successor_value

            return state_value, state_action

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        _, action = self.expectimax(gameState)

        return action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    capsules = currentGameState.getCapsules()
    foods = currentGameState.getFood().asList()
    ghost_states = currentGameState.getGhostStates()
    pacman_position = currentGameState.getPacmanPosition()
    value = currentGameState.getScore()

    food_distance = 0
    for food in foods:
        if food == pacman_position:
            value += 200
        else:
            manhattan_distance = manhattanDistance(pacman_position, food)
            food_distance += manhattan_distance

    capsule_distance = 0
    for capsule in capsules:
        if pacman_position == capsule:
            value += 400
        else:
            manhattan_distance = manhattanDistance(pacman_position, capsule)
            capsule_distance += manhattan_distance

    scared_ghost_counter = 0
    active_ghost_counter = 0
    for ghost_state in ghost_states:
        ghost_position = ghost_state.getPosition()

        if ghost_state.scaredTimer:
            if ghost_position == pacman_position:
                value += 2000

            manhattan_distance = manhattanDistance(pacman_position, ghost_position)
            if manhattan_distance <= 1:
                scared_ghost_counter += 1

        else:
            if ghost_position == pacman_position:
                value -= 2000

            manhattan_distance = manhattanDistance(pacman_position, ghost_position)
            if manhattan_distance <= 1:
                active_ghost_counter += 1

    value -= food_distance * 0.1
    value -= capsule_distance * 40
    value -= active_ghost_counter * 200
    value += scared_ghost_counter * 100

    return value


# Abbreviation
better = betterEvaluationFunction
