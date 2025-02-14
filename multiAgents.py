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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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

        "*** YOUR CODE HERE ***"

        foodDistance = float('inf')
        for food in newFood.asList():
            foodDistance = min(foodDistance, manhattanDistance(food, newPos))

        ghostDistance = float('inf')
        for ghostState in newGhostStates:
            if ghostState.scaredTimer == 0:
                ghostDistance = min(ghostDistance, manhattanDistance(ghostState.getPosition(), newPos))

        ghostPenalty = 0 if ghostDistance > 1 else -200

        score = successorGameState.getScore()
        score += 1.0 / (1.0 + foodDistance)
        score += ghostPenalty

        return score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        def minimax(state, depth, agentIndex):
            #verifica daca jocul s-a terminat sau daca s-a ajuns la adancimea maxima
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), None

            #calculeaza pentru Pacman (agentIndex 0 este întotdeauna Pacman)
            if agentIndex == 0:
                return max_value(state, depth, agentIndex)
            #calculeaza pentru fantome
            else:
                return min_value(state, depth, agentIndex)

        def max_value(state, depth, agentIndex):
            best_score = float("-inf")
            best_action = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = minimax(successor, depth, 1)
                if score > best_score:
                    best_score, best_action = score, action
            return best_score, best_action

        def min_value(state, depth, agentIndex):
            best_score = float("inf")
            best_action = None
            next_agent = agentIndex + 1
            #verifica daca urmatorul agent este din nou Pacman si creste adancimea
            if next_agent == state.getNumAgents():
                next_agent = 0
                depth += 1
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = minimax(successor, depth, next_agent)
                if score < best_score:
                    best_score, best_action = score, action
            return best_score, best_action

        #initiaza minimax-ul cu agentIndex 0 (Pacman) și adancimea 0
        _, action = minimax(gameState, 0, 0)
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def alpha_beta(state, depth, alpha, beta, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth * state.getNumAgents():
                return self.evaluationFunction(state), None

            if agentIndex == 0:  #randul lui Pacman
                return max_value(state, depth, alpha, beta, agentIndex)
            else:  #randul fantomei
                return min_value(state, depth, alpha, beta, agentIndex)

        def max_value(state, depth, alpha, beta, agentIndex):
            v = float("-inf")
            best_action = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = alpha_beta(successor, depth + 1, alpha, beta, (depth + 1) % state.getNumAgents())
                if score > v:
                    v, best_action = score, action
                if v > beta:
                    return v, best_action
                alpha = max(alpha, v)
            return v, best_action

        def min_value(state, depth, alpha, beta, agentIndex):
            v = float("inf")
            best_action = None
            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                score, _ = alpha_beta(successor, depth + 1, alpha, beta, (depth + 1) % state.getNumAgents())
                if score < v:
                    v, best_action = score, action
                if v < alpha:
                    return v, best_action
                beta = min(beta, v)
            return v, best_action

        alpha = float("-inf")
        beta = float("inf")
        _, action = alpha_beta(gameState, 0, alpha, beta, 0)
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        if self.index == 0:
            return self.expectAction(gameState, self.index, self.depth + 1)[1]

        #fantoma:
        else:
            return random.choice(gameState.getLegalActions(self.index))

    def expectAction(self, gameState, agentIndex, depth):
        if (agentIndex == 0):
            nextDepth = depth - 1
        else:
            nextDepth = depth
        if (gameState.isWin() or gameState.isLose() or nextDepth == 0):
            return self.evaluationFunction(gameState), None

        legalMoves = gameState.getLegalActions(agentIndex)
        nextAgent = (agentIndex + 1) % gameState.getNumAgents()
        expectedValue = 0.0
        heroValue = -float("inf")
        heroAction = None

        if agentIndex == 0:
            for line in legalMoves:
                expectedValue = self.expectAction(gameState.generateSuccessor(agentIndex, line), nextAgent, nextDepth)[
                    0]
                if expectedValue > heroValue:
                    heroValue = expectedValue
                    heroAction = line
            return heroValue, heroAction

        else:
            for line in legalMoves:
                expectedValue += self.expectAction(gameState.generateSuccessor(agentIndex, line), nextAgent, nextDepth)[
                    0]
            return expectedValue / len(legalMoves), None

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    nearestFood = float('inf')
    nearestGhost = float('inf')
    foodList = newFood.asList()

    if not foodList:
        nearestFood = 0
    for food in foodList:
        nearestFood = min(nearestFood, manhattanDistance(food, newPos))

    for ghost in newGhostStates:
        ghostX, ghostY = ghost.getPosition()
        if ghost.scaredTimer == 0:
            nearestGhost = min(nearestGhost, manhattanDistance((ghostX, ghostY), newPos))

    hero = (currentGameState.getScore() - 5 / (nearestGhost + 1)) - (nearestFood / 2)
    return hero

# Abbreviation
better = betterEvaluationFunction
