import random
import math

from environment.util import manhattanDistance
from environment.game import Directions, Agent
import environment.util as util

def scoreEvaluationFunction(currentGameState):
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    foodPos = newFood.asList()
    foodCount = len(foodPos)

    score = currentGameState.getScore()

    closestDistance = 1e6
    for i in range(foodCount):
        distance = manhattanDistance(newPos, foodPos[i]) + foodCount * 100
        if distance < closestDistance:
            closestDistance = distance
            closestFood = foodPos

    if foodCount == 0:
        closestDistance = 0

    score -= closestDistance

    for i in range(len(newGhostStates)):
        ghostPos = currentGameState.getGhostPosition(i+1)
        if manhattanDistance(newPos, ghostPos) <= 1:
            score -= 1e6

    return score

class MCTSInterface(Agent, object):
    class Node(object):
        _visited_states = set()

        def __init__(self, game_state, parent=None, action=None):
            if parent is None:
                self._visited_states = set()
            self._visited_states.add(game_state.getPacmanPosition())

            self._game_state = game_state
            self._parent = parent
            # Store an action to get to this node from the parent
            self._action = action

            self._n_backpropagations = 0
            self._total_reward = 0


            self._children = []
            self._possible_actions = []
            if self._game_state.isLose():
                self._total_reward = -200
                self._n_backpropagations = 1
            else:
                for u in self._game_state.getLegalActions():
                    if u != 'Stop':
                        self._possible_actions.append(u)

        # Derived class should implement this method
        def select(self):
            util.raiseNotDefined()

        # Derived class should implement this method
        def expand(self):
            util.raiseNotDefined()

        def backpropagate(self, reward, depth=1):
            self._n_backpropagations += 1 / math.sqrt(depth)
            self._total_reward += reward * 1 / math.sqrt(depth)
            if self._parent is not None:
                self._parent.backpropagate(reward, depth=depth + 1)

        def get_game_state(self):
            return self._game_state.deepCopy()

        # Useful helper
        @property
        def _average_reward(self):
            if self._n_backpropagations == 0:
                raise Exception('Trying to calculate average reward without any backpropagations.')
            return self._total_reward / self._n_backpropagations

        @property
        def _best_child_UCB(self):
            best_childs = self._children[0]
            best_score = None
            for i in self._children:
                result = (i._total_reward/i._n_backpropagations) + 1*math.sqrt((2*math.log(self._n_backpropagations))/i._n_backpropagations)
                if result == best_score:
                    bestChild.append(i)
                if best_score is None or result > best_score:
                    best_score = result
                    bestChild = []
                    bestChild.append(i)
            return bestChild[random.randint(0, len(bestChild) - 1)]

        @property
        def _best_child_UCT(self):
            C = math.sqrt(2)
            best_childs = self._children[0]
            best_score = None
            for i in self._children:
                result = (i._total_reward/i._n_backpropagations) + 2*C*math.sqrt((2*math.log(self._n_backpropagations))/i._n_backpropagations)
                if result == best_score:
                    bestChild.append(i)
                if best_score is None or result > best_score:
                    best_score = result
                    bestChild = []
                    bestChild.append(i)
            return bestChild[random.randint(0, len(bestChild) - 1)]

        @property
        def _best_child(self):
            best_childs = []
            best_score = None
            for curr_child in self._children:
                if best_score is None or curr_child._average_reward > best_score:
                    best_score = curr_child._average_reward
                    best_childs = [curr_child]
                elif curr_child._average_reward == best_score:
                    best_childs.append(curr_child)
            return random.choice(best_childs)

        # Useful helper
        def choose_best_action(self):
            return self._best_child._action

        def choose_best_action_max(self):
            L = [n._total_reward for n in self._children]
            return self._children[L.index(max(L))]._action

    def __init__(self):
        self.first_time = True
        self._max_mcts_iterations = 200
        self._max_simulation_iterations = 1

    def getAction(self, gameState):
        if self.first_time:
            import time
            time.sleep(5)
            self.first_time = False
        root = self.create_tree(gameState)
        for _ in range(self._max_mcts_iterations):
            selected_node = root.select()
            expanded_node = selected_node.expand()
            reward = self.simulate(expanded_node.get_game_state())
            expanded_node.backpropagate(reward)
        action = root.choose_best_action()
        return action

    # Implement this in derived class
    def create_tree(self, gameState):
        util.raiseNotDefined()

    # Reimplement this in derived class if needed
    def simulate(self, gameState):
        # Do not simulate anything, just give the current score
        return gameState.getScore()


class SimpleMCTS(MCTSInterface):
    """
      A MCTS agent chooses an action using Monte-Carlo Tree Search.
    """
    class Node(MCTSInterface.Node):
        def __init__(self, game_state, parent=None, action=None):
            super(self.__class__, self).__init__(game_state, parent, action)
            self._max_mcts_iterations = 300
            self._max_simulation_iterations = 1


        def select(self):
            # if len(self._possible_actions) != 0 or len(self._children) == 0:
            if len(self._children) == 0:
                return self
            if len(self._possible_actions) == 0:
                return self._best_child.select()
            if random.random() < 0.5:
                return self
            else:
                return self._best_child.select()


        def expand(self):
            if len(self._children) != 0 and len(self._possible_actions) == 0:
                raise Exception("Couldn't select a node without possible children, bit with existing.")
            while len(self._possible_actions) != 0:
                action = self._possible_actions[0]
                self._possible_actions.pop(0)
                next_game_state = self._game_state.generatePacmanSuccessor(action)
                # if next_game_state.getPacmanPosition() in self._visited_states:
                #     continue
                expanded_child = self.__class__(
                    game_state=self._game_state.generatePacmanSuccessor(action),
                    parent=self,
                    action=action,
                )
                self._children.append(expanded_child)
                return expanded_child
            return self

    def __init__(self):
        super(self.__class__, self).__init__()

    def create_tree(self, game_state):
        return self.Node(game_state=game_state)

    def simulate(self, game_state):
        def dist_l1(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def dist_l2(pos1, pos2):
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        def find_nearest_food(game_state):
            curr_pos = game_state.getPacmanPosition()
            food_grid = game_state.getFood()
            nearest = None
            for x in range(food_grid.width):
                for y in range(food_grid.height):
                    if not food_grid[x][y]:
                        continue
                    if nearest is None or \
                            dist_l1(curr_pos, (x, y)) < dist_l1(curr_pos, nearest):
                        nearest = (x, y)
            return nearest

        nearest_food = None
        n_iter = 0
        while not game_state.isWin() and not game_state.isLose() and \
              n_iter < self._max_simulation_iterations:
            if nearest_food is None:
                nearest_food = find_nearest_food(game_state)
            if nearest_food is None:
                game_state = game_state.generatePacmanSuccessor(
                        random.choice(game_state.getLegalActions())
                )
            else:
                best_dist = None
                best_next_state = None
                for action in game_state.getLegalActions():
                    next_state = game_state.generatePacmanSuccessor(action)
                    curr_dist = dist_l1(next_state.getPacmanPosition(), nearest_food)
                    if best_dist is None or curr_dist < best_dist:
                        best_dist = curr_dist
                        best_next_state = next_state
                game_state = best_next_state
                if game_state.getPacmanPosition() == nearest_food:
                    nearest_food = None
            n_iter += 1
        return scoreEvaluationFunction(game_state)


class MCTS_FPU(MCTSInterface):
    """
      A MCTS agent chooses an action using Monte-Carlo Tree Search.
    """
    class Node(MCTSInterface.Node):
        def __init__(self, game_state, parent=None, action=None):
            super(self.__class__, self).__init__(game_state, parent, action)

        def select(self):
            if len(self._possible_actions) != 0 or len(self._children) == 0:
                return self
            return self._best_child.select()

        def expand(self):
            if len(self._children) != 0 and len(self._possible_actions) == 0:
                raise Exception("Couldn't select a node without possible children, bit with existing.")
            while len(self._possible_actions) != 0:
                action = self._possible_actions[0]
                self._possible_actions.pop(0)
                next_game_state = self._game_state.generatePacmanSuccessor(action)
                # if next_game_state.getPacmanPosition() in self._visited_states:
                #     continue
                expanded_child = self.__class__(
                    game_state=self._game_state.generatePacmanSuccessor(action),
                    parent=self,
                    action=action,
                )
                self._children.append(expanded_child)
                return expanded_child
            return self

    def __init__(self):
        super(self.__class__, self).__init__()

    def create_tree(self, game_state):
        return self.Node(game_state=game_state)

    def simulate(self, game_state):
        def dist_l1(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def dist_l2(pos1, pos2):
            return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

        def find_nearest_food(game_state):
            curr_pos = game_state.getPacmanPosition()
            food_grid = game_state.getFood()
            nearest = None
            for x in range(food_grid.width):
                for y in range(food_grid.height):
                    if not food_grid[x][y]:
                        continue
                    if nearest is None or \
                            dist_l1(curr_pos, (x, y)) < dist_l1(curr_pos, nearest):
                        nearest = (x, y)
            return nearest

        nearest_food = None
        n_iter = 0
        while not game_state.isWin() and not game_state.isLose() and \
              n_iter < self._max_simulation_iterations:
            if nearest_food is None:
                nearest_food = find_nearest_food(game_state)
            if nearest_food is None:
                game_state = game_state.generatePacmanSuccessor(
                        random.choice(game_state.getLegalActions())
                )
            else:
                best_dist = None
                best_next_state = None
                for action in game_state.getLegalActions():
                    next_state = game_state.generatePacmanSuccessor(action)
                    curr_dist = dist_l1(next_state.getPacmanPosition(), nearest_food)
                    if best_dist is None or curr_dist < best_dist:
                        best_dist = curr_dist
                        best_next_state = next_state
                game_state = best_next_state
                if game_state.getPacmanPosition() == nearest_food:
                    nearest_food = None
            n_iter += 1
        return scoreEvaluationFunction(game_state)


class MCTS_UCB(MCTSInterface):
    """
      A MCTS agent chooses an action using Monte-Carlo Tree Search.
    """
    class Node(MCTSInterface.Node):
        def __init__(self, game_state, parent=None, action=None):
            super(self.__class__, self).__init__(game_state, parent, action)

        def select(self):
            if len(self._possible_actions) != 0 or len(self._children) == 0:
                return self
            return self._best_child_UCB.select()

        def expand(self):
            if len(self._children) != 0 and len(self._possible_actions) == 0:
                raise Exception("Couldn't select a node without possible children, bit with existing.")
            while len(self._possible_actions) != 0:
                action = self._possible_actions[0]
                self._possible_actions.pop(0)
                next_game_state = self._game_state.generatePacmanSuccessor(action)
                if next_game_state.getPacmanPosition() in self._visited_states:
                    continue
                expanded_child = self.__class__(
                    game_state=self._game_state.generatePacmanSuccessor(action),
                    parent=self,
                    action=action,
                )
                self._children.append(expanded_child)
                return expanded_child
            return self

    def __init__(self):
        super(self.__class__, self).__init__()

    def create_tree(self, game_state):
        return self.Node(game_state=game_state)

    def simulate(self, game_state):
        def dist_l1(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def find_nearest_food(game_state):
            curr_pos = game_state.getPacmanPosition()
            food_grid = game_state.getFood()
            nearest = None
            for x in range(food_grid.width):
                for y in range(food_grid.height):
                    if not food_grid[x][y]:
                        continue
                    if nearest is None or \
                            dist_l1(curr_pos, (x, y)) < dist_l1(curr_pos, nearest):
                        nearest = (x, y)
            return nearest

        nearest_food = None
        n_iter = 0
        while not game_state.isWin() and not game_state.isLose() and \
              n_iter < self._max_simulation_iterations:
            if nearest_food is None:
                nearest_food = find_nearest_food(game_state)
            if nearest_food is None:
                game_state = game_state.generatePacmanSuccessor(
                        random.choice(game_state.getLegalActions())
                )
            else:
                best_dist = None
                best_next_state = None
                for action in game_state.getLegalActions():
                    next_state = game_state.generatePacmanSuccessor(action)
                    curr_dist = dist_l1(next_state.getPacmanPosition(), nearest_food)
                    if best_dist is None or curr_dist < best_dist:
                        best_dist = curr_dist
                        best_next_state = next_state
                game_state = best_next_state
                if game_state.getPacmanPosition() == nearest_food:
                    nearest_food = None
            n_iter += 1
        return scoreEvaluationFunction(game_state)


class MCTS_UCT(MCTSInterface):
    """
      A MCTS agent chooses an action using Monte-Carlo Tree Search.
    """
    class Node(MCTSInterface.Node):
        def __init__(self, game_state, parent=None, action=None):
            super(self.__class__, self).__init__(game_state, parent, action)

        def select(self):
            if len(self._possible_actions) != 0 or len(self._children) == 0:
                return self
            return self._best_child_UCT.select()

        def expand(self):
            if len(self._children) != 0 and len(self._possible_actions) == 0:
                raise Exception("Couldn't select a node without possible children, bit with existing.")
            while len(self._possible_actions) != 0:
                action = self._possible_actions[0]
                self._possible_actions.pop(0)
                next_game_state = self._game_state.generatePacmanSuccessor(action)
                if next_game_state.getPacmanPosition() in self._visited_states:
                    continue
                expanded_child = self.__class__(
                    game_state=self._game_state.generatePacmanSuccessor(action),
                    parent=self,
                    action=action,
                )
                self._children.append(expanded_child)
                return expanded_child
            return self

    def __init__(self):
        super(self.__class__, self).__init__()

    def create_tree(self, game_state):
        return self.Node(game_state=game_state)

    def simulate(self, game_state):
        def dist_l1(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def find_nearest_food(game_state):
            curr_pos = game_state.getPacmanPosition()
            food_grid = game_state.getFood()
            nearest = None
            for x in range(food_grid.width):
                for y in range(food_grid.height):
                    if not food_grid[x][y]:
                        continue
                    if nearest is None or \
                            dist_l1(curr_pos, (x, y)) < dist_l1(curr_pos, nearest):
                        nearest = (x, y)
            return nearest

        nearest_food = None
        n_iter = 0
        while not game_state.isWin() and not game_state.isLose() and \
              n_iter < self._max_simulation_iterations:
            if nearest_food is None:
                nearest_food = find_nearest_food(game_state)
            if nearest_food is None:
                game_state = game_state.generatePacmanSuccessor(
                        random.choice(game_state.getLegalActions())
                )
            else:
                best_dist = None
                best_next_state = None
                for action in game_state.getLegalActions():
                    next_state = game_state.generatePacmanSuccessor(action)
                    curr_dist = dist_l1(next_state.getPacmanPosition(), nearest_food)
                    if best_dist is None or curr_dist < best_dist:
                        best_dist = curr_dist
                        best_next_state = next_state
                game_state = best_next_state
                if game_state.getPacmanPosition() == nearest_food:
                    nearest_food = None
            n_iter += 1
        return scoreEvaluationFunction(game_state)

class MCTSFull(MCTSInterface):
    """
      A MCTS agent chooses an action using Monte-Carlo Tree Search.
    """

    @property
    def _best_child(self):
        best_childs = []
        lowest_depth = None
        for curr_child in self._children:
            if lowest_depth is None or curr_child.depth < lowest_depth:
                lowest_depth = curr_child.depth
                best_childs = [curr_child]
            elif curr_child.depth == lowest_depth:
                best_childs.append(curr_child)
        return random.choice(best_childs)

    class Node(MCTSInterface.Node):
        def __init__(self, game_state, parent=None, action=None):
            super(self.__class__, self).__init__(game_state, parent, action)
            self.depth = 1

        def select(self):
            if len(self._possible_actions) != 0 or len(self._children) == 0:
                return self
            return self._best_child.select()

        def expand(self):
            if len(self._children) != 0 and len(self._possible_actions) == 0:
                raise Exception("Couldn't select a node without possible children, bit with existing.")
            while len(self._possible_actions) != 0:
                action = self._possible_actions[0]
                self._possible_actions.pop(0)
                next_game_state = self._game_state.generatePacmanSuccessor(action)
                # if next_game_state.getPacmanPosition() in self._visited_states:
                #     continue
                expanded_child = self.__class__(
                    game_state=self._game_state.generatePacmanSuccessor(action),
                    parent=self,
                    action=action,
                )
                self._children.append(expanded_child)
                return expanded_child
            return self

        def backpropagate(self, reward, depth=1):
            if depth > self.depth:
                self.depth = depth
            self._n_backpropagations += 1 / math.sqrt(depth)
            self._total_reward += reward * 1 / math.sqrt(depth)
            if self._parent is not None:
                self._parent.backpropagate(reward, depth=depth + 1)


    def __init__(self):
        super(self.__class__, self).__init__()

    def create_tree(self, game_state):
        return self.Node(game_state=game_state)

    def simulate(self, game_state):
        def dist_l1(pos1, pos2):
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        def find_nearest_food(game_state):
            curr_pos = game_state.getPacmanPosition()
            food_grid = game_state.getFood()
            nearest = None
            for x in range(food_grid.width):
                for y in range(food_grid.height):
                    if not food_grid[x][y]:
                        continue
                    if nearest is None or \
                            dist_l1(curr_pos, (x, y)) < dist_l1(curr_pos, nearest):
                        nearest = (x, y)
            return nearest

        nearest_food = None
        n_iter = 0
        while not game_state.isWin() and not game_state.isLose() and \
              n_iter < self._max_simulation_iterations:
            if nearest_food is None:
                nearest_food = find_nearest_food(game_state)
            if nearest_food is None:
                game_state = game_state.generatePacmanSuccessor(
                        random.choice(game_state.getLegalActions())
                )
            else:
                best_dist = None
                best_next_state = None
                for action in game_state.getLegalActions():
                    next_state = game_state.generatePacmanSuccessor(action)
                    curr_dist = dist_l1(next_state.getPacmanPosition(), nearest_food)
                    if best_dist is None or curr_dist < best_dist:
                        best_dist = curr_dist
                        best_next_state = next_state
                game_state = best_next_state
                if game_state.getPacmanPosition() == nearest_food:
                    nearest_food = None
            n_iter += 1
        return scoreEvaluationFunction(game_state)
