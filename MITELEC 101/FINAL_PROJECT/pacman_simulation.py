import random
import time
from collections import defaultdict
import math

# ====================== Environment Setup ======================

class Maze:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.grid = [['.' for _ in range(width)] for _ in range(height)]
        self.shared_corridors = set()
        self.walls = set()
        self.pellets = set()
        self.shared_route_status = {}  # Tracks shared corridor status: UNLOCKED or LOCKED_BY_AGENT
        self._generate_maze()

    def _generate_maze(self):
        # Create simple walls and shared corridors
        for _ in range(int(self.width * self.height * 0.1)):
            wall = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            self.grid[wall[0]][wall[1]] = '#'
            self.walls.add(wall)

        for _ in range(int(self.width * self.height * 0.15)):
            shared = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if shared not in self.walls:
                self.grid[shared[0]][shared[1]] = '='
                self.shared_corridors.add(shared)

        for i in range(self.height):
            for j in range(self.width):
                if self.grid[i][j] == '.':
                    self.pellets.add((i, j))

    def display(self, agents):
        grid_copy = [row[:] for row in self.grid]
        for agent in agents:
            x, y = agent.pos
            grid_copy[x][y] = agent.symbol
        for row in grid_copy:
            print(' '.join(row))
        print()

    def lock(self, position, agent_name):
        """Lock a shared corridor position for exclusive use by an agent"""
        if position in self.shared_corridors:
            self.shared_route_status[position] = f"LOCKED_BY_{agent_name}"
            return True
        return False

    def unlock(self, position):
        """Unlock a shared corridor position"""
        if position in self.shared_route_status:
            del self.shared_route_status[position]
            return True
        return False

    def is_locked(self, position):
        """Check if a position is currently locked"""
        return position in self.shared_route_status

# ====================== Agent Setup ======================

class Agent:
    def __init__(self, name, symbol, pos, energy=100):
        self.name = name
        self.symbol = symbol
        self.pos = pos
        self.energy = energy
        self.score = 0
        self.wait_time = 0
        self.failed_negotiations = 0
        self.active = True
        self.negotiation_history = []  # For learning strategies
        self.learning_rate = 0.1  # Simple learning parameter
        self.sensing_radius = 3  # For conflict pre-detection

    def move(self, direction, maze):
        x, y = self.pos
        dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}.get(direction, (0, 0))
        nx, ny = x + dx, y + dy

        # Boundary and wall check
        if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
            return (nx, ny)
        return self.pos

    def update_energy(self, delta):
        self.energy += delta
        if self.energy < 0:
            self.energy = 0

    def learn_from_negotiation(self, success, opponent, issue):
        # Simple reinforcement learning: adjust strategy based on outcome
        self.negotiation_history.append((success, opponent, issue))
        if success:
            # Increase confidence in similar situations
            pass  # Could be extended with more complex learning

    def make_multi_issue_proposal(self, opponents, maze):
        # Multi-issue negotiation: propose trade-offs
        # For example, offer to yield corridor access in exchange for future pellet sharing
        if not opponents:
            return None
        opponent = random.choice(opponents)
        # Simple proposal: offer to wait if opponent shares pellets later
        proposal = {
            'type': 'trade',
            'offer': 'yield_corridor',
            'request': 'share_pellets',
            'from': self.name,
            'to': opponent.name
        }
        return proposal

    def predict_next_move(self, direction, maze):
        """Predict the next position based on current direction"""
        x, y = self.pos
        dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}.get(direction, (0, 0))
        nx, ny = x + dx, y + dy
        
        # Boundary and wall check
        if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
            return (nx, ny)
        return self.pos

    def detect_potential_conflicts(self, agents, maze):
        """Detect potential conflicts within sensing radius"""
        potential_conflicts = []
        my_next_moves = ['U', 'D', 'L', 'R']
        
        for direction in my_next_moves:
            my_next_pos = self.predict_next_move(direction, maze)
            if my_next_pos == self.pos:  # Skip if we can't move there
                continue
                
            for other_agent in agents:
                if other_agent != self and other_agent.active:
                    # Check if other agent is within sensing radius
                    distance = math.sqrt((self.pos[0] - other_agent.pos[0])**2 +
                                       (self.pos[1] - other_agent.pos[1])**2)
                    if distance <= self.sensing_radius:
                        # Check if other agent might move to same position
                        for other_direction in ['U', 'D', 'L', 'R']:
                            other_next_pos = other_agent.predict_next_move(other_direction, maze)
                            if other_next_pos == my_next_pos and other_next_pos != other_agent.pos:
                                potential_conflicts.append({
                                    'position': my_next_pos,
                                    'with_agent': other_agent,
                                    'my_direction': direction,
                                    'other_direction': other_direction
                                })
        return potential_conflicts

# ====================== Ghost Setup ======================

class Ghost:
    def __init__(self, name, symbol, pos):
        self.name = name
        self.symbol = symbol
        self.pos = pos
        self.active = True

    def move(self, maze, agents):
        # Simple ghost movement: chase nearest agent or random
        x, y = self.pos
        directions = ['U', 'D', 'L', 'R']
        random.shuffle(directions)
        
        # Try to move towards the nearest agent
        best_dir = None
        min_dist = float('inf')
        for dir in directions:
            dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[dir]
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
                # Check distance to nearest active agent
                for agent in agents:
                    if agent.active:
                        dist = math.sqrt((nx - agent.pos[0])**2 + (ny - agent.pos[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_dir = (nx, ny)
        if best_dir:
            return best_dir
        # Fallback to random move
        for dir in directions:
            dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[dir]
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
                return (nx, ny)
        return self.pos

# ====================== Negotiation & Conflict Handling ======================

class ConflictManager:
    def __init__(self):
        self.conflicts = 0
        self.successful_negotiations = 0
        self.waiting_times = defaultdict(int)
        self.multi_issue_negotiations = 0
        self.priority_based_resolutions = 0
        self.alternating_offer_resolutions = 0
        self.conflict_log = []
        self.negotiation_log = []
        self.pre_detected_conflicts = 0

    def negotiate(self, agents, proposed_positions, maze):
        reverse_map = defaultdict(list)
        for agent, pos in proposed_positions.items():
            reverse_map[pos].append(agent)

        resolved_positions = {}
        for pos, contenders in reverse_map.items():
            if len(contenders) == 1:
                resolved_positions[contenders[0]] = pos
            else:
                # Conflict detected
                self.conflicts += 1
                self.log_conflict_event(contenders, pos, 'position_conflict')
                
                # Choose negotiation strategy based on situation
                if len(contenders) == 2:
                    # For 2 agents, use alternating offers protocol
                    winner = self.alternating_offer_negotiate(contenders, maze)
                else:
                    # For multiple agents, use priority-based resolution
                    winner = self.resolve_conflict(contenders)
                
                self.successful_negotiations += 1
                for agent in contenders:
                    if agent == winner:
                        resolved_positions[agent] = pos
                    else:
                        self.waiting_times[agent.name] += 1
                        resolved_positions[agent] = agent.pos
                        agent.wait_time += 1
        return resolved_positions

    def resolve_conflict(self, contenders):
        """Negotiation Strategy 1: Priority-based resolution (highest score wins)"""
        self.priority_based_resolutions += 1
        
        # Log the conflict
        conflict_record = {
            'type': 'priority_based',
            'contenders': [a.name for a in contenders],
            'scores': [a.score for a in contenders],
            'energies': [a.energy for a in contenders],
            'timestamp': time.time()
        }
        self.conflict_log.append(conflict_record)
        
        # Higher score wins, else random among top scorers
        contenders.sort(key=lambda a: (a.score, a.energy), reverse=True)
        top = [a for a in contenders if a.score == contenders[0].score]
        winner = random.choice(top)
        
        # Log negotiation outcome
        negotiation_record = {
            'strategy': 'priority_based',
            'winner': winner.name,
            'contenders': [a.name for a in contenders],
            'resolution': f"{winner.name} won based on score/energy priority"
        }
        self.negotiation_log.append(negotiation_record)
        
        return winner

    def alternating_offer_negotiate(self, contenders, maze):
        """Negotiation Strategy 2: Formal alternating offers protocol"""
        self.alternating_offer_resolutions += 1
        
        if len(contenders) < 2:
            return self.resolve_conflict(contenders)
            
        # Select two agents for negotiation
        agent1, agent2 = random.sample(contenders, 2)
        
        # Message structure for alternating offers
        messages = []
        max_rounds = 3
        current_round = 0
        
        # Agent1 makes initial offer
        current_offer = {
            'from': agent1.name,
            'to': agent2.name,
            'proposal': 'yield_position',
            'compensation': 'future_pellet_share',
            'round': current_round
        }
        messages.append(current_offer)
        
        while current_round < max_rounds:
            # Evaluate utility of the offer for agent2
            utility_agent2 = self.evaluate_offer_utility(current_offer, agent2, maze)
            
            if utility_agent2 > 0.5:  # Accept if utility is high enough
                # Log successful negotiation
                negotiation_record = {
                    'strategy': 'alternating_offer',
                    'winner': agent1.name,  # Agent1 gets the position
                    'contenders': [agent1.name, agent2.name],
                    'rounds': current_round + 1,
                    'resolution': f"Accepted offer from {agent1.name} after {current_round + 1} rounds"
                }
                self.negotiation_log.append(negotiation_record)
                return agent1
            else:
                # Agent2 makes counter-offer
                current_round += 1
                counter_offer = {
                    'from': agent2.name,
                    'to': agent1.name,
                    'proposal': 'yield_position',
                    'compensation': 'immediate_energy_boost',
                    'round': current_round
                }
                messages.append(counter_offer)
                
                # Evaluate utility for agent1
                utility_agent1 = self.evaluate_offer_utility(counter_offer, agent1, maze)
                
                if utility_agent1 > 0.5:
                    # Log successful negotiation
                    negotiation_record = {
                        'strategy': 'alternating_offer',
                        'winner': agent2.name,  # Agent2 gets the position
                        'contenders': [agent1.name, agent2.name],
                        'rounds': current_round + 1,
                        'resolution': f"Accepted counter-offer from {agent2.name} after {current_round + 1} rounds"
                    }
                    self.negotiation_log.append(negotiation_record)
                    return agent2
        
        # If no agreement reached, fall back to priority-based
        negotiation_record = {
            'strategy': 'alternating_offer',
            'winner': 'fallback',
            'contenders': [agent1.name, agent2.name],
            'rounds': current_round,
            'resolution': "No agreement reached, falling back to priority-based"
        }
        self.negotiation_log.append(negotiation_record)
        return self.resolve_conflict([agent1, agent2])

    def evaluate_offer_utility(self, offer, agent, maze):
        """Evaluate the utility of an offer for an agent"""
        # Simple utility calculation based on agent state
        utility = 0.0
        
        if offer['proposal'] == 'yield_position':
            # If agent is being asked to yield, evaluate based on energy and score
            if agent.energy < 30:
                utility += 0.3  # Low energy agents more likely to yield
            if agent.score > 50:
                utility -= 0.2  # High score agents less likely to yield
                
        if offer['compensation'] == 'future_pellet_share':
            utility += 0.2
        elif offer['compensation'] == 'immediate_energy_boost':
            utility += 0.4
            
        # Add some randomness
        utility += random.uniform(-0.2, 0.2)
        
        return max(0.0, min(1.0, utility))

    def multi_issue_negotiate(self, contenders, pos):
        """Attempt multi-issue negotiation among contenders"""
        if len(contenders) < 2:
            return False

        # Randomly select two agents to negotiate
        agents_to_negotiate = random.sample(contenders, min(2, len(contenders)))
        agent1, agent2 = agents_to_negotiate[0], agents_to_negotiate[1]

        # Agent1 makes a proposal
        proposal = agent1.make_multi_issue_proposal([agent2], None)  # Maze not needed for now
        if proposal and random.random() < 0.5:  # 50% chance of acceptance for simplicity
            # For now, assume proposal is accepted: agent1 yields to agent2
            # In a real system, this would involve more complex logic
            agent1.learn_from_negotiation(True, agent2, 'corridor_access')
            agent2.learn_from_negotiation(True, agent1, 'corridor_access')
            return True
        return False

    def log_conflict_event(self, agents, position, conflict_type):
        """Log a conflict event with comprehensive details"""
        conflict_record = {
            'timestamp': time.time(),
            'position': position,
            'type': conflict_type,
            'agents_involved': [a.name for a in agents],
            'agent_scores': [a.score for a in agents],
            'agent_energies': [a.energy for a in agents]
        }
        self.conflict_log.append(conflict_record)

# ====================== Simulation Logic ======================

class PacmanSimulation:
    def __init__(self):
        self.maze = Maze()
        self.agents = [
            Agent("PacmanA", "A", (0, 0)),
            Agent("PacmanB", "B", (9, 0)),
            Agent("PacmanC", "C", (0, 9))
        ]
        self.ghosts = [
            Ghost("Ghost1", "G", (5, 5)),
            Ghost("Ghost2", "H", (5, 4))
        ]
        self.manager = ConflictManager()
        self.turns = 50

    def run(self):
        token_holder = 0  # Token passing for synchronization
        active_agents = self.agents.copy()
        
        for t in range(self.turns):
            print(f"===== Turn {t+1} =====")
            
            # Check win/loss conditions
            if len(self.maze.pellets) == 0:
                print("All pellets collected! Game over.")
                break
                
            active_agents = [a for a in active_agents if a.energy > 0 and a.active]
            if len(active_agents) == 0:
                print("All agents have dropped out!")
                break
            elif len(active_agents) == 1:
                print(f"Only {active_agents[0].name} remains!")
                break

            # Ghosts move
            for ghost in self.ghosts:
                if ghost.active:
                    new_pos = ghost.move(self.maze, active_agents)
                    ghost.pos = new_pos
                    # Check if ghost caught any agent
                    for agent in active_agents:
                        if agent.pos == ghost.pos and agent.active:
                            agent.update_energy(-20)  # Significant energy loss when caught
                            print(f"{agent.name} was caught by {ghost.name}! Energy reduced.")
                            if agent.energy <= 0:
                                agent.active = False
                                print(f"{agent.name} has dropped out due to ghost attack!")

            proposed_positions = {}
            for idx, agent in enumerate(active_agents):
                if not agent.active:
                    continue
                    
                # Only the token holder can propose a move in a shared corridor this turn
                if agent.pos in self.maze.shared_corridors:
                    if idx == token_holder:
                        direction = random.choice(['U', 'D', 'L', 'R'])
                        proposed_positions[agent] = agent.move(direction, self.maze)
                    else:
                        # Must wait for token
                        proposed_positions[agent] = agent.pos
                        agent.wait_time += 1
                else:
                    direction = random.choice(['U', 'D', 'L', 'R'])
                    proposed_positions[agent] = agent.move(direction, self.maze)

            resolved = self.manager.negotiate(active_agents, proposed_positions, self.maze)
            for agent, pos in resolved.items():
                agent.pos = pos
                if pos in self.maze.pellets:
                    agent.score += 10
                    self.maze.pellets.remove(pos)
                    self.maze.grid[pos[0]][pos[1]] = ' '
                if pos in self.maze.shared_corridors:
                    agent.update_energy(-1)
                else:
                    agent.update_energy(-0.5)
                    
                # Check if agent should drop out
                if agent.energy <= 0:
                    agent.active = False
                    print(f"{agent.name} has dropped out due to low energy!")

            # Display the maze with agents and ghosts
            self.maze.display(active_agents + self.ghosts)
            time.sleep(0.1)

            # Token passing: next agent gets the token for shared corridor
            token_holder = (token_holder + 1) % len(active_agents)

        self.report_metrics()

    def report_metrics(self):
        print("\n===== Simulation Report =====")
        active_agents = [a for a in self.agents if a.active]
        
        if active_agents:
            winner = max(active_agents, key=lambda a: a.score)
            print(f"WINNER: {winner.name} with {winner.score} points!")
        
        for agent in self.agents:
            status = "ACTIVE" if agent.active else "DROPPED OUT"
            print(f"{agent.name}: Score={agent.score}, Energy={agent.energy}, WaitTime={agent.wait_time} [{status}]")
        
        print(f"Conflicts: {self.manager.conflicts}")
        print(f"Successful Negotiations: {self.manager.successful_negotiations}")
        print(f"Multi-issue Negotiations: {self.manager.multi_issue_negotiations}")
        print(f"Priority-based Resolutions: {self.manager.priority_based_resolutions}")
        print(f"Alternating Offer Resolutions: {self.manager.alternating_offer_resolutions}")
        print(f"Pre-detected Conflicts: {self.manager.pre_detected_conflicts}")
        
        active_count = len(active_agents)
        if active_count > 0:
            avg_wait = sum(a.wait_time for a in self.agents) / len(self.agents)
            print(f"Average Wait: {avg_wait:.2f}")
            
            # Fairness metric: Gini coefficient approximation
            scores = [a.score for a in self.agents if a.score > 0]
            if scores:
                mean_score = sum(scores) / len(scores)
                fairness = 1 - (sum(abs(s - mean_score) for s in scores) / (2 * len(scores) * mean_score)) if mean_score > 0 else 1
                print(f"Fairness Index: {fairness:.3f}")
        
        # Detailed logging summary
        print(f"\n--- Detailed Logging ---")
        print(f"Total Conflict Events: {len(self.manager.conflict_log)}")
        print(f"Total Negotiation Events: {len(self.manager.negotiation_log)}")
        
        # Show negotiation strategy distribution
        strategy_counts = {}
        for log in self.manager.negotiation_log:
            strategy = log['strategy']
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} negotiations")
        
        print("=============================")

# ====================== Entry Point ======================

if __name__ == "__main__":
    sim = PacmanSimulation()
    sim.run()