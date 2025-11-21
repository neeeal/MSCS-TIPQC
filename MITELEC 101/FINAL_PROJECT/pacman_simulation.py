import random
import time
from collections import defaultdict
import math
import statistics
import csv
import os

# --- CONFIGURATION ---
NUM_TRIALS = 30
# ---------------------

# ====================== Environment Setup (UNCHANGED) ======================

class Maze:
    def __init__(self, width=10, height=10):
        self.width = width
        self.height = height
        self.grid = [['.' for _ in range(width)] for _ in range(height)]
        self.shared_corridors = set()
        self.walls = set()
        self.pellets = set()
        self.shared_route_status = {}  
        self._generate_maze()

    def _generate_maze(self):
        agent_starts = {(0, 0), (self.height - 1, 0), (0, self.width - 1)}
        for _ in range(int(self.width * self.height * 0.1)):
            wall = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if wall not in agent_starts:
                self.grid[wall[0]][wall[1]] = '#'
                self.walls.add(wall)
        mid_row = self.height // 2
        for j in range(self.width // 2 - 2, self.width // 2 + 3):
            pos = (mid_row, j)
            if pos not in self.walls and pos not in agent_starts:
                self.grid[pos[0]][pos[1]] = '='
                self.shared_corridors.add(pos)
        for i in range(self.height):
            for j in range(self.width):
                pos = (i, j)
                if self.grid[i][j] == '.':
                    self.pellets.add(pos)
                if pos in self.shared_corridors:
                     self.shared_route_status[pos] = None

    def lock(self, position, agent_name):
        if position in self.shared_corridors:
            if self.shared_route_status[position] is None:
                self.shared_route_status[position] = agent_name
                return True
            return False 
        return True

    def unlock(self, position):
        if position in self.shared_corridors:
            self.shared_route_status[position] = None
            return True
        return False

# ====================== Agent Setup (ADDED FORMAL GOALS) ======================

class Agent:
    def __init__(self, name, symbol, pos, energy=300): 
        self.name = name
        self.symbol = symbol
        self.pos = pos
        self.energy = energy
        self.score = 0
        self.wait_time = 0
        self.failed_negotiations = 0
        self.active = True
        self.current_lock = None
        self.sensing_radius = 3
        # Formal Goal-Based Architecture Components (for thesis documentation)
        self.Goals = ['Maximize Score', 'Maintain Energy > 0'] 
        self.Beliefs = {'SharedRouteLocked': False, 'OpponentScore': 0} # Beliefs are derived from perception/sensing
        self.Intentions = None # Represents the committed course of action (e.g., path plan)


    def move(self, direction, maze):
        x, y = self.pos
        dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}.get(direction, (0, 0))
        nx, ny = x + dx, y + dy
        if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
            return (nx, ny)
        return self.pos

    def update_energy(self, delta):
        self.energy += delta
        if self.energy <= 0:
            self.energy = 0
            self.active = False
            
    def make_alternating_offer(self, opponent, corridor_pos, round_num):
        # The logic here acts as the Agent's decision module based on perceived state (Beliefs)
        if round_num % 2 == 0:
            compensation = 'immediate_energy_boost' if opponent.energy < 150 else 'future_pellet_share'
            return {
                'from': self.name,
                'to': opponent.name,
                'performative': 'PROPOSE',
                'content': {'action': 'I_GO_FIRST', 'compensation': compensation, 'waits': 1}
            }
        else:
            return {
                'from': self.name,
                'to': opponent.name,
                'performative': 'PROPOSE',
                'content': {'action': 'YOU_GO_FIRST', 'compensation': 'immediate_energy_boost', 'waits': 0}
            }

# ====================== Ghost Setup (RE-INTRODUCED) ======================

class Ghost:
    def __init__(self, name, symbol, pos):
        self.name = name
        self.symbol = symbol
        self.pos = pos
        self.active = True

    def move(self, maze, agents):
        x, y = self.pos
        directions = ['U', 'D', 'L', 'R']
        random.shuffle(directions)
        
        min_dist = float('inf')
        best_pos = None
        
        for dir in directions:
            dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[dir]
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
                for agent in agents:
                    if agent.active:
                        dist = math.sqrt((nx - agent.pos[0])**2 + (ny - agent.pos[1])**2)
                        if dist < min_dist:
                            min_dist = dist
                            best_pos = (nx, ny)

        if best_pos:
            return best_pos
        
        for dir in directions:
             dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[dir]
             nx, ny = x + dx, y + dy
             if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
                 return (nx, ny)
        
        return self.pos

# ====================== Negotiation & Conflict Handling (Modified to remove verbose prints) ======================

class ConflictManager:
    def __init__(self, experiment_strategy, trial_num):
        self.conflicts = 0
        self.successful_negotiations = 0
        self.priority_based_resolutions = 0
        self.alternating_offer_resolutions = 0
        self.conflict_log = []
        self.negotiation_log = []
        self.pre_detected_conflicts = 0
        self.experiment_strategy = experiment_strategy 
        self.PRIORITY_PENALTY = -2 
        self.FALLBACK_PENALTY = -8
        self.trial_num = trial_num
        self.current_time = 0 

    def set_time(self, t):
        self.current_time = t

    def negotiate(self, agents, proposed_positions, maze):
        reverse_map = defaultdict(list)
        resolved_positions = {}
        agents_in_conflict = set()

        for agent, pos in proposed_positions.items():
            reverse_map[pos].append(agent)
            resolved_positions[agent] = pos 

        # 2. Resolve Direct Position Collisions
        for pos, contenders in reverse_map.items():
            if len(contenders) > 1:
                self.conflicts += 1
                self.log_conflict_event(contenders, pos, 'Position_Collision')
                
                if self.experiment_strategy == 'ALTERNATING_OFFER' and len(contenders) == 2:
                    winner, success = self.alternating_offer_negotiate(contenders, pos)
                    self.alternating_offer_resolutions += 1
                else:
                    winner, success = self.resolve_conflict(contenders)
                    self.priority_based_resolutions += 1
                
                if success: self.successful_negotiations += 1
                
                for agent in contenders:
                    if agent == winner:
                        resolved_positions[agent] = pos
                        if pos in maze.shared_corridors:
                            maze.lock(pos, agent.name)
                            agent.current_lock = pos
                    else:
                        self.penalize_loser(agent, 'Position_Conflict_Loss', self.PRIORITY_PENALTY, winner.name)
                        resolved_positions[agent] = agent.pos
                agents_in_conflict.update(contenders)
        
        # 3. Check for Shared Corridor Contention/Lock Failures
        for agent, pos in proposed_positions.items():
            if agent in agents_in_conflict or agent.current_lock == pos:
                continue 

            if pos in maze.shared_corridors:
                if not maze.lock(pos, agent.name):
                    locking_agent_name = maze.shared_route_status.get(pos)
                    locking_agent = next((a for a in agents if a.name == locking_agent_name), None)

                    if locking_agent and agent != locking_agent:
                        self.conflicts += 1
                        self.pre_detected_conflicts += 1
                        contenders = [agent, locking_agent]
                        
                        if self.experiment_strategy == 'ALTERNATING_OFFER':
                            winner, success = self.alternating_offer_negotiate(contenders, pos)
                            self.alternating_offer_resolutions += 1
                            if not success: 
                                self.priority_based_resolutions += 1
                        else:
                            winner, success = self.resolve_conflict(contenders)
                            self.priority_based_resolutions += 1
                        
                        if winner:
                            self.successful_negotiations += 1
                            if winner == agent:
                                maze.unlock(pos)
                                locking_agent.current_lock = None
                                self.penalize_loser(locking_agent, 'Negotiation_Yield', self.PRIORITY_PENALTY, agent.name)
                                resolved_positions[locking_agent] = locking_agent.pos
                                maze.lock(pos, agent.name)
                                agent.current_lock = pos
                                resolved_positions[agent] = pos
                            elif winner == locking_agent:
                                self.penalize_loser(agent, 'Negotiation_Loss', self.PRIORITY_PENALTY, locking_agent.name)
                                resolved_positions[agent] = agent.pos
                    
                else:
                    agent.current_lock = pos

        return resolved_positions

    def penalize_loser(self, agent, reason, penalty, winner_name):
        agent.wait_time += 1
        agent.update_energy(penalty) 
        self.negotiation_log.append({
            'time': self.current_time,
            'strategy': self.experiment_strategy,
            'loser': agent.name,
            'winner': winner_name,
            'reason': reason,
            'penalty': penalty,
            'loser_score': agent.score
        })

    def resolve_conflict(self, contenders):
        """Negotiation Strategy 1: Priority-based resolution (highest score wins)"""
        self.log_conflict_event(contenders, contenders[0].pos, 'priority_based')
        contenders.sort(key=lambda a: (a.score, a.energy), reverse=True)
        top_score = contenders[0].score
        top_contenders = [a for a in contenders if a.score == top_score]
        winner = random.choice(top_contenders)
        return winner, True

    def alternating_offer_negotiate(self, contenders, pos):
        """Negotiation Strategy 2: Formal alternating offers protocol"""
        if len(contenders) != 2:
            return self.resolve_conflict(contenders) 
        
        agent1, agent2 = contenders[0], contenders[1]
        if agent1.score < agent2.score:
            proposer, respondent = agent1, agent2
        else:
            proposer, respondent = agent2, agent1
        
        winner = None
        max_rounds = 3
        
        for current_round in range(max_rounds):
            offer = proposer.make_alternating_offer(respondent, pos, current_round)
            utility = self.evaluate_offer_utility(offer, respondent)
            
            if utility > 0.5: # ACCEPT
                winner = proposer if offer['content']['action'] == 'I_GO_FIRST' else respondent
                return winner, True
            else: # REJECT (Counter-offer)
                proposer, respondent = respondent, proposer 
        
        # Negotiation failed (Fallback)
        for agent in contenders:
            agent.update_energy(self.FALLBACK_PENALTY) 
            agent.failed_negotiations += 1
            
        return self.resolve_conflict(contenders)

    def evaluate_offer_utility(self, offer, agent):
        # This utility function reflects the agent's *Goal* to maintain energy and maximize score
        cost_of_yielding = 0.5
        benefit_of_compensation = 0.0
        
        if offer['content'].get('compensation') == 'immediate_energy_boost':
            benefit_of_compensation = 0.8
        elif offer['content'].get('compensation') == 'future_pellet_share':
            benefit_of_compensation = 0.4
        
        # Agent rationality is influenced by current state (Beliefs)
        utility = (benefit_of_compensation - cost_of_yielding) * (1 - agent.energy/600) + 0.5
        utility += random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, utility))

    def log_conflict_event(self, agents, position, conflict_type):
        conflict_record = {
            'time': self.current_time,
            'position': position,
            'type': conflict_type,
            'agents_involved': [a.name for a in agents],
            'agent_scores': [a.score for a in agents],
        }
        self.conflict_log.append(conflict_record)


# ====================== Simulation Logic ======================

class PacmanSimulation:
    def __init__(self, strategy='ALTERNATING_OFFER', trial_num=1):
        self.maze = Maze()
        self.manager = ConflictManager(strategy, trial_num)
        self.agents = [
            Agent("PacmanA", "A", (0, 0)),
            Agent("PacmanB", "B", (self.maze.height - 1, 0)),
            Agent("PacmanC", "C", (0, self.maze.width - 1))
        ]
        self.ghosts = [
            Ghost("Ghost1", "G", (self.maze.height // 2 - 1, self.maze.width // 2 - 1)),
            Ghost("Ghost2", "H", (self.maze.height // 2 + 1, self.maze.width // 2 + 1))
        ]
        self.ghost_log = [] 
        self.turns = 250
        self.trial_num = trial_num
        self.energy_loss_ghost = -20 

    def log_ghost_event(self, time, ghost, agent=None, type='Move'):
        event = {
            'time': time,
            'ghost_name': ghost.name,
            'ghost_pos': ghost.pos,
            'event_type': type
        }
        if agent:
            event['agent_name'] = agent.name
            event['agent_energy_after'] = agent.energy
        self.ghost_log.append(event)


    def run(self):
        
        for t in range(self.turns):
            active_agents = [a for a in self.agents if a.active]
            
            if not active_agents or len(self.maze.pellets) == 0:
                break
            
            self.manager.set_time(t)

            # --- GHOST MOVEMENT AND ATTACK LOGIC ---
            for ghost in self.ghosts:
                if ghost.active:
                    new_pos = ghost.move(self.maze, active_agents)
                    self.log_ghost_event(t, ghost, type='Move_Start') 
                    ghost.pos = new_pos
                    self.log_ghost_event(t, ghost, type='Move_End') 
                    
                    for agent in active_agents:
                        if agent.pos == ghost.pos and agent.active:
                            agent.update_energy(self.energy_loss_ghost)
                            self.log_ghost_event(t, ghost, agent, type='Capture')
                            
            # --- AGENT MOVEMENT ---
            proposed_positions = {}
            for agent in active_agents:
                closest_shared_pos = None
                min_dist = float('inf')
                for pos in self.maze.shared_corridors:
                    dist = abs(pos[0] - agent.pos[0]) + abs(pos[1] - agent.pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest_shared_pos = pos
                        
                if closest_shared_pos and min_dist < self.maze.width / 2:
                    current_x, current_y = agent.pos
                    target_x, target_y = closest_shared_pos
                    move_dir = random.choice(['U', 'D', 'L', 'R']) 
                    
                    if abs(target_x - current_x) >= abs(target_y - current_y):
                        if target_x > current_x: move_dir = 'D'
                        elif target_x < current_x: move_dir = 'U'
                    else:
                        if target_y > current_y: move_dir = 'R'
                        elif target_y < current_y: move_dir = 'L'
                    target_pos = agent.move(move_dir, self.maze)

                elif self.maze.pellets:
                    pellet_pos = random.choice(list(self.maze.pellets))
                    current_x, current_y = agent.pos
                    target_x, target_y = pellet_pos
                    
                    move_dir = random.choice(['U', 'D', 'L', 'R'])
                    if abs(target_x - current_x) >= abs(target_y - current_y):
                        if target_x > current_x: move_dir = 'D'
                        elif target_x < current_x: move_dir = 'U'
                    else:
                        if target_y > current_y: move_dir = 'R'
                        elif target_y < current_y: move_dir = 'L'
                    target_pos = agent.move(move_dir, self.maze)
                else:
                    direction = random.choice(['U', 'D', 'L', 'R'])
                    target_pos = agent.move(direction, self.maze)
                        
                proposed_positions[agent] = target_pos

            resolved = self.manager.negotiate(active_agents, proposed_positions, self.maze)
            
            for agent, pos in resolved.items():
                if agent.current_lock is not None and pos != agent.current_lock:
                    self.maze.unlock(agent.current_lock)
                    agent.current_lock = None
                    
                agent.pos = pos
                
                if pos in self.maze.pellets:
                    agent.score += 10
                    self.maze.pellets.remove(pos)
                    
                if pos in self.maze.shared_corridors:
                    agent.update_energy(-1.0) 
                else:
                    agent.update_energy(-0.2)
        
        return self.report_metrics()

    def report_metrics(self):
        active_agents = [a for a in self.agents if a.active]
        
        total_wait_time = sum(a.wait_time for a in self.agents)
        avg_wait = total_wait_time / len(self.agents)
        
        fairness_index = 1.000 
        all_waits = [a.wait_time for a in self.agents]
        total_waits = sum(all_waits)
        
        if total_waits > 0 and len(all_waits) > 1:
            n = len(all_waits)
            numerator = sum(all_waits) ** 2
            denominator = n * sum(w ** 2 for w in all_waits)
            fairness_index = numerator / denominator if denominator != 0 else 0
            
        metrics = {
            'strategy': self.manager.experiment_strategy,
            'trial_num': self.trial_num,
            'avg_wait_time': avg_wait,
            'access_fairness_index': fairness_index,
            'conflicts': self.manager.conflicts,
        }
        
        metrics['conflict_log'] = self.manager.conflict_log
        metrics['negotiation_log'] = self.manager.negotiation_log
        metrics['ghost_log'] = self.ghost_log 
        
        return metrics

# ====================== Automation Execution with CSV Writing ======================

def write_csv_report(filename, header, data, mode='w'):
    """Writes or appends data to a CSV file."""
    if not data: return 
    with open(filename, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if mode == 'w':
            writer.writeheader()
        writer.writerows(data)

def run_experiment(strategy, num_trials, final_summary_writer):
    """Runs a simulation strategy multiple times and generates CSV reports."""
    
    STRATEGY_SHORT = 'AO' if strategy == 'ALTERNATING_OFFER' else 'PB'
    MASTER_CSV_FILENAME = f"MITELEC101_Trial_Summary_{STRATEGY_SHORT}.csv"
    DETAIL_DIR = f"MITELEC101_{STRATEGY_SHORT}_Conflict_Details"
    
    os.makedirs(DETAIL_DIR, exist_ok=True)
    
    MASTER_HEADER = ['trial_num', 'avg_wait_time', 'access_fairness_index', 'conflicts']
    CONFLICT_DETAIL_HEADER = ['time', 'position', 'type', 'agents_involved', 'agent_scores']
    NEGOTIATION_DETAIL_HEADER = ['time', 'strategy', 'loser', 'winner', 'reason', 'penalty', 'loser_score']
    GHOST_DETAIL_HEADER = ['time', 'ghost_name', 'ghost_pos', 'event_type', 'agent_name', 'agent_energy_after']

    write_csv_report(MASTER_CSV_FILENAME, ['strategy'] + MASTER_HEADER, [], mode='w')
    
    all_results = defaultdict(list)
    master_summary_data = []

    print(f"\n--- Running Automated Experiment: {strategy} ({num_trials} Trials) ---")
    
    for i in range(1, num_trials + 1):
        random.seed(i)
        
        sim = PacmanSimulation(strategy=strategy, trial_num=i)
        metrics = sim.run()
        
        # 3. Final Trial Report (Master.csv)
        summary_row = {k: metrics[k] for k in MASTER_HEADER}
        master_summary_data.append({'strategy': strategy, **summary_row})

        # 2. Conflict Detail Reports (CSV Output for logs)
        write_csv_report(os.path.join(DETAIL_DIR, f"Conflict_Trial_{i}.csv"), CONFLICT_DETAIL_HEADER, metrics['conflict_log'])
        write_csv_report(os.path.join(DETAIL_DIR, f"Negotiation_Trial_{i}.csv"), NEGOTIATION_DETAIL_HEADER, metrics['negotiation_log'])
        write_csv_report(os.path.join(DETAIL_DIR, f"Ghost_Trial_{i}.csv"), GHOST_DETAIL_HEADER, metrics['ghost_log']) 
        
        # Accumulate metrics for averaging
        for key in MASTER_HEADER[1:]:
            all_results[key].append(metrics[key])

    # Append all trial summaries to the master CSV
    write_csv_report(MASTER_CSV_FILENAME, ['strategy'] + MASTER_HEADER, master_summary_data, mode='a')
    print(f"\n>>> MASTER TRIAL SUMMARY SAVED TO: {MASTER_CSV_FILENAME} <<<")
    print(f">>> DETAILED CONFLICT LOGS SAVED TO: {DETAIL_DIR}/ <<<")


    # Calculate Averages
    avg_metrics = {
        'Strategy': strategy,
        'Avg Wait Time (Efficiency)': statistics.mean(all_results['avg_wait_time']),
        'Avg Access Fairness (Jain\'s)': statistics.mean(all_results['access_fairness_index']),
        'Total Conflicts (Avg)': statistics.mean(all_results['conflicts']),
    }
    
    # 4. Final Statistical Report (Summary.csv)
    final_summary_writer.writerow(avg_metrics)

    # Print Summary Report (Console Output)
    print("\n\n" + "#"*50)
    print(f"### FINAL STATISTICAL REPORT: {strategy} ({num_trials} TRIALS) ###")
    print("#"*50)
    print(f"{'Metric':<35} {'Average Result':>15}")
    print("-" * 50)
    for key, value in avg_metrics.items():
        if isinstance(value, float):
            print(f"{key:<35} {value:>15.3f}")
        else:
            print(f"{key:<35} {value}")
    print("#"*50)
    
    return avg_metrics

if __name__ == "__main__":
    
    random.seed(time.time())

    FINAL_SUMMARY_FILE = "MITELEC101_Final_Statistical_Summary.csv"
    FINAL_SUMMARY_HEADER = ['Strategy', 'Avg Wait Time (Efficiency)', 'Avg Access Fairness (Jain\'s)', 'Total Conflicts (Avg)']
    
    with open(FINAL_SUMMARY_FILE, 'w', newline='') as f_summary:
        writer = csv.DictWriter(f_summary, fieldnames=FINAL_SUMMARY_HEADER)
        writer.writeheader()

        # Run Alternating Offer (Strategy 2)
        run_experiment('ALTERNATING_OFFER', NUM_TRIALS, writer)

        print("\n" + "="*80 + "\n")

        # Run Priority-Based (Strategy 1)
        run_experiment('PRIORITY_BASED', NUM_TRIALS, writer)
        
    print(f"\nFinal Averages Report saved to: {FINAL_SUMMARY_FILE}")