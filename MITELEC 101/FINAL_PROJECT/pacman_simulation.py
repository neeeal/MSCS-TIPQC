import random
import time
from collections import defaultdict
import math
import csv
import os
import sys
import statistics

# --- CONFIGURATION ---
NUM_TRIALS = 1
FPS = 15 
MAX_TURNS = 1000 # Shorter max turns to ensure we get data quickly

# --- ANSI COLORS ---
class Colors:
    RESET = "\033[0m"
    RED = "\033[91m"      # Ghosts
    YELLOW = "\033[93m"   # Agent A
    CYAN = "\033[96m"     # Agent B
    MAGENTA = "\033[95m"  # Agent C
    BLUE = "\033[94m"     # Walls
    GREEN = "\033[92m"    # Text/Energy
    GREY = "\033[90m"     # Shared

# ====================== Environment Setup ======================

class Maze:
    def __init__(self, width=15, height=15):
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
        for _ in range(int(self.width * self.height * 0.2)):
            wall = (random.randint(0, self.height - 1), random.randint(0, self.width - 1))
            if wall not in agent_starts:
                self.grid[wall[0]][wall[1]] = '#'
                self.walls.add(wall)
        
        mid_row = self.height // 2
        for j in range(self.width // 2 - 3, self.width // 2 + 4):
            pos = (mid_row, j)
            if 0 <= j < self.width:
                if pos in self.walls: self.walls.remove(pos)
                self.grid[pos[0]][pos[1]] = '='
                self.shared_corridors.add(pos)
        
        for i in range(self.height):
            for j in range(self.width):
                pos = (i, j)
                if self.grid[i][j] == '.' or self.grid[i][j] == '=':
                    if pos not in self.walls:
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

# ====================== RL Agent Setup ======================

class Agent:
    def __init__(self, name, symbol, pos, color_code, energy=400): 
        self.name = name
        self.symbol = symbol
        self.pos = pos
        self.color_code = color_code
        self.energy = energy
        self.max_energy = energy 
        self.score = 0
        self.active = True
        self.current_lock = None
        self.sensing_radius = 4
        
        # --- METRICS TRACKING ---
        self.wait_time = 0 # Accumulates whenever agent doesn't move
        
        # --- Q-LEARNING BRAIN ---
        self.q_table = defaultdict(float) 
        self.alpha = 0.1      
        self.gamma = 0.9      
        self.epsilon = 0.2    
        self.last_state = None
        self.last_action = None
        self.last_reward_val = 0
        
        self.actions = ['U', 'D', 'L', 'R', 'W']

    def get_state(self, maze, ghosts):
        x, y = self.pos
        
        w_u = 1 if not (0 <= x-1 < maze.height and (x-1, y) not in maze.walls) else 0
        w_d = 1 if not (0 <= x+1 < maze.height and (x+1, y) not in maze.walls) else 0
        w_l = 1 if not (0 <= y-1 < maze.width and (x, y-1) not in maze.walls) else 0
        w_r = 1 if not (0 <= y+1 < maze.width and (x, y+1) not in maze.walls) else 0
        
        ghost_dir = 'None'
        ghost_is_critical = 0 
        
        min_g_dist = float('inf')
        closest_g = None
        
        for g in ghosts:
            if g.active:
                dist = abs(g.pos[0] - x) + abs(g.pos[1] - y)
                if dist < self.sensing_radius and dist < min_g_dist:
                    min_g_dist = dist
                    closest_g = g.pos
        
        if closest_g:
            gx, gy = closest_g
            if gx < x: ghost_dir = 'U'
            elif gx > x: ghost_dir = 'D'
            elif gy < y: ghost_dir = 'L'
            elif gy > y: ghost_dir = 'R'
            if min_g_dist <= 2: ghost_is_critical = 1

        target_dir = 'None'
        targets = list(maze.pellets)
        if not targets: targets = list(maze.shared_corridors)
        
        closest_t = None
        min_t_dist = float('inf')
        
        for t in targets:
            dist = abs(t[0] - x) + abs(t[1] - y)
            if dist < min_t_dist:
                min_t_dist = dist
                closest_t = t
        
        if closest_t:
            tx, ty = closest_t
            if abs(tx - x) > abs(ty - y):
                target_dir = 'U' if tx < x else 'D'
            else:
                target_dir = 'L' if ty < y else 'R'

        return (ghost_dir, ghost_is_critical, target_dir, w_u, w_d, w_l, w_r)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        q_values = [self.q_table[(state, a)] for a in self.actions]
        max_q = max(q_values)
        best_actions = [self.actions[i] for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, current_state, action, reward, next_state):
        old_value = self.q_table[(current_state, action)]
        next_max = max([self.q_table[(next_state, a)] for a in self.actions])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[(current_state, action)] = new_value

    def update_energy(self, delta):
        self.energy += delta
        if self.energy > self.max_energy: self.energy = self.max_energy
        if self.energy <= 0:
            self.energy = 0
            self.active = False

# ====================== Conflict & Ghost Setup ======================

class Ghost:
    def __init__(self, name, symbol, pos):
        self.name = name
        self.symbol = symbol
        self.pos = pos
        self.active = True

    def move(self, maze, agents):
        x, y = self.pos
        directions = ['U', 'D', 'L', 'R']
        if random.random() < 0.3:
            min_dist = float('inf')
            best_pos = None
            for dir in directions:
                dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[dir]
                nx, ny = x + dx, y + dy
                if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
                    for agent in agents:
                        if agent.active:
                            dist = abs(nx - agent.pos[0]) + abs(ny - agent.pos[1])
                            if dist < min_dist:
                                min_dist = dist
                                best_pos = (nx, ny)
            if best_pos: return best_pos

        random.shuffle(directions)
        for dir in directions:
             dx, dy = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[dir]
             nx, ny = x + dx, y + dy
             if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
                 return (nx, ny)
        return self.pos

class ConflictManager:
    def __init__(self, experiment_strategy, trial_num):
        self.conflicts = 0
        self.experiment_strategy = experiment_strategy 
        self.conflict_log = [] # For CSV
        self.current_time = 0 

    def set_time(self, t):
        self.current_time = t

    def negotiate(self, agents, proposed_positions, maze):
        reverse_map = defaultdict(list)
        resolved_positions = {}
        
        for agent, pos in proposed_positions.items():
            reverse_map[pos].append(agent)

        for pos, contenders in reverse_map.items():
            if len(contenders) == 1:
                agent = contenders[0]
                if pos in maze.shared_corridors:
                    if not maze.lock(pos, agent.name):
                        locker = maze.shared_route_status.get(pos)
                        if locker != agent.name:
                            resolved_positions[agent] = agent.pos 
                            self.conflicts += 1
                            self.log_conflict(agent, pos, "Lock_Collision") # LOG
                        else:
                            resolved_positions[agent] = pos
                    else:
                        agent.current_lock = pos
                        resolved_positions[agent] = pos
                else:
                    resolved_positions[agent] = pos
            else:
                self.conflicts += 1
                winner = random.choice(contenders) 
                self.log_conflict(contenders, pos, "Position_Collision") # LOG
                
                for agent in contenders:
                    if agent == winner:
                        resolved_positions[agent] = pos
                        if pos in maze.shared_corridors:
                            maze.lock(pos, agent.name)
                            agent.current_lock = pos
                    else:
                        resolved_positions[agent] = agent.pos 

        return resolved_positions

    def log_conflict(self, agents, position, type):
        if not isinstance(agents, list): agents = [agents]
        record = {
            'time': self.current_time,
            'position': position,
            'type': type,
            'agents': [a.name for a in agents]
        }
        self.conflict_log.append(record)

# ====================== Simulation Wrapper ======================

class PacmanSimulation:
    def __init__(self, strategy='Q_LEARNING', trial_num=1):
        self.maze = Maze()
        self.manager = ConflictManager(strategy, trial_num)
        self.agents = [
            Agent("PacmanA", "A", (0, 0), Colors.YELLOW),
            Agent("PacmanB", "B", (self.maze.height - 1, 0), Colors.CYAN),
            Agent("PacmanC", "C", (0, self.maze.width - 1), Colors.MAGENTA)
        ]
        self.ghosts = [
            Ghost("Ghost1", "G", (self.maze.height // 2 - 1, self.maze.width // 2 - 1)),
            Ghost("Ghost2", "H", (self.maze.height // 2 + 1, self.maze.width // 2 + 1))
        ]
        self.time_step = 0
        self.max_turns = MAX_TURNS 
        self.running = True
        self.energy_loss_ghost = -50 

    def render_text(self):
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{Colors.GREEN}Time: {self.time_step} | RL Mode: Metabolism + Fear{Colors.RESET}")
        print("-" * (self.maze.width * 2 + 2))

        agent_map = {a.pos: a for a in self.agents if a.active}
        ghost_map = {g.pos: g for g in self.ghosts if g.active}

        for r in range(self.maze.height):
            row_str = "|"
            for c in range(self.maze.width):
                pos = (r, c)
                char = " "
                if pos in ghost_map:
                    char = f"{Colors.RED}{ghost_map[pos].symbol}{Colors.RESET}"
                elif pos in agent_map:
                    a = agent_map[pos]
                    char = f"{a.color_code}{a.symbol}{Colors.RESET}"
                elif self.maze.grid[r][c] == '#':
                    char = f"{Colors.BLUE}#{Colors.RESET}"
                elif self.maze.grid[r][c] == '=':
                    char = f"{Colors.GREY}={Colors.RESET}"
                elif pos in self.maze.pellets:
                    char = "."
                row_str += char + " " 
            print(row_str + "|")
        print("-" * (self.maze.width * 2 + 2))

        # Display Metrics Live
        for agent in self.agents:
            color = agent.color_code if agent.active else Colors.GREY
            max_bars = 10
            pct = max(0, agent.energy / agent.max_energy)
            filled = int(max_bars * pct)
            bar = "|" * filled + "." * (max_bars - filled)
            print(f"{color}{agent.name}: [{bar}] En:{int(agent.energy)} Wait:{agent.wait_time} Rwd:{agent.last_reward_val}{Colors.RESET}")

    def update(self):
        if not any(a.active for a in self.agents) or not self.maze.pellets or self.time_step >= self.max_turns:
            self.running = False
            return

        self.manager.set_time(self.time_step)

        # 1. GHOSTS
        for ghost in self.ghosts:
            if ghost.active:
                ghost.pos = ghost.move(self.maze, self.agents)

        # 2. AGENT DECISION
        proposed_positions = {}
        agent_intents = {} 

        for agent in self.agents:
            if not agent.active: continue
            state = agent.get_state(self.maze, self.ghosts)
            action = agent.choose_action(state)
            intended_pos = agent.pos
            
            if action == 'U': target = (agent.pos[0]-1, agent.pos[1])
            elif action == 'D': target = (agent.pos[0]+1, agent.pos[1])
            elif action == 'L': target = (agent.pos[0], agent.pos[1]-1)
            elif action == 'R': target = (agent.pos[0], agent.pos[1]+1)
            elif action == 'W': target = agent.pos
            
            hit_wall = False
            if not (0 <= target[0] < self.maze.height and 0 <= target[1] < self.maze.width and target not in self.maze.walls):
                intended_pos = agent.pos 
                hit_wall = True
            else:
                intended_pos = target

            proposed_positions[agent] = intended_pos
            agent_intents[agent] = {'state': state, 'action': action, 'hit_wall': hit_wall}

        # 3. CONFLICTS
        resolved = self.manager.negotiate(self.agents, proposed_positions, self.maze)
        
        # 4. REWARDS & METRICS
        for agent, actual_pos in resolved.items():
            if not agent.active: continue
            
            intent = agent_intents[agent]
            prev_state = intent['state']
            action = intent['action']
            
            # --- METRIC: WAIT TIME TRACKING ---
            # If the agent did NOT change position, it "waited" (due to wall, conflict, or choice)
            if actual_pos == agent.pos:
                agent.wait_time += 1
            # ----------------------------------

            if agent.current_lock and actual_pos != agent.current_lock:
                self.maze.unlock(agent.current_lock)
                agent.current_lock = None
            
            agent.pos = actual_pos
            reward = -1 
            
            hit_ghost = False
            for g in self.ghosts:
                if g.active and g.pos == agent.pos:
                    hit_ghost = True
                    break
            
            if hit_ghost:
                reward = -100 
                agent.update_energy(self.energy_loss_ghost) 
            else:
                if intent['hit_wall']: reward -= 5 
                if actual_pos in self.maze.pellets:
                    reward += 20
                    agent.score += 10
                    agent.update_energy(15) 
                    self.maze.pellets.remove(actual_pos)
                elif actual_pos != agent.pos:
                    reward -= 2 
                if proposed_positions[agent] != actual_pos and not intent['hit_wall']:
                    reward -= 2 

                min_g_dist = float('inf')
                for g in self.ghosts:
                    d = abs(g.pos[0] - agent.pos[0]) + abs(g.pos[1] - agent.pos[1])
                    if d < min_g_dist: min_g_dist = d
                if min_g_dist <= 2: reward -= 5 

            new_state = agent.get_state(self.maze, self.ghosts)
            agent.learn(prev_state, action, reward, new_state)
            agent.last_reward_val = reward 
            agent.last_state = prev_state
            agent.last_action = action
            if agent.epsilon > 0.01: agent.epsilon *= 0.999
            agent.update_energy(-0.5) 

        self.time_step += 1

    def save_logs(self):
        """Calculates Fairness and Saves Logs"""
        print("\n--- CALULATING FINAL METRICS ---")
        
        # Calculate Fairness
        all_waits = [a.wait_time for a in self.agents]
        n = len(all_waits)
        fairness_index = 0
        if sum(all_waits) > 0:
            numerator = sum(all_waits) ** 2
            denominator = n * sum(w ** 2 for w in all_waits)
            fairness_index = numerator / denominator if denominator != 0 else 0
        
        avg_wait = statistics.mean(all_waits) if all_waits else 0
        
        print(f"Total Conflicts: {self.manager.conflicts}")
        print(f"Avg Wait Time: {avg_wait:.2f}")
        print(f"Jain's Fairness Index: {fairness_index:.4f}")
        
        # Write to CSV
        filename = "MITELEC101_RL_Metrics.csv"
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Total Turns', self.time_step])
            writer.writerow(['Total Conflicts', self.manager.conflicts])
            writer.writerow(['Avg Wait Time', avg_wait])
            writer.writerow(['Fairness Index', fairness_index])
            writer.writerow(['PacmanA Wait', self.agents[0].wait_time])
            writer.writerow(['PacmanB Wait', self.agents[1].wait_time])
            writer.writerow(['PacmanC Wait', self.agents[2].wait_time])
            
        print(f"Metrics saved to {filename}")

        # Write Conflict Log
        if self.manager.conflict_log:
            with open("MITELEC101_RL_Conflict_Log.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['time', 'position', 'type', 'agents'])
                writer.writeheader()
                writer.writerows(self.manager.conflict_log)
            print("Detailed Conflict Log saved.")

if __name__ == "__main__":
    random.seed(time.time())
    sim = PacmanSimulation()
    print("Starting Text-Based Simulation...")
    
    try:
        while sim.running:
            sim.update()
            sim.render_text()
            time.sleep(1 / FPS)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    
    # SAVE METRICS ON EXIT
    sim.save_logs()