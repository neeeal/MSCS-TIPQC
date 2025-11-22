import random
import time
from collections import defaultdict, deque 
import math
import csv
import os
import pygame
import statistics

# --- CONFIGURATION ---
NUM_TRIALS = 300  # Loop N times
CELL_SIZE = 40
SIDEBAR_WIDTH = 250
FPS = 1000  # Extremely fast to get through N trials
MAX_TURNS = 2000

# --- COLORS ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)     
GREY = (100, 100, 100) 
RED = (255, 0, 0)      
YELLOW = (255, 255, 0) 
CYAN = (0, 255, 255)   
MAGENTA = (255, 0, 255)
GREEN = (0, 255, 0)    
ORANGE = (255, 165, 0) 

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
        """Generates a maze and ensures full connectivity using Flood Fill."""
        while True:
            self.grid = [['.' for _ in range(self.width)] for _ in range(self.height)]
            self.walls = set()
            self.shared_corridors = set()
            self.pellets = set()
            
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
            
            if self._is_fully_connected(agent_starts):
                break 

        for i in range(self.height):
            for j in range(self.width):
                pos = (i, j)
                if self.grid[i][j] == '.' or self.grid[i][j] == '=':
                    if pos not in self.walls:
                        self.pellets.add(pos)
                if pos in self.shared_corridors:
                     self.shared_route_status[pos] = None

    def _is_fully_connected(self, required_points):
        start_node = (0, 0)
        if start_node in self.walls: return False
        visited = set()
        queue = deque([start_node])
        visited.add(start_node)
        valid_tiles_count = 0
        for r in range(self.height):
            for c in range(self.width):
                if (r,c) not in self.walls: valid_tiles_count += 1
        reached_count = 0
        while queue:
            curr = queue.popleft()
            reached_count += 1
            r, c = curr
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if (nr, nc) not in self.walls and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        
        if reached_count != valid_tiles_count: return False
        for p in required_points:
            if p not in visited: return False
        return True

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

# ====================== Agent Setup ======================

class Agent:
    def __init__(self, name, symbol, pos, color, energy=400, existing_q_table=None): 
        self.name = name
        self.symbol = symbol
        self.pos = pos
        self.color = color
        self.energy = energy
        self.max_energy = energy 
        self.score = 0
        self.active = True
        self.current_lock = None
        self.sensing_radius = 4
        self.wait_time = 0 
        
        if existing_q_table is not None:
            self.q_table = existing_q_table
        else:
            self.q_table = defaultdict(float) 
            
        self.alpha = 0.2      
        self.gamma = 0.9      
        self.epsilon = 0.4    
        self.last_state = None
        self.last_action = None
        self.last_reward_val = 0
        self.recent_history = deque(maxlen=4) 
        self.prev_dist_to_target = float('inf') 
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
        
        self.current_closest_target_dist = min_t_dist 

        if closest_t:
            tx, ty = closest_t
            if abs(tx - x) > abs(ty - y): target_dir = 'U' if tx < x else 'D'
            else: target_dir = 'L' if ty < y else 'R'

        return (ghost_dir, ghost_is_critical, target_dir, w_u, w_d, w_l, w_r)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon: return random.choice(self.actions)
        q_values = [self.q_table[(state, a)] for a in self.actions]
        max_q = max(q_values)
        best_actions = [self.actions[i] for i, q in enumerate(q_values) if q == max_q]
        return random.choice(best_actions)

    def learn(self, current_state, action, reward, next_state):
        old_value = self.q_table[(current_state, action)]
        next_max = max([self.q_table[(next_state, a)] for a in self.actions])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[(current_state, action)] = new_value

    def make_alternating_offer(self, opponent, corridor_pos, round_num):
        if round_num % 2 == 0:
            compensation = 'immediate_energy_boost' if opponent.energy < 150 else 'future_pellet_share'
            return {'from': self.name, 'to': opponent.name, 'performative': 'PROPOSE', 'content': {'action': 'I_GO_FIRST', 'compensation': compensation, 'waits': 1}}
        else:
            return {'from': self.name, 'to': opponent.name, 'performative': 'PROPOSE', 'content': {'action': 'YOU_GO_FIRST', 'compensation': 'immediate_energy_boost', 'waits': 0}}

    def move_heuristic(self, maze, ghosts):
        moves = [('U', (-1, 0)), ('D', (1, 0)), ('L', (0, -1)), ('R', (0, 1))]
        valid_moves = []
        closest_ghost = None
        min_g_dist = float('inf')
        for g in ghosts:
            d = abs(g.pos[0] - self.pos[0]) + abs(g.pos[1] - self.pos[1])
            if d < min_g_dist: min_g_dist = d; closest_ghost = g.pos

        for direction, (dx, dy) in moves:
            nx, ny = self.pos[0] + dx, self.pos[1] + dy
            if 0 <= nx < maze.height and 0 <= ny < maze.width and (nx, ny) not in maze.walls:
                valid_moves.append((nx, ny))
        
        if not valid_moves: return self.pos
        if min_g_dist <= 3:
            valid_moves.sort(key=lambda p: abs(p[0] - closest_ghost[0]) + abs(p[1] - closest_ghost[1]), reverse=True)
            return valid_moves[0]
        
        closest_p = None
        min_p_dist = float('inf')
        targets = list(maze.pellets) if maze.pellets else list(maze.shared_corridors)
        for p in targets:
            d = abs(p[0] - self.pos[0]) + abs(p[1] - self.pos[1])
            if d < min_p_dist: min_p_dist = d; closest_p = p
        if closest_p:
            valid_moves.sort(key=lambda p: abs(p[0] - closest_p[0]) + abs(p[1] - closest_p[1]))
            return valid_moves[0]
        return random.choice(valid_moves)

    def update_energy(self, delta):
        self.energy += delta
        if self.energy > self.max_energy: self.energy = self.max_energy
        if self.energy <= 0: self.energy = 0; self.active = False

# ====================== Conflict Manager ======================

class ConflictManager:
    def __init__(self, experiment_strategy, trial_num):
        self.conflicts = 0
        self.experiment_strategy = experiment_strategy 
        self.conflict_log = []
        self.current_time = 0 
        self.PRIORITY_PENALTY = -2
        self.FALLBACK_PENALTY = -8

    def set_time(self, t): self.current_time = t

    def log_conflict(self, agents, position, type):
        if not isinstance(agents, list): agents = [agents]
        self.conflict_log.append({'time': self.current_time, 'position': position, 'type': type, 'agents': [a.name for a in agents]})

    def evaluate_offer_utility(self, offer, agent):
        cost_of_yielding = 0.5
        benefit = 0.8 if offer['content'].get('compensation') == 'immediate_energy_boost' else 0.4
        utility = (benefit - cost_of_yielding) * (1 - agent.energy/600) + 0.5
        utility += random.uniform(-0.1, 0.1)
        return max(0.0, min(1.0, utility))

    def resolve_conflict_heuristic(self, contenders):
        contenders.sort(key=lambda a: (a.score, a.energy), reverse=True)
        return contenders[0] 

    def alternating_offer_negotiate(self, contenders, pos):
        if len(contenders) != 2: return self.resolve_conflict_heuristic(contenders)
        agent1, agent2 = contenders[0], contenders[1]
        if agent1.score < agent2.score: proposer, respondent = agent1, agent2
        else: proposer, respondent = agent2, agent1
        for round in range(3): 
            offer = proposer.make_alternating_offer(respondent, pos, round)
            utility = self.evaluate_offer_utility(offer, respondent)
            if utility > 0.5:
                return (proposer if offer['content']['action'] == 'I_GO_FIRST' else respondent)
            proposer, respondent = respondent, proposer 
        return None 

    def negotiate(self, agents, proposed_positions, maze):
        reverse_map = defaultdict(list)
        resolved_positions = {}
        for agent, pos in proposed_positions.items(): reverse_map[pos].append(agent)

        for pos, contenders in reverse_map.items():
            if len(contenders) == 1:
                agent = contenders[0]
                if pos in maze.shared_corridors:
                    if not maze.lock(pos, agent.name):
                        locker = maze.shared_route_status.get(pos)
                        if locker != agent.name:
                            if self.experiment_strategy == 'Q_LEARNING':
                                resolved_positions[agent] = agent.pos 
                                self.conflicts += 1
                                self.log_conflict(agent, pos, "Lock_Collision")
                            else:
                                locking_agent = next((a for a in agents if a.name == locker), None)
                                if locking_agent:
                                    self.conflicts += 1
                                    winner = self.handle_strategy([agent, locking_agent], pos)
                                    if winner == agent:
                                        maze.unlock(pos); resolved_positions[locking_agent] = locking_agent.pos 
                                        maze.lock(pos, agent.name); resolved_positions[agent] = pos
                                    else: resolved_positions[agent] = agent.pos
                                else: resolved_positions[agent] = agent.pos
                        else: resolved_positions[agent] = pos
                    else: agent.current_lock = pos; resolved_positions[agent] = pos
                else: resolved_positions[agent] = pos
            else:
                self.conflicts += 1
                self.log_conflict(contenders, pos, "Position_Collision")
                winner = self.handle_strategy(contenders, pos)
                for agent in contenders:
                    if agent == winner:
                        resolved_positions[agent] = pos
                        if pos in maze.shared_corridors: maze.lock(pos, agent.name); agent.current_lock = pos
                    else:
                        resolved_positions[agent] = agent.pos 
                        if self.experiment_strategy != 'Q_LEARNING':
                            penalty = self.PRIORITY_PENALTY if self.experiment_strategy == 'PRIORITY_BASED' else -8
                            agent.update_energy(penalty)
        return resolved_positions

    def handle_strategy(self, contenders, pos):
        if self.experiment_strategy == 'Q_LEARNING': return random.choice(contenders)
        elif self.experiment_strategy == 'PRIORITY_BASED': return self.resolve_conflict_heuristic(contenders)
        elif self.experiment_strategy == 'ALTERNATING_OFFER':
            winner = self.alternating_offer_negotiate(contenders, pos)
            if winner: return winner
            else: return self.resolve_conflict_heuristic(contenders) 
        return contenders[0]

# ====================== GUI Wrapper ======================

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
            min_dist = float('inf'); best_pos = None
            for dir in directions:
                dx, dy = {'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}[dir]
                nx, ny = x+dx, y+dy
                if 0<=nx<maze.height and 0<=ny<maze.width and (nx,ny) not in maze.walls:
                    for agent in agents:
                        if agent.active:
                            d = abs(nx-agent.pos[0])+abs(ny-agent.pos[1])
                            if d<min_dist: min_dist=d; best_pos=(nx,ny)
            if best_pos: return best_pos
        random.shuffle(directions)
        for dir in directions:
             dx, dy = {'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}[dir]
             nx, ny = x+dx, y+dy
             if 0<=nx<maze.height and 0<=ny<maze.width and (nx,ny) not in maze.walls: return (nx,ny)
        return self.pos

class PacmanSimulation:
    def __init__(self, strategy='Q_LEARNING', trial_num=1, existing_brains=None):
        self.maze = Maze()
        self.width_px = self.maze.width * CELL_SIZE + SIDEBAR_WIDTH
        self.height_px = self.maze.height * CELL_SIZE
        self.trial_num = trial_num
        
        self.screen = pygame.display.set_mode((self.width_px, self.height_px))
        pygame.display.set_caption(f"MAS Pac-Man: {strategy} (Trial {trial_num}/{NUM_TRIALS})")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)

        self.manager = ConflictManager(strategy, trial_num)
        self.strategy = strategy
        
        # Load Existing Brains
        brain_A = existing_brains['A'] if existing_brains else None
        brain_B = existing_brains['B'] if existing_brains else None
        brain_C = existing_brains['C'] if existing_brains else None
        
        self.agents = [
            Agent("PacmanA", "A", (0, 0), YELLOW, existing_q_table=brain_A),
            Agent("PacmanB", "B", (self.maze.height - 1, 0), CYAN, existing_q_table=brain_B),
            Agent("PacmanC", "C", (0, self.maze.width - 1), MAGENTA, existing_q_table=brain_C)
        ]
        
        self.ghosts = []
        while len(self.ghosts) < 2:
            gx = random.randint(1, self.maze.height - 2)
            gy = random.randint(1, self.maze.width - 2)
            pos = (gx, gy)
            if pos not in self.maze.walls and pos != (0,0) and pos != (self.maze.height-1, 0) and pos != (0, self.maze.width-1):
                name = f"Ghost{len(self.ghosts)+1}"
                self.ghosts.append(Ghost(name, "G", pos))

        self.time_step = 0
        self.max_turns = MAX_TURNS
        self.running = True
        self.energy_loss_ghost = -50 
        self.outcome = "RUNNING"

    def draw(self):
        self.screen.fill(BLACK)
        for r in range(self.maze.height):
            for c in range(self.maze.width):
                rect = pygame.Rect(c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                char = self.maze.grid[r][c]
                if char == '#': pygame.draw.rect(self.screen, BLUE, rect)
                elif char == '=': pygame.draw.rect(self.screen, GREY, rect)
                if (r, c) in self.maze.pellets:
                    center = (c * CELL_SIZE + CELL_SIZE // 2, r * CELL_SIZE + CELL_SIZE // 2)
                    pygame.draw.circle(self.screen, WHITE, center, 4)

        for agent in self.agents:
            if agent.active:
                center = (agent.pos[1] * CELL_SIZE + CELL_SIZE // 2, agent.pos[0] * CELL_SIZE + CELL_SIZE // 2)
                pygame.draw.circle(self.screen, agent.color, center, CELL_SIZE // 2 - 2)
        for ghost in self.ghosts:
            center = (ghost.pos[1] * CELL_SIZE + CELL_SIZE // 2, ghost.pos[0] * CELL_SIZE + CELL_SIZE // 2)
            pygame.draw.circle(self.screen, RED, center, CELL_SIZE // 2 - 2)

        ui_x = self.maze.width * CELL_SIZE + 10
        y_offset = 10
        self.screen.blit(self.font.render(f"Trial: {self.trial_num} / {NUM_TRIALS}", True, GREEN), (ui_x, y_offset))
        y_offset += 20
        self.screen.blit(self.font.render(f"Time: {self.time_step}", True, GREEN), (ui_x, y_offset))
        y_offset += 20
        self.screen.blit(self.font.render(f"Mode: {self.strategy}", True, GREEN), (ui_x, y_offset))
        y_offset += 30
        
        all_waits = [a.wait_time for a in self.agents]
        avg_wait = statistics.mean(all_waits) if all_waits else 0
        fairness_index = 1.0
        if sum(all_waits) > 0:
            num = sum(all_waits) ** 2
            den = len(all_waits) * sum(w ** 2 for w in all_waits)
            fairness_index = num / den if den != 0 else 0
            
        self.screen.blit(self.font.render(f"Conflicts: {self.manager.conflicts}", True, WHITE), (ui_x, y_offset))
        y_offset += 20
        self.screen.blit(self.font.render(f"Avg Wait: {avg_wait:.1f}", True, WHITE), (ui_x, y_offset))
        y_offset += 20
        self.screen.blit(self.font.render(f"Jain's Idx: {fairness_index:.2f}", True, WHITE), (ui_x, y_offset))
        y_offset += 30
        
        for agent in self.agents:
            color = agent.color if agent.active else GREY
            q_size = len(agent.q_table)
            stats = f"{agent.symbol}: Sc:{agent.score} Q:{q_size} Rwd:{agent.last_reward_val}"
            self.screen.blit(self.font.render(stats, True, color), (ui_x, y_offset))
            y_offset += 20
            if agent.active:
                pct = max(0, agent.energy / agent.max_energy)
                pygame.draw.rect(self.screen, (50, 50, 50), (ui_x, y_offset, 200, 8))
                pygame.draw.rect(self.screen, GREEN if pct > 0.5 else RED, (ui_x, y_offset, int(200 * pct), 8))
            y_offset += 25
        pygame.display.flip()

    def update(self):
        if not any(a.active for a in self.agents):
            self.running = False
            self.outcome = "LOSS"
            return
        if len(self.maze.pellets) == 0:
            self.running = False
            self.outcome = "WIN"
            return
        if self.time_step >= self.max_turns:
            self.running = False
            self.outcome = "TIMEOUT"
            return

        self.manager.set_time(self.time_step)

        for ghost in self.ghosts:
            if ghost.active: ghost.pos = ghost.move(self.maze, self.agents)

        proposed_positions = {}
        agent_intents = {} 

        for agent in self.agents:
            if not agent.active: continue
            
            if self.strategy == 'Q_LEARNING':
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
                    intended_pos = agent.pos; hit_wall = True
                else: intended_pos = target
                agent_intents[agent] = {'state': state, 'action': action, 'hit_wall': hit_wall}
                proposed_positions[agent] = intended_pos
            
            else:
                target = agent.move_heuristic(self.maze, self.ghosts)
                proposed_positions[agent] = target
                agent_intents[agent] = {'state': None, 'action': None, 'hit_wall': False} 

        resolved = self.manager.negotiate(self.agents, proposed_positions, self.maze)
        
        for agent, actual_pos in resolved.items():
            if not agent.active: continue
            
            if actual_pos == agent.pos: agent.wait_time += 1
            if agent.current_lock and actual_pos != agent.current_lock:
                self.maze.unlock(agent.current_lock)
                agent.current_lock = None
            
            if actual_pos != agent.pos:
                agent.recent_history.append(actual_pos)
            
            prev_pos = agent.pos
            agent.pos = actual_pos
            
            reward = -1 
            hit_ghost = False
            for g in self.ghosts:
                if g.active and g.pos == agent.pos: hit_ghost = True; break
            
            if hit_ghost:
                reward = -100
                agent.update_energy(self.energy_loss_ghost)
            else:
                if agent_intents[agent]['hit_wall']: reward -= 5 
                
                if actual_pos in self.maze.pellets:
                    reward += 20
                    agent.score += 10
                    agent.update_energy(15) 
                    self.maze.pellets.remove(actual_pos)
                
                elif actual_pos != prev_pos and self.strategy == 'Q_LEARNING':
                    targets = list(self.maze.pellets) if self.maze.pellets else list(self.maze.shared_corridors)
                    current_min_dist = float('inf')
                    for t in targets:
                        d = abs(t[0] - agent.pos[0]) + abs(t[1] - agent.pos[1])
                        if d < current_min_dist: current_min_dist = d
                    
                    if current_min_dist < agent.prev_dist_to_target:
                        reward += 1 
                    else:
                        reward -= 1.5 
                        
                    agent.prev_dist_to_target = current_min_dist
                
                if self.strategy == 'Q_LEARNING':
                    if actual_pos in agent.recent_history and len(agent.recent_history) > 2:
                        reward -= 10 

                if proposed_positions[agent] != actual_pos and not agent_intents[agent]['hit_wall']: reward -= 4 

                min_g_dist = float('inf')
                for g in self.ghosts:
                    d = abs(g.pos[0]-agent.pos[0]) + abs(g.pos[1]-agent.pos[1])
                    if d < min_g_dist: min_g_dist = d
                if min_g_dist <= 2: reward -= 5 

            if self.strategy == 'Q_LEARNING':
                new_state = agent.get_state(self.maze, self.ghosts)
                agent.learn(agent_intents[agent]['state'], agent_intents[agent]['action'], reward, new_state)
                if agent.epsilon > 0.01: agent.epsilon *= 0.999
            
            agent.last_reward_val = reward 
            agent.update_energy(-0.5) 

        self.time_step += 1

    def get_trial_data(self):
        all_waits = [a.wait_time for a in self.agents]
        n = len(all_waits)
        fairness_index = 0
        if sum(all_waits) > 0:
            num = sum(all_waits) ** 2
            den = n * sum(w ** 2 for w in all_waits)
            fairness_index = num / den if den != 0 else 0
        
        avg_wait = statistics.mean(all_waits) if all_waits else 0
        
        return {
            'Strategy': self.strategy,
            'Trial': self.trial_num,
            'Outcome': self.outcome,
            'Turns': self.time_step,
            'Conflicts': self.manager.conflicts,
            'Avg_Wait_Time': avg_wait,
            'Fairness_Index': fairness_index,
            'PacmanA_Score': self.agents[0].score,
            'PacmanB_Score': self.agents[1].score,
            'PacmanC_Score': self.agents[2].score
        }

if __name__ == "__main__":
    random.seed(time.time())
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    font = pygame.font.SysFont("Arial", 24)
    
    selected_strategy = None
    while not selected_strategy:
        screen.fill(BLACK)
        screen.blit(font.render("Select Strategy for N Trials:", True, WHITE), (50, 50))
        screen.blit(font.render("1. Priority-Based", True, YELLOW), (50, 100))
        screen.blit(font.render("2. Alternating Offers", True, CYAN), (50, 150))
        screen.blit(font.render("3. Q-Learning (RL)", True, MAGENTA), (50, 200))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: selected_strategy = 'PRIORITY_BASED'
                elif event.key == pygame.K_2: selected_strategy = 'ALTERNATING_OFFER'
                elif event.key == pygame.K_3: selected_strategy = 'Q_LEARNING'

    print(f"Starting 30-Trial Run for: {selected_strategy}")
    
    window_open = True
    paused = False
    overlay_font = pygame.font.SysFont("Arial", 40, bold=True)
    
    master_log = []
    
    # --- PERSISTENT BRAINS FOR RL ---
    saved_brains = {
        'A': defaultdict(float),
        'B': defaultdict(float),
        'C': defaultdict(float)
    }
    
    for trial_idx in range(1, NUM_TRIALS + 1):
        if not window_open: break

        sim = PacmanSimulation(selected_strategy, trial_num=trial_idx, existing_brains=saved_brains)
        
        # Trial Loop
        while sim.running and window_open:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: window_open = False  
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: paused = not paused

            if not paused:
                sim.update()

            sim.draw()
            
            if paused:
                text = overlay_font.render("PAUSED", True, YELLOW)
                sim.screen.blit(text, (sim.width_px//2-50, sim.height_px//2))

            pygame.display.flip() 
            sim.clock.tick(FPS)

        # --- TRIAL ENDED ---
        # 1. Capture Data
        trial_data = sim.get_trial_data()
        master_log.append(trial_data)
        print(f"Trial {trial_idx} Finished: {trial_data['Outcome']} in {trial_data['Turns']} turns.")

        # 2. Save Brains (RL Only)
        if selected_strategy == 'Q_LEARNING':
            saved_brains['A'] = sim.agents[0].q_table
            saved_brains['B'] = sim.agents[1].q_table
            saved_brains['C'] = sim.agents[2].q_table
        
        # 3. Short Pause before next trial
        if window_open and trial_idx < NUM_TRIALS:
            time.sleep(0.5) # 0.5s delay between trials

    # --- ALL 30 TRIALS DONE ---
    print("\n--- BATCH COMPLETE. SAVING MASTER LOG ---")
    
    # Save Master Log to CSV
    csv_filename = f"MITELEC101_Master_Log_{selected_strategy}.csv"
    if master_log:
        keys = master_log[0].keys()
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(master_log)
        print(f"Saved complete log to: {csv_filename}")

    pygame.quit()