import pygame
import sys
import os
from pacman_simulation import PacmanSimulation
import random

class PacmanVisualizer:
    def __init__(self, simulation, cell_size=50):
        self.simulation = simulation
        self.cell_size = cell_size
        self.width = simulation.maze.width * cell_size
        self.height = simulation.maze.height * cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Pac-Man Multi-Agent Simulation")
        
        # Colors
        self.colors = {
            'background': (0, 0, 0),
            'wall': (0, 0, 150),
            'pellet': (255, 255, 255),
            'shared_corridor': (100, 100, 255),
            'agent_a': (255, 0, 0),      # Red
            'agent_b': (0, 255, 0),      # Green  
            'agent_c': (255, 255, 0),    # Yellow
            'ghost': (255, 0, 255),      # Pink
            'text': (255, 255, 255)
        }
        
        # Font for displaying scores
        self.font = pygame.font.Font(None, 24)
        
    def draw_maze(self):
        """Draw the maze grid with walls, pellets, and shared corridors"""
        self.screen.fill(self.colors['background'])
        
        for i in range(self.simulation.maze.height):
            for j in range(self.simulation.maze.width):
                x = j * self.cell_size
                y = i * self.cell_size
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                
                # Draw different cell types
                if (i, j) in self.simulation.maze.walls:
                    pygame.draw.rect(self.screen, self.colors['wall'], rect)
                elif (i, j) in self.simulation.maze.shared_corridors:
                    pygame.draw.rect(self.screen, self.colors['shared_corridor'], rect)
                elif (i, j) in self.simulation.maze.pellets:
                    # Draw pellet as small circle
                    center = (x + self.cell_size // 2, y + self.cell_size // 2)
                    pygame.draw.circle(self.screen, self.colors['pellet'], center, self.cell_size // 8)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
    
    def draw_agents(self):
        """Draw all active agents"""
        for agent in self.simulation.agents:
            if agent.active:
                x = agent.pos[1] * self.cell_size + self.cell_size // 2
                y = agent.pos[0] * self.cell_size + self.cell_size // 2
                
                # Different colors for different agents
                if agent.symbol == 'A':
                    color = self.colors['agent_a']
                elif agent.symbol == 'B':
                    color = self.colors['agent_b']
                else:
                    color = self.colors['agent_c']
                
                # Draw agent as circle
                pygame.draw.circle(self.screen, color, (x, y), self.cell_size // 3)
                
                # Draw agent symbol
                text = self.font.render(agent.symbol, True, (0, 0, 0))
                text_rect = text.get_rect(center=(x, y))
                self.screen.blit(text, text_rect)
    
    def draw_ghosts(self):
        """Draw all ghosts"""
        for ghost in self.simulation.ghosts:
            if ghost.active:
                x = ghost.pos[1] * self.cell_size + self.cell_size // 2
                y = ghost.pos[0] * self.cell_size + self.cell_size // 2
                
                # Draw ghost as larger circle
                pygame.draw.circle(self.screen, self.colors['ghost'], (x, y), self.cell_size // 2.5)
                
                # Draw ghost symbol
                text = self.font.render(ghost.symbol, True, (0, 0, 0))
                text_rect = text.get_rect(center=(x, y))
                self.screen.blit(text, text_rect)
    
    def draw_stats(self, turn):
        """Draw current turn and agent statistics"""
        stats_y = 10
        
        # Draw turn number
        turn_text = self.font.render(f"Turn: {turn}", True, self.colors['text'])
        self.screen.blit(turn_text, (10, stats_y))
        stats_y += 30
        
        # Draw agent stats
        for agent in self.simulation.agents:
            status = "ACTIVE" if agent.active else "DROPPED"
            text = f"{agent.name}: Score={agent.score}, Energy={agent.energy:.1f}"
            stat_text = self.font.render(text, True, self.colors['text'])
            self.screen.blit(stat_text, (10, stats_y))
            stats_y += 25
    
    def handle_events(self):
        """Handle Pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def run_visualization(self):
        """Run the simulation with visualization"""
        clock = pygame.time.Clock()
        running = True
        
        token_holder = 0
        active_agents = self.simulation.agents.copy()
        turn = 0
        
        while running and turn < self.simulation.turns:
            turn += 1
            
            # Handle events
            if not self.handle_events():
                break
            
            # Run one simulation step
            # Check win/loss conditions
            if len(self.simulation.maze.pellets) == 0:
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
            for ghost in self.simulation.ghosts:
                if ghost.active:
                    new_pos = ghost.move(self.simulation.maze, active_agents)
                    ghost.pos = new_pos
                    # Check if ghost caught any agent
                    for agent in active_agents:
                        if agent.pos == ghost.pos and agent.active:
                            agent.update_energy(-20)
                            print(f"{agent.name} was caught by {ghost.name}! Energy reduced.")
                            if agent.energy <= 0:
                                agent.active = False
                                print(f"{agent.name} has dropped out due to ghost attack!")

            proposed_positions = {}
            for idx, agent in enumerate(active_agents):
                if not agent.active:
                    continue
                    
                # Only the token holder can propose a move in a shared corridor this turn
                if agent.pos in self.simulation.maze.shared_corridors:
                    if idx == token_holder:
                        direction = random.choice(['U', 'D', 'L', 'R'])
                        proposed_positions[agent] = agent.move(direction, self.simulation.maze)
                    else:
                        # Must wait for token
                        proposed_positions[agent] = agent.pos
                        agent.wait_time += 1
                else:
                    direction = random.choice(['U', 'D', 'L', 'R'])
                    proposed_positions[agent] = agent.move(direction, self.simulation.maze)

            resolved = self.simulation.manager.negotiate(active_agents, proposed_positions)
            for agent, pos in resolved.items():
                agent.pos = pos
                if pos in self.simulation.maze.pellets:
                    agent.score += 10
                    self.simulation.maze.pellets.remove(pos)
                    self.simulation.maze.grid[pos[0]][pos[1]] = ' '
                if pos in self.simulation.maze.shared_corridors:
                    agent.update_energy(-1)
                else:
                    agent.update_energy(-0.5)
                    
                # Check if agent should drop out
                if agent.energy <= 0:
                    agent.active = False
                    print(f"{agent.name} has dropped out due to low energy!")

            # Update visualization
            self.draw_maze()
            self.draw_agents()
            self.draw_ghosts()
            self.draw_stats(turn)
            
            pygame.display.flip()
            clock.tick(2)  # 2 FPS for better visibility
            
            # Token passing: next agent gets the token for shared corridor
            token_holder = (token_holder + 1) % len(active_agents)
        
        # Final metrics report
        self.simulation.report_metrics()
        
        # Keep window open until user closes it
        print("Simulation finished. Close the window to exit.")
        while running:
            if not self.handle_events():
                break
            pygame.time.wait(100)
        
        pygame.quit()

if __name__ == "__main__":
    # Check if Pygame is available
    try:
        sim = PacmanSimulation()
        visualizer = PacmanVisualizer(sim)
        visualizer.run_visualization()
    except ImportError:
        print("Pygame not available. Running in text mode...")
        sim = PacmanSimulation()
        sim.run()