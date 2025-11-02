import pygame
import sys
import os
from pacman_simulation import PacmanSimulation
import random
import math

class PacmanVisualizer:
    def __init__(self, simulation, cell_size=50):
        self.simulation = simulation
        self.cell_size = cell_size
        self.width = simulation.maze.width * cell_size + 300  # Extra space for metrics panel
        self.height = simulation.maze.height * cell_size
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Enhanced Pac-Man Multi-Agent Simulation")
        
        # Enhanced color scheme
        self.colors = {
            'background': (0, 0, 0),
            'wall': (0, 0, 150),
            'pellet': (255, 255, 255),
            'shared_corridor_unlocked': (100, 100, 255),    # Blue for unlocked
            'shared_corridor_locked': (255, 100, 100),      # Red for locked
            'agent_a': (255, 0, 0),      # Red
            'agent_b': (0, 255, 0),      # Green
            'agent_c': (255, 255, 0),    # Yellow
            'ghost': (255, 0, 255),      # Pink
            'text': (255, 255, 255),
            'conflict_alert': (255, 165, 0),  # Orange for conflict
            'negotiation_active': (255, 215, 0),  # Gold for negotiation
            'path_anticipation': (0, 255, 255),   # Cyan for path anticipation
            'metrics_panel': (30, 30, 30),        # Dark gray for metrics panel
            'success': (0, 255, 0),               # Green for success
            'warning': (255, 255, 0),             # Yellow for warning
            'danger': (255, 0, 0)                 # Red for danger
        }
        
        # Fonts for different text sizes
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.large_font = pygame.font.Font(None, 32)
        
        # Visualization state
        self.current_conflict = None
        self.negotiation_messages = []
        self.conflict_history = []
        self.last_negotiation_outcome = None
        
    def draw_maze(self):
        """Draw the maze grid with walls, pellets, and shared corridors with status"""
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
                    # Check shared route status (UNLOCKED or LOCKED)
                    if self.simulation.maze.is_locked((i, j)):
                        # Draw locked shared corridor in red
                        pygame.draw.rect(self.screen, self.colors['shared_corridor_locked'], rect)
                        # Add lock symbol
                        lock_text = self.small_font.render("🔒", True, (255, 255, 255))
                        lock_rect = lock_text.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2))
                        self.screen.blit(lock_text, lock_rect)
                    else:
                        # Draw unlocked shared corridor in blue
                        pygame.draw.rect(self.screen, self.colors['shared_corridor_unlocked'], rect)
                        # Add unlock symbol
                        unlock_text = self.small_font.render("🔓", True, (255, 255, 255))
                        unlock_rect = unlock_text.get_rect(center=(x + self.cell_size//2, y + self.cell_size//2))
                        self.screen.blit(unlock_text, unlock_rect)
                elif (i, j) in self.simulation.maze.pellets:
                    # Draw pellet as small circle
                    center = (x + self.cell_size // 2, y + self.cell_size // 2)
                    pygame.draw.circle(self.screen, self.colors['pellet'], center, self.cell_size // 8)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)
    
    def draw_agents(self):
        """Draw all active agents with enhanced features"""
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
                
                # Draw agent as circle with energy indicator
                pygame.draw.circle(self.screen, color, (x, y), self.cell_size // 3)
                
                # Draw energy bar above agent
                energy_width = (self.cell_size // 2) * (agent.energy / 100)
                energy_rect = pygame.Rect(x - self.cell_size//4, y - self.cell_size//2 - 5, energy_width, 3)
                pygame.draw.rect(self.screen, self.colors['success'], energy_rect)
                
                # Draw agent symbol
                text = self.font.render(agent.symbol, True, (0, 0, 0))
                text_rect = text.get_rect(center=(x, y))
                self.screen.blit(text, text_rect)
                
                # Draw path anticipation (if applicable)
                self.draw_path_anticipation(agent)
    
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
    
    def draw_path_anticipation(self, agent):
        """Draw path anticipation and potential conflicts for an agent"""
        # Get potential conflicts within sensing radius
        potential_conflicts = agent.detect_potential_conflicts(
            [a for a in self.simulation.agents if a != agent and a.active],
            self.simulation.maze
        )
        
        for conflict in potential_conflicts:
            pos = conflict['position']
            x = pos[1] * self.cell_size + self.cell_size // 2
            y = pos[0] * self.cell_size + self.cell_size // 2
            
            # Draw conflict anticipation indicator
            pygame.draw.circle(self.screen, self.colors['path_anticipation'], (x, y), self.cell_size // 6, 2)
            
            # Draw line from agent to potential conflict
            agent_x = agent.pos[1] * self.cell_size + self.cell_size // 2
            agent_y = agent.pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.line(self.screen, self.colors['path_anticipation'],
                           (agent_x, agent_y), (x, y), 1)

    def draw_conflict_detection(self):
        """Draw active conflicts and negotiation processes"""
        # Draw conflict alerts for recent conflicts
        if self.current_conflict:
            pos = self.current_conflict['position']
            x = pos[1] * self.cell_size + self.cell_size // 2
            y = pos[0] * self.cell_size + self.cell_size // 2
            
            # Draw pulsating conflict circle
            pulse = (pygame.time.get_ticks() // 200) % 2
            radius = self.cell_size // 3 + (pulse * 5)
            pygame.draw.circle(self.screen, self.colors['conflict_alert'], (x, y), radius, 3)
            
            # Draw conflict text
            conflict_text = self.font.render("CONFLICT!", True, self.colors['conflict_alert'])
            text_rect = conflict_text.get_rect(center=(x, y - self.cell_size//2))
            self.screen.blit(conflict_text, text_rect)

    def draw_negotiation_messages(self):
        """Draw negotiation messages and outcomes"""
        if self.negotiation_messages:
            message_y = self.simulation.maze.height * self.cell_size - 100
            for i, message in enumerate(self.negotiation_messages[-3:]):  # Show last 3 messages
                msg_text = self.small_font.render(message, True, self.colors['negotiation_active'])
                self.screen.blit(msg_text, (10, message_y + i * 20))

    def draw_enhanced_metrics(self, turn):
        """Draw comprehensive real-time metrics"""
        # Draw metrics panel background
        metrics_x = self.simulation.maze.width * self.cell_size + 10
        metrics_width = self.width - metrics_x - 10
        metrics_rect = pygame.Rect(metrics_x, 10, metrics_width, self.height - 20)
        pygame.draw.rect(self.screen, self.colors['metrics_panel'], metrics_rect)
        pygame.draw.rect(self.screen, self.colors['text'], metrics_rect, 2)
        
        stats_y = 20
        
        # Simulation info
        title = self.large_font.render("Enhanced MAS Metrics", True, self.colors['text'])
        self.screen.blit(title, (metrics_x + 10, stats_y))
        stats_y += 40
        
        # Turn and active agents
        turn_text = self.font.render(f"Turn: {turn}/{self.simulation.turns}", True, self.colors['text'])
        self.screen.blit(turn_text, (metrics_x + 10, stats_y))
        stats_y += 25
        
        active_count = len([a for a in self.simulation.agents if a.active])
        active_text = self.font.render(f"Active Agents: {active_count}", True, self.colors['success'])
        self.screen.blit(active_text, (metrics_x + 10, stats_y))
        stats_y += 25
        
        # Conflict metrics
        conflicts_text = self.font.render(f"Total Conflicts: {self.simulation.manager.conflicts}", True, self.colors['text'])
        self.screen.blit(conflicts_text, (metrics_x + 10, stats_y))
        stats_y += 25
        
        successful_neg_text = self.font.render(f"Successful Negotiations: {self.simulation.manager.successful_negotiations}", True, self.colors['success'])
        self.screen.blit(successful_neg_text, (metrics_x + 10, stats_y))
        stats_y += 25
        
        pre_detected_text = self.font.render(f"Pre-detected Conflicts: {self.simulation.manager.pre_detected_conflicts}", True, self.colors['path_anticipation'])
        self.screen.blit(pre_detected_text, (metrics_x + 10, stats_y))
        stats_y += 25
        
        # Negotiation strategy distribution
        stats_y += 10
        strategy_title = self.font.render("Negotiation Strategies:", True, self.colors['text'])
        self.screen.blit(strategy_title, (metrics_x + 10, stats_y))
        stats_y += 25
        
        priority_text = self.small_font.render(f"Priority-based: {self.simulation.manager.priority_based_resolutions}", True, self.colors['text'])
        self.screen.blit(priority_text, (metrics_x + 20, stats_y))
        stats_y += 20
        
        alternating_text = self.small_font.render(f"Alternating Offers: {self.simulation.manager.alternating_offer_resolutions}", True, self.colors['text'])
        self.screen.blit(alternating_text, (metrics_x + 20, stats_y))
        stats_y += 20
        
        multi_issue_text = self.small_font.render(f"Multi-issue: {self.simulation.manager.multi_issue_negotiations}", True, self.colors['text'])
        self.screen.blit(multi_issue_text, (metrics_x + 20, stats_y))
        stats_y += 30
        
        # Agent detailed stats
        stats_y += 10
        agent_title = self.font.render("Agent Details:", True, self.colors['text'])
        self.screen.blit(agent_title, (metrics_x + 10, stats_y))
        stats_y += 25
        
        for agent in self.simulation.agents:
            status_color = self.colors['success'] if agent.active else self.colors['danger']
            status = "ACTIVE" if agent.active else "DROPPED"
            agent_text = self.small_font.render(
                f"{agent.name}: Score={agent.score}, Energy={agent.energy:.1f}, Wait={agent.wait_time}",
                True, status_color
            )
            self.screen.blit(agent_text, (metrics_x + 20, stats_y))
            stats_y += 18
            
            # Show negotiation history count
            history_text = self.small_font.render(
                f"  Negotiations: {len(agent.negotiation_history)}",
                True, self.colors['text']
            )
            self.screen.blit(history_text, (metrics_x + 30, stats_y))
            stats_y += 16
        
        # Last negotiation outcome
        if self.last_negotiation_outcome:
            stats_y += 10
            outcome_title = self.font.render("Last Negotiation:", True, self.colors['negotiation_active'])
            self.screen.blit(outcome_title, (metrics_x + 10, stats_y))
            stats_y += 25
            
            outcome_text = self.small_font.render(self.last_negotiation_outcome, True, self.colors['text'])
            self.screen.blit(outcome_text, (metrics_x + 10, stats_y))
    
    def handle_events(self):
        """Handle Pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    # Pause/unpause simulation
                    return "PAUSE"
        return True

    def update_visualization_state(self, active_agents, turn):
        """Update visualization state based on simulation events"""
        # Check for new conflicts in the conflict log
        if self.simulation.manager.conflict_log:
            latest_conflict = self.simulation.manager.conflict_log[-1]
            # Only update if it's a new conflict
            if not self.current_conflict or self.current_conflict != latest_conflict:
                self.current_conflict = latest_conflict
                conflict_msg = f"CONFLICT: {', '.join(latest_conflict['agents_involved'])} at {latest_conflict['position']}"
                self.negotiation_messages.append(conflict_msg)
        
        # Check for new negotiation outcomes
        if self.simulation.manager.negotiation_log:
            latest_negotiation = self.simulation.manager.negotiation_log[-1]
            outcome_msg = f"NEGOTIATION: {latest_negotiation['strategy']} - {latest_negotiation['resolution']}"
            if outcome_msg not in self.negotiation_messages:
                self.negotiation_messages.append(outcome_msg)
                self.last_negotiation_outcome = f"{latest_negotiation['strategy']}: {latest_negotiation['winner']} won"
        
        # Keep only recent messages (last 10)
        if len(self.negotiation_messages) > 10:
            self.negotiation_messages = self.negotiation_messages[-10:]

    def draw_stats(self, turn):
        """Draw basic stats (legacy method for compatibility)"""
        # This is kept for compatibility, but we use enhanced metrics now
        pass
    
    def run_visualization(self):
        """Run the simulation with enhanced visualization"""
        clock = pygame.time.Clock()
        running = True
        paused = False
        
        token_holder = 0
        active_agents = self.simulation.agents.copy()
        turn = 0
        
        while running and turn < self.simulation.turns:
            if not paused:
                turn += 1
                
                # Handle events
                event_result = self.handle_events()
                if event_result == False:
                    break
                elif event_result == "PAUSE":
                    paused = not paused
                    continue
                
                # Run one simulation step using the enhanced simulation logic
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

                # Use enhanced negotiation with maze parameter
                resolved = self.simulation.manager.negotiate(active_agents, proposed_positions, self.simulation.maze)
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

                # Update visualization state with new conflicts and negotiations
                self.update_visualization_state(active_agents, turn)
                
                # Token passing: next agent gets the token for shared corridor
                token_holder = (token_holder + 1) % len(active_agents)

            # Update visualization (always draw, even when paused)
            self.draw_maze()
            self.draw_agents()
            self.draw_ghosts()
            self.draw_conflict_detection()
            self.draw_negotiation_messages()
            self.draw_enhanced_metrics(turn)
            
            # Show pause indicator if paused
            if paused:
                pause_text = self.large_font.render("PAUSED - Press SPACE to continue", True, self.colors['warning'])
                text_rect = pause_text.get_rect(center=(self.width//2, self.height//2))
                self.screen.blit(pause_text, text_rect)
            
            pygame.display.flip()
            clock.tick(2)  # 2 FPS for better visibility
        
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