import pygame
import numpy as np
import random

CELL_SIZE = 50
GRID_COLS = 11
GRID_ROWS = 11
GRID_WIDTH = GRID_COLS * CELL_SIZE
GRID_HEIGHT = GRID_ROWS * CELL_SIZE

GOAL = 1
FAIL = 2
WALL = 3

ACTIONS = ['up', 'down', 'left', 'right']
ACTION_TO_DELTA = {
    'up': (0, -1),
    'down': (0, 1),
    'left': (-1, 0),
    'right': (1, 0)
}

WHITE = (200, 200, 200)
GREEN = (0, 128, 0)
LIGHT_GRAY = (220, 220, 220)

class Grid:
    def __init__(self):
        #het speelveld/grid
        self.grid = [
            [3,3,3,3,3,3,3,3,3,3,3],
            [3,2,3,0,0,0,0,0,0,0,3],
            [3,0,3,1,3,3,3,2,0,0,3],
            [3,0,3,3,3,0,3,0,0,0,3],
            [3,0,0,0,0,0,3,0,0,0,3],
            [3,0,3,0,0,0,3,0,0,0,3],
            [3,0,3,0,0,3,3,0,0,0,3],
            [3,3,3,3,0,0,0,0,0,0,3],
            [3,0,0,0,0,3,3,3,3,0,3],
            [3,2,0,0,0,3,2,0,0,0,3],
            [3,3,3,3,3,3,3,3,3,3,3],
        ]
        self.rows = GRID_ROWS
        self.cols = GRID_COLS
        self.cell_size = CELL_SIZE

        #alle kleuren en afbeeldingen
        self.WALL_ORIG = pygame.image.load("src/PyGameRL/wall.png")
        self.WALL_TILE = pygame.transform.smoothscale(self.WALL_ORIG, (CELL_SIZE, CELL_SIZE))
        self.CHEESEWHEEL_ORIG = pygame.image.load("src/PyGameRL/cheesewheel.png")
        self.CHEESEWHEEL_TILE = pygame.transform.smoothscale(self.CHEESEWHEEL_ORIG, (CELL_SIZE, CELL_SIZE))
        self.MOUSETRAP_ORIG = pygame.image.load("src/PyGameRL/mousetrap.png")
        self.MOUSETRAP_TILE = pygame.transform.smoothscale(self.MOUSETRAP_ORIG, (CELL_SIZE, CELL_SIZE))

    def initialize_grid(self, surface, top_left):
        for y, row in enumerate(self.grid):
            for x, tile in enumerate(row):
                rect = pygame.Rect(
                    top_left[0] + x * self.cell_size,
                    top_left[1] + y * self.cell_size,
                    self.cell_size, self.cell_size
                )
                if tile == GOAL:
                    cheesewheel_rect = self.CHEESEWHEEL_TILE.get_rect(center=(rect.centerx, rect.centery))
                    surface.blit(self.CHEESEWHEEL_TILE, cheesewheel_rect)
                elif tile == FAIL:
                    mousetrap_rect = self.MOUSETRAP_TILE.get_rect(center=(rect.centerx, rect.centery))
                    surface.blit(self.MOUSETRAP_TILE, mousetrap_rect)
                elif tile == WALL:
                    wall_rect = self.WALL_TILE.get_rect(center=(rect.centerx, rect.centery))
                    surface.blit(self.WALL_TILE, wall_rect)

class Agent:
    def __init__(self):
        self.Q = np.zeros((GRID_ROWS, GRID_COLS, len(ACTIONS)))
        self.LEARNING_RATE = 0.1      # alpha, hoe snel de agent leert
        self.DISCOUNT_FACTOR = 0.9      # gamma, helpt bij het bepalen van de waarde van toekomstige beloningen
        self.EXPLORATION_RATE = 0.2    # epsilon, de kans dat de agent een willekeurige actie kiest in plaats van de mogelijk beste actie
        self.grid_x = GRID_COLS // 2
        self.grid_y = GRID_ROWS // 2
        self.current_episode = 0
        self.evaluation_successes = 0

    #reset van alles
    def reset(self):
        self.grid_x = GRID_COLS // 2
        self.grid_y = GRID_ROWS // 2
        self.Q = np.zeros((GRID_ROWS, GRID_COLS, len(ACTIONS)))
        self.current_episode = 0
        self.evaluation_successes = 0
        self.EXPLORATION_RATE = 0.2

    #het soms random kiezen van een actie
    def choose_action(self, state):
        if random.random() < self.EXPLORATION_RATE:
            return random.randint(0, len(ACTIONS)-1)
        else:
            return np.argmax(self.Q[state[1], state[0]])

    #het bereken van de positieve of negatieve reward en of het een doel of faal heeft bereikt
    def calculate_reward(self, tile):
        reward = -0.1  #negatieve rewards voor elke stap
        done = False
        wall = False
        if tile == WALL:
            reward = -0.5 #negatieve reward voor het proberen een muur in te gaan
            wall = True
        elif tile == FAIL:
            reward = -10 #hoge negatieve reward voor het falen
            done = True
        elif tile == GOAL:
            reward = 100 #hoge positieve reward voor het bereiken van het doel
            done = True
        return reward, done, wall

    #het verplaatsen van de againt op het grid en hierbij wordt rekenning gehouden met muren en het doel/faalpunt
    def move_agent(self, grid, state, action_idx, evaluation_successes=0):
        dx, dy = ACTION_TO_DELTA[ACTIONS[action_idx]]
        new_x = max(0, min(GRID_COLS-1, state[0] + dx))
        new_y = max(0, min(GRID_ROWS-1, state[1] + dy))
        tile = grid[new_y][new_x]
        wall = False
        reward, done, wall = self.calculate_reward(tile)

        if wall:
            new_x, new_y = state

        #positie op het grid bijwerken
        self.grid_x, self.grid_y = new_x, new_y

        return (new_x, new_y), reward, done

    #update de Q-waarde
    def update_q_value(self, state, action, reward, next_state):
        best_next = np.max(self.Q[next_state[1], next_state[0]])
        self.Q[state[1], state[0], action] += self.LEARNING_RATE * (reward + self.DISCOUNT_FACTOR * best_next - self.Q[state[1], state[0], action])

    #Toon de Q-waarde van elke actie in elke plek/tile van het grid
    def show_q_value(self):
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                q_values = self.Q[y, x]
                max_q = np.max(q_values)
                if max_q > 0:
                    action_idx = np.argmax(q_values)
                    action = ACTIONS[action_idx]
                    print(f"State ({x}, {y}) - Best Action: {action} with Q-value: {max_q:.2f}")

class InfoPanel:
    def __init__(self, grid_top_left):
        self.grid_top_left = grid_top_left

    #tekent het informatiepaneel met de huidige episode, exploratie rate en evaluatie successen en uitleg over controls
    def draw_info_panel(self, screen, episode = 0, success=0, exploration_rate=0.2):
        font = pygame.font.SysFont(None, 32)
        info_x = self.grid_top_left[0] + GRID_WIDTH + 40
        info_y = self.grid_top_left[1]
        text1 = font.render(f"Episodes: {episode}", True, (0, 0, 0))
        text3 = font.render(f"Evaluation successes: {success}/10", True, (0, 0, 0))
        text2 = font.render(f"Exploration rate: {exploration_rate:.2f}", True, (0, 0, 0))
        modes_text = font.render(f"Modes: 1: Ai | 2: Manual", True, (0, 0, 0))
        reset_text = font.render(f"Reset: press 'r'", True, (0, 0, 0))
        control_text = font.render(f"Move: w,a,s,d", True, (0, 0, 0))
        screen.blit(text1, (info_x, info_y))
        screen.blit(text2, (info_x, info_y + 40))
        screen.blit(text3, (info_x, info_y + 80))
        screen.blit(modes_text, (info_x, info_y + 120))
        screen.blit(reset_text, (info_x, info_y + 160))
        screen.blit(control_text, (info_x, info_y + 200))

    #het teken van de Q-waarden in het grid met positieve en negatieve kleuren
    def draw_q_values(self, screen, Q):
        font = pygame.font.SysFont(None, 16)
        for y in range(GRID_ROWS):
            for x in range(GRID_COLS):
                q = Q[y, x]
                cell_x = self.grid_top_left[0] + x * CELL_SIZE
                cell_y = self.grid_top_left[1] + y * CELL_SIZE
                #Omhoog
                up_text = font.render(f"{q[0]:.1f}", True,(0, 100, 0) if q[0] > 0 else (255, 0, 0) if q[0] < 0 else (0, 0, 255))
                screen.blit(up_text, (cell_x + CELL_SIZE//2 - 8, cell_y + 2))
                #Omlaag
                down_text = font.render(f"{q[1]:.1f}", True, (0, 100, 0) if q[1] > 0 else (255, 0, 0) if q[1] < 0 else (0, 0, 255))
                screen.blit(down_text, (cell_x + CELL_SIZE//2 - 8, cell_y + CELL_SIZE - 16))
                #Naar links
                left_text = font.render(f"{q[2]:.1f}", True, (0, 100, 0) if q[2] > 0 else (255, 0, 0) if q[2] < 0 else (0, 0, 255))
                screen.blit(left_text, (cell_x + 2, cell_y + CELL_SIZE//2 - 8))
                #Naar rechts
                right_text = font.render(f"{q[3]:.1f}", True, (0, 100, 0) if q[3] > 0 else (255, 0, 0) if q[3] < 0 else (0, 0, 255))
                screen.blit(right_text, (cell_x + CELL_SIZE - 18, cell_y + CELL_SIZE//2 - 8))

class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1280, 720))
        self.clock = pygame.time.Clock()
        self.running = True
        self.grid = Grid()
        self.agent = Agent()
        self.GRID_TOP_LEFT = (
            (self.screen.get_width() - GRID_WIDTH) // 2,
            (self.screen.get_height() - GRID_HEIGHT) // 2
        )
        self.info_panel = InfoPanel(self.GRID_TOP_LEFT)
        self.PLAYER_ORIG = pygame.image.load("src/PyGameRL/mouse.png")
        self.PLAYER = pygame.transform.smoothscale(self.PLAYER_ORIG, (CELL_SIZE, CELL_SIZE))
        self.player_mode = False

    def reset(self):
        self.agent.reset()

    def render(self):
        self.screen.fill(GREEN)
        pygame.draw.rect(
            self.screen,
            LIGHT_GRAY,
            (self.GRID_TOP_LEFT[0], self.GRID_TOP_LEFT[1], GRID_WIDTH, GRID_HEIGHT)
        )
        self.grid.initialize_grid(self.screen, self.GRID_TOP_LEFT)
        self.info_panel.draw_q_values(self.screen, self.agent.Q)
        self.info_panel.draw_info_panel(self.screen, self.agent.current_episode, self.agent.evaluation_successes, self.agent.EXPLORATION_RATE)
        player_rect = self.PLAYER.get_rect(center=(
            self.GRID_TOP_LEFT[0] + self.agent.grid_x * CELL_SIZE + CELL_SIZE // 2,
            self.GRID_TOP_LEFT[1] + self.agent.grid_y * CELL_SIZE + CELL_SIZE // 2
        ))
        self.screen.blit(self.PLAYER, player_rect)
        pygame.display.flip()

    def train_q_agent(self, episodes=500):
        for ep in range(episodes):
            self.agent.current_episode += 1
            state = (GRID_COLS // 2, GRID_ROWS // 2)
            self.agent.grid_x, self.agent.grid_y = state
            total_reward = 0
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                action_idx = self.agent.choose_action(state)
                next_state, reward, done = self.agent.move_agent(self.grid.grid, state, action_idx)
                self.agent.update_q_value(state, action_idx, reward, next_state)
                state = next_state
                total_reward += reward
                self.render()
                pygame.time.delay(1) #delay tussen stappen in
            print(f"Episode :{ep+1} | reward: {total_reward:.2f} | state: {state}")
            self.agent.EXPLORATION_RATE = max(0.01, self.agent.EXPLORATION_RATE * 0.99)

    def evaluate_agent(self, runs=10):
        successes = 0
        self.agent.saved_exploration_rate = self.agent.EXPLORATION_RATE
        self.agent.EXPLORATION_RATE = 0.0
        for ep in range(runs):
            state = (GRID_COLS // 2, GRID_ROWS // 2)
            self.agent.grid_x, self.agent.grid_y = state
            total_reward = 0
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                action_idx = self.agent.choose_action(state)
                next_state, reward, done = self.agent.move_agent(self.grid.grid, state, action_idx)
                self.agent.update_q_value(state, action_idx, reward, next_state)
                state = next_state
                total_reward += reward
                self.render()
                pygame.time.delay(75) #delay tussen stappen in
            if self.grid.grid[state[1]][state[0]] == GOAL:
                successes += 1
            self.agent.evaluation_successes = successes
            self.info_panel.draw_info_panel(self.screen, self.agent.current_episode, self.agent.evaluation_successes, self.agent.EXPLORATION_RATE)
            pygame.display.flip()
            pygame.time.delay(200) #delay tussen evaluatie runs om te kijken naar het resultaat van de run
            print(f"Success :{successes} | reward: {total_reward:.2f} | state: {state}")
        self.agent.EXPLORATION_RATE = self.agent.saved_exploration_rate  # Reset exploration rate terug naar de waarde voor evaluatie
        print(f"Evaluation: {successes}/{runs} runs were successful (goal reached).")
        return successes

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        self.train_q_agent(100)
                        self.evaluate_agent(10)
                    if event.key == pygame.K_2:
                        self.player_mode = not self.player_mode
                    if event.key == pygame.K_r:
                        self.reset()
                    if self.player_mode:
                        new_x, new_y = self.agent.grid_x, self.agent.grid_y
                        if event.key == pygame.K_w:
                            new_y = max(0, self.agent.grid_y - 1)
                            action_idx = 0
                        elif event.key == pygame.K_s:
                            new_y = min(GRID_ROWS - 1, self.agent.grid_y + 1)
                            action_idx = 1
                        elif event.key == pygame.K_a:
                            new_x = max(0, self.agent.grid_x - 1)
                            action_idx = 2
                        elif event.key == pygame.K_d:
                            new_x = min(GRID_COLS - 1, self.agent.grid_x + 1)
                            action_idx = 3
                        else:
                            action_idx = None

                        if action_idx is not None:
                            prev_state = (self.agent.grid_x, self.agent.grid_y)

                            if self.grid.grid[new_y][new_x] != WALL:
                                self.agent.grid_x, self.agent.grid_y = new_x, new_y
                                tile = self.grid.grid[self.agent.grid_y][self.agent.grid_x]
                                reward, done, wall = self.agent.calculate_reward(tile)
                                next_state = (self.agent.grid_x, self.agent.grid_y)
                                self.agent.update_q_value(prev_state, action_idx, reward, next_state)
                            else:
                                reward, done, wall = self.agent.calculate_reward(WALL)
                                next_state = prev_state
                                self.agent.update_q_value(prev_state, action_idx, reward, next_state)

                            if self.grid.grid[new_y][new_x] == FAIL:
                                self.agent.grid_x, self.agent.grid_y = GRID_COLS // 2, GRID_ROWS // 2
                                print("You hit a mousetrap! Resetting position.")
                            if self.grid.grid[new_y][new_x] == GOAL:
                                self.agent.grid_x, self.agent.grid_y = GRID_COLS // 2, GRID_ROWS // 2
                                print("You reached the goal! Resetting position.")

            self.render()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()