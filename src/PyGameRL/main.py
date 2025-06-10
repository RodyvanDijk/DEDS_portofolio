#imports for the game library and numpy for RL AI
import pygame
import numpy as np
import random

CELL_SIZE = 30
GRID_COLS = 11
GRID_ROWS = 11
GRID_WIDTH = GRID_COLS * CELL_SIZE
GRID_HEIGHT = GRID_ROWS * CELL_SIZE
ai_mode = False

# BLACK = (0, 0, 0)
WALL_ORIG = pygame.image.load("src/PyGameRL/wall.png")
WALL_TILE = pygame.transform.smoothscale(WALL_ORIG, (CELL_SIZE, CELL_SIZE))
WHITE = (200, 200, 200)
GREEN = (0, 128, 0)
CHEESEWHEEL_ORIG = pygame.image.load("src/PyGameRL/cheesewheel.png")
CHEESEWHEEL_TILE = pygame.transform.smoothscale(CHEESEWHEEL_ORIG, (CELL_SIZE, CELL_SIZE))
# PLUS_GREEN = (0, 200, 0)
MOUSETRAP_ORIG = pygame.image.load("src/PyGameRL/mousetrap.png")
MOUSETRAP_TILE = pygame.transform.smoothscale(MOUSETRAP_ORIG, (CELL_SIZE, CELL_SIZE))
# RED = (200, 0, 0)
PLAYER_ORIG = pygame.image.load("src/PyGameRL/mouse.png")
PLAYER = pygame.transform.smoothscale(PLAYER_ORIG, (CELL_SIZE, CELL_SIZE))

#The different types of tiles in the grid except for 0 which is empty
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

# Q-table: (y, x, action)
Q = np.zeros((GRID_ROWS, GRID_COLS, len(ACTIONS)))

# Hyperparameters
LEARNING_RATE = 0.1      # alpha
DISCOUNT_FACTOR = 0.9      # gamma
EXPLORATION_RATE = 0.2    # epsilon

pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

GRID_TOP_LEFT = (
    (screen.get_width() - GRID_WIDTH) // 2,
    (screen.get_height() - GRID_HEIGHT) // 2
)

# 0 = empty, 1 = green, 2 = red, 3 = black
grid = [
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

def initialize_grid(surface, grid, top_left, cell_size):
    for y, row in enumerate(grid):
        for x, tile in enumerate(row):
            rect = pygame.Rect(
                top_left[0] + x * cell_size,
                top_left[1] + y * cell_size,
                cell_size, cell_size
            )
            if tile == GOAL:
                cheesewheel_rect = CHEESEWHEEL_TILE.get_rect(
                    center=(rect.centerx, rect.centery)
                )
                screen.blit(CHEESEWHEEL_TILE, cheesewheel_rect)
            elif tile == FAIL:
                mousetrap_rect = MOUSETRAP_TILE.get_rect(
                    center=(rect.centerx, rect.centery)
                )
                screen.blit(MOUSETRAP_TILE, mousetrap_rect)
            elif tile == WALL:
                wall_rect = WALL_TILE.get_rect(
                    center=(rect.centerx, rect.centery)
                )
                screen.blit(WALL_TILE, wall_rect)

# Set initial player grid position (centered)
grid_x = GRID_COLS // 2
grid_y = GRID_ROWS // 2

def choose_action(state):
    if random.random() < EXPLORATION_RATE:
        return random.randint(0, len(ACTIONS)-1)
    else:
        return np.argmax(Q[state[1], state[0]])

def step(state, action_idx):
    dx, dy = ACTION_TO_DELTA[ACTIONS[action_idx]]
    new_x = max(0, min(GRID_COLS-1, state[0] + dx))
    new_y = max(0, min(GRID_ROWS-1, state[1] + dy))
    tile = grid[new_y][new_x]
    reward = -0.01  # small penalty for each move
    done = False
    if tile == WALL:
        new_x, new_y = state  # can't move into wall
        reward = -0.05
    elif tile == FAIL:
        reward = -10
        done = True
    elif tile == GOAL:
        reward = 100
        done = True
    return (new_x, new_y), reward, done

def train_q_agent(episodes=500):
    for ep in range(episodes):
        state = (GRID_COLS // 2, GRID_ROWS // 2)
        done = False
        while not done:
            action_idx = choose_action(state)
            next_state, reward, done = step(state, action_idx)
            best_next = np.max(Q[next_state[1], next_state[0]])
            Q[state[1], state[0], action_idx] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_next - Q[state[1], state[0], action_idx])
            state = next_state
        print(f"Episode {ep+1}/{episodes} completed.")

# Train the agent before the game loop
train_q_agent(100)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                ai_mode = not ai_mode
                print("AI mode:", ai_mode)
            if not ai_mode:
                new_x, new_y = grid_x, grid_y
                if event.key == pygame.K_w:
                    new_y = max(0, grid_y - 1)
                if event.key == pygame.K_s:
                    new_y = min(GRID_ROWS - 1, grid_y + 1)
                if event.key == pygame.K_a:
                    new_x = max(0, grid_x - 1)
                if event.key == pygame.K_d:
                    new_x = min(GRID_COLS - 1, grid_x + 1)
                # Prevent moving into walls (3)
                if grid[new_y][new_x] != WALL:
                    grid_x, grid_y = new_x, new_y
                # Prevent moving into walls (2)
                if grid[new_y][new_x] == FAIL:
                    grid_x, grid_y = GRID_COLS // 2, GRID_ROWS // 2
                    print("You hit a mousetrap! Resetting position.")
                if grid[new_y][new_x] == GOAL:
                    grid_x, grid_y = GRID_COLS // 2, GRID_ROWS // 2
                    print("You reached the goal! Resetting position.")

    if ai_mode:
        state = (grid_x, grid_y)
        action_idx = choose_action(state)
        next_state, reward, done = step(state, action_idx)
        best_next = np.max(Q[next_state[1], next_state[0]])
        Q[state[1], state[0], action_idx] += LEARNING_RATE * (reward + DISCOUNT_FACTOR * best_next - Q[state[1], state[0], action_idx])
        grid_x, grid_y = next_state
        if done:
            grid_x, grid_y = GRID_COLS // 2, GRID_ROWS // 2
        
        print(f"AI Action: {ACTIONS[action_idx]}, Position: ({grid_x}, {grid_y}), Reward: {reward}")
        pygame.time.wait(100)

    screen.fill(GREEN)
    pygame.draw.rect(
        screen,
        LIGHT_GRAY := (220, 220, 220),
        (GRID_TOP_LEFT[0], GRID_TOP_LEFT[1], GRID_WIDTH, GRID_HEIGHT)
    )
    initialize_grid(screen, grid, GRID_TOP_LEFT, CELL_SIZE)

    # Draw player centered in grid cell
    player_rect = PLAYER.get_rect(center=(
        GRID_TOP_LEFT[0] + grid_x * CELL_SIZE + CELL_SIZE // 2,
        GRID_TOP_LEFT[1] + grid_y * CELL_SIZE + CELL_SIZE // 2
    ))
    screen.blit(PLAYER, player_rect)

    pygame.display.flip()
    clock.tick(60)

pygame.quit()