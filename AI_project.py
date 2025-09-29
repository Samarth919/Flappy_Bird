import os
import random
import sys
import pygame
import numpy as np
import matplotlib.pyplot as plt
from pygame.locals import *

# ------------------ Configuration ------------------ #
SW, SH = 280, 511
BASEY = SH * 0.8
FPS = 32

# ------------------ Globals ------------------ #
IMAGES = {}
Q = np.zeros((7, 21, 2), dtype=float)
WINDOW = None
Font = None
FPSCLOCK = None

# ------------------ Helper Functions ------------------ #
def load_images():
    """Load images safely and check if files exist."""
    global IMAGES
    base_path = os.path.join(os.path.dirname(__file__), 'imgs')

    def path_check(filename):
        full_path = os.path.join(base_path, filename)
        if not os.path.exists(full_path):
            print(f"Error: '{filename}' not found in {base_path}")
            sys.exit()
        return full_path

    IMAGES['bird'] = pygame.image.load(path_check('bird1.png')).convert_alpha()
    pipe_img = pygame.image.load(path_check('pipe.png')).convert_alpha()
    IMAGES['pipe'] = (pygame.transform.rotate(pipe_img, 180), pipe_img)
    IMAGES['background'] = pygame.image.load(path_check('bg.png')).convert()
    IMAGES['base'] = pygame.image.load(path_check('base.png')).convert_alpha()

def static_screen():
    """Display static start screen."""
    bird_x = SW // 5
    bird_y = (SH - IMAGES['bird'].get_height()) // 2
    base_x = 0

    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == KEYDOWN and (event.key == K_SPACE or event.key == K_UP):
                return

        WINDOW.blit(IMAGES['background'], (0, 0))
        WINDOW.blit(IMAGES['bird'], (bird_x, bird_y))
        WINDOW.blit(IMAGES['base'], (base_x, BASEY))
        WINDOW.blit(Font.render("", True, (255, 255, 255)), (SW//2, SH//2))
        WINDOW.blit(Font.render("", True, (255, 255, 255)), (10, 50))
        pygame.display.update()
        FPSCLOCK.tick(FPS)

def get_new_pipe():
    """Generate new pipe positions."""
    gap = SH // 4
    pipe_height = IMAGES['pipe'][1].get_height()
    y2 = gap + random.randrange(0, int(SH - IMAGES['base'].get_height() - 1.2*gap))
    pipe_x = SW + 300
    y1 = pipe_height - y2 + gap
    return [{'x': pipe_x, 'y': -y1}, {'x': pipe_x, 'y': y2}]

def convert(bird_x, bird_y, bttm_pipes):
    """Convert positions into discrete states for Q-learning."""
    x = min(SW, bttm_pipes[0]['x'])
    y = bttm_pipes[0]['y'] - bird_y
    if y < 0:
        y = abs(y) + 408
    return int(x / 40 - 1), int(y / 40)

def ai_play(x, y):
    """Decide whether to jump based on Q-table."""
    jump = False
    if Q[x][y][1] > Q[x][y][0]:
        jump = True
    return jump

def update_Q(x_prev, y_prev, jump, reward, x_new, y_new):
    """Update Q-table."""
    if jump:
        Q[x_prev][y_prev][1] = 0.4 * Q[x_prev][y_prev][1] + 0.6 * (reward + max(Q[x_new][y_new]))
    else:
        Q[x_prev][y_prev][0] = 0.4 * Q[x_prev][y_prev][0] + 0.6 * (reward + max(Q[x_new][y_new]))

def collision(bird_x, bird_y, up_pipes, bttm_pipes):
    """Check for collisions."""
    if bird_y >= BASEY - IMAGES['bird'].get_height() or bird_y < 0:
        return True
    for pipe in up_pipes:
        if bird_y < pipe['y'] + IMAGES['pipe'][0].get_height() and abs(bird_x - pipe['x']) < IMAGES['pipe'][0].get_width():
            return True
    for pipe in bttm_pipes:
        if bird_y + IMAGES['bird'].get_height() > pipe['y'] and abs(bird_x - pipe['x']) < IMAGES['pipe'][0].get_width():
            return True
    return False

def game_start(generation, x_scores, y_scores):
    """Main game loop for one generation/trial."""
    score = 0
    bird_x = SW // 5
    bird_y = SH // 2
    base_x1, base_x2 = 0, SW
    bg_x1, bg_x2 = 0, IMAGES['background'].get_width()

    pipe1, pipe2 = get_new_pipe(), get_new_pipe()
    up_pipes = [pipe1[0], pipe2[0]]
    bttm_pipes = [pipe1[1], pipe2[1]]
    pipe_vel_x = -4

    bird_vel_y = -9
    bird_max_vel = 10
    bird_acc = 1
    player_flap_acc = -8
    player_flapped = False

    while True:
        x_prev, y_prev = convert(bird_x, bird_y, bttm_pipes)
        jump = ai_play(x_prev, y_prev)

        for event in pygame.event.get():
            if event.type == QUIT:
                plt.scatter(x_scores, y_scores)
                plt.xlabel("Generation / Trial")
                plt.ylabel("Score")
                plt.title("Flappy Birds")
                plt.show()
                pygame.quit()
                sys.exit()

        if jump and bird_y > 0:
            bird_vel_y = player_flap_acc
            player_flapped = True

        # Move bird
        if bird_vel_y < bird_max_vel and not player_flapped:
            bird_vel_y += bird_acc
        if player_flapped:
            player_flapped = False
        bird_y += min(bird_vel_y, BASEY - bird_y - IMAGES['bird'].get_height())

        # Move pipes
        for up_pipe, down_pipe in zip(up_pipes, bttm_pipes):
            up_pipe['x'] += pipe_vel_x
            down_pipe['x'] += pipe_vel_x

        # Generate new pipes
        if 0 < up_pipes[0]['x'] < 5:
            new_pipe = get_new_pipe()
            up_pipes.append(new_pipe[0])
            bttm_pipes.append(new_pipe[1])
        if up_pipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            up_pipes.pop(0)
            bttm_pipes.pop(0)

        # Move base and background
        base_x1 -= 4
        base_x2 -= 4
        if base_x1 <= -IMAGES['base'].get_width():
            base_x1 = base_x2
            base_x2 = base_x1 + IMAGES['base'].get_width()
        bg_x1 -= 2
        bg_x2 -= 2
        if bg_x1 <= -IMAGES['background'].get_width():
            bg_x1 = bg_x2
            bg_x2 = bg_x1 + IMAGES['background'].get_width()

        # Check collision
        crash = collision(bird_x, bird_y, up_pipes, bttm_pipes)
        x_new, y_new = convert(bird_x, bird_y, bttm_pipes)
        reward = -1000 if crash else 15
        update_Q(x_prev, y_prev, jump, reward, x_new, y_new)
        if crash:
            return score

        # Score update
        player_mid_pos = bird_x + IMAGES['bird'].get_width() / 2
        for pipe in up_pipes:
            pipe_mid_pos = pipe['x'] + IMAGES['pipe'][0].get_width() / 2
            if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                score += 1

        # Draw everything
        WINDOW.blit(IMAGES['background'], (bg_x1, 0))
        WINDOW.blit(IMAGES['background'], (bg_x2, 0))
        for up_pipe, down_pipe in zip(up_pipes, bttm_pipes):
            WINDOW.blit(IMAGES['pipe'][0], (up_pipe['x'], up_pipe['y']))
            WINDOW.blit(IMAGES['pipe'][1], (down_pipe['x'], down_pipe['y']))
        WINDOW.blit(IMAGES['base'], (base_x1, BASEY))
        WINDOW.blit(IMAGES['base'], (base_x2, BASEY))
        WINDOW.blit(IMAGES['bird'], (bird_x, bird_y))
        WINDOW.blit(Font.render(f"Score: {score}", True, (255, 255, 255)), (SW-10-100, 10))
        WINDOW.blit(Font.render(f"Generation: {generation}", True, (255, 255, 255)), (0, 0))

        pygame.display.update()
        FPSCLOCK.tick(FPS)

# ------------------ Main ------------------ #
if __name__ == "__main__":
    pygame.init()
    WINDOW = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Flappy Bird")
    FPSCLOCK = pygame.time.Clock()
    Font = pygame.font.SysFont("comicsans", 30)
    load_images()
    static_screen()

    generation = 1
    x_scores, y_scores = [], []

    while True:
        score = game_start(generation, x_scores, y_scores)
        x_scores.append(generation)
        y_scores.append(score)
        generation += 1
