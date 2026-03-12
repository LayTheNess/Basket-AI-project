import pygame
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Initialize Pygame
pygame.init()

# Constants
PENALITE = -20
WIDTH, HEIGHT = 1000,800
MAX_DISTANCE = math.sqrt(WIDTH**2 + HEIGHT**2)
CLOSE_REWARD_SCALE = 1000
FPS = 60
BALL_RADIUS = 8
HOOP_X, HOOP_Y = 800, 150
HOOP_RADIUS = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
BROWN = (139, 69, 19)
GREEN = (34, 139, 34)

class BasketballShooterEnv(gym.Env):
    metadata = {"render_modes": ["human", "no_render"], "render_fps": FPS}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.font = None
        self.gravity = 0.3
        
        # Action space: [angle (20-80 degrees), power (10-25)]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space: [ball_x (0-WIDTH), distance_to_hoop (0-diagonal of screen)]
        #max_distance = math.sqrt(WIDTH**2 + HEIGHT**2)
        self.observation_space = spaces.Box(low=0, high=1500, shape=(2,), dtype=np.float32)

        self.score = 0
        
    def reset_shot(self, angle=None, power=None):
        self.start_x = random.randint(100, 400)
        self.start_y = HEIGHT - 100

        self.ball_x = self.start_x
        self.ball_y = self.start_y
                
        self.trajectory = [(self.ball_x, self.ball_y)]

        self.shot_complete = False
        self.scored = False
        self.frame_count = 0
        
        
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_shot()
        return self._get_observation(), {}
        
    
    def _get_observation(self):
        distance = math.sqrt((self.ball_x - HOOP_X)**2 + (self.ball_y - HOOP_Y)**2)
        return np.array([float(self.ball_x), distance], dtype=np.float32)
    
    def step(self, action):
        self.angle = np.interp(action[0], [-1, 1], [20, 80])
        self.power = np.interp(action[1], [-1, 1], [10, 25])  

        angle_rad = math.radians(self.angle)
        self.vx = self.power * math.cos(angle_rad)
        self.vy = -self.power * math.sin(angle_rad)

        terminated = False
        truncated = False
        reward = 0
        dist_min = 1000
        enable_bonus = 0
        self.max_ball_y = self.start_y
        max_steps = 400

        while not self.shot_complete:
            self.vy += self.gravity
            self.ball_x += self.vx
            self.ball_y += self.vy

            distance = math.sqrt((self.ball_x - HOOP_X)**2 + (self.ball_y - HOOP_Y)**2)
            distance_reward = CLOSE_REWARD_SCALE * (1 - (dist_min/ MAX_DISTANCE))
            if distance < dist_min and self.ball_y > HOOP_Y and self.vy > 0:
                dist_min = distance

            self.trajectory.append( (int(self.ball_x), int(self.ball_y)) )
            self.frame_count += 1
            
            if self.ball_y < 0: #too high
                self.shot_complete = True
                reward = PENALITE
                break

            if self.ball_y > HEIGHT: #too low (hit the ground)
                self.shot_complete = True
                reward = PENALITE
                break

            if self.frame_count > 400:
                self.shot_complete = True
                break

            if self.vy <=0 and distance < HOOP_RADIUS: #scoring backward
                reward = PENALITE
                self.shot_complete = True
                break
            
            if self.ball_y <= HOOP_Y and not self.scored:
                distance = math.sqrt((self.ball_x - HOOP_X)**2 + (self.ball_y - HOOP_Y)**2)
                if self.ball_x > HOOP_X - 150:  
                    enable_bonus = 1
                if distance < HOOP_RADIUS:
                    if self.vy > 0:
                        self.scored = True
                        self.score += 1
                        reward = 100
                        self.shot_complete = True
                        break
    
            

            if self.ball_x > HOOP_X +100 and self.vy <0 and dist_min == 1000: #powershot bellow the basket
                
                self.shot_complete = True
                reward = PENALITE
                break

            if self.render_mode == "human":
                self.render()

        if not self.scored and self.shot_complete:
            self.shot_complete = True
            if enable_bonus:
                reward += 200/(1 + dist_min) # pénalité proportionnelle à la distance minimale
            

        terminated = self.shot_complete
                
        ret = self._get_observation(), reward, terminated, truncated, {}
        return ret
    
    def render(self):
        if self.screen is None:
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Basketball Shooter AI")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
        
        self.screen.fill(GREEN)
        pygame.draw.rect(self.screen, BROWN, (0, HEIGHT - 50, WIDTH, 50))
        pygame.draw.circle(self.screen, RED, (HOOP_X, HOOP_Y), HOOP_RADIUS, 3)
        pygame.draw.line(self.screen, BROWN, (HOOP_X - 30, HOOP_Y - 50), (HOOP_X - 20, HOOP_Y), 3)
        pygame.draw.line(self.screen, BROWN, (HOOP_X + 30, HOOP_Y - 50), (HOOP_X + 20, HOOP_Y), 3)
        
        if len(self.trajectory) > 1:
            pygame.draw.lines(self.screen, WHITE, False, self.trajectory, 1)
        
        pygame.draw.circle(self.screen, ORANGE, (int(self.ball_x), int(self.ball_y)), BALL_RADIUS)
        
        status = "SCORED!" if self.scored else "MISSED!" if self.shot_complete else "SHOOTING..."
        status_color = (0, 255, 0) if self.scored else RED if self.shot_complete else WHITE
        
        status_text = self.font.render(status, True, status_color)
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        
        self.screen.blit(status_text, (20, 20))
        self.screen.blit(score_text, (20, 60))
        
        pygame.display.flip()
        self.clock.tick(FPS)
        pygame.event.pump()
    
    def close(self):
        if self.screen is not None:
            pygame.quit()

if __name__ == "__main__":

    env = BasketballShooterEnv(render_mode="human")
    obs, info = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        x = obs[0]
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"{x} | {action} | {obs} | {terminated} | {reward}")

        if terminated:
            obs, info = env.reset()
    
    