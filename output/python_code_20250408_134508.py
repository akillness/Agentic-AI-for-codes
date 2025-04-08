import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FPS = 60
NUM_PARTICLES = 100
GRAVITY = 0.1

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Particle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = random.randint(2, 5)
        self.color = (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = random.uniform(2, 5)
        self.vx = self.speed * math.cos(self.angle)
        self.vy = self.speed * math.sin(self.angle)

    def update(self):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fireworks Simulation")
    clock = pygame.time.Clock()
    particles = []

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if random.random() < 0.05:  # Create new fireworks
            for _ in range(NUM_PARTICLES):
                particles.append(Particle(WIDTH // 2, HEIGHT // 2))

        screen.fill(BLACK)

        for particle in particles[:]:
            particle.update()
            particle.draw(screen)
            if particle.y > HEIGHT:  # Remove particles that fall off the screen
                particles.remove(particle)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()