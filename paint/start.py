import pygame
import math
import random
import sys

pygame.init()

WIDTH, HEIGHT = 800, 600
BACKGROUND_COLOR = (230, 230, 250)
BUTTON_COLOR = (30, 144, 255)
BUTTON_HOVER_COLOR = (100, 149, 237)
TEXT_COLOR = (255, 255, 255)
FONT_SIZE = 100
SPLASH_COUNT = 5
MAX_SPLASHES = 20
NUM_RAYS = 360

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Projector Palette")

font = pygame.font.Font(None, FONT_SIZE)


class Ray:
    def __init__(self, x1, y1, dirX, dirY):
        self.x1 = x1
        self.y1 = y1
        self.dirX = dirX
        self.dirY = dirY

    def collide(self, wall):
        wx1 = wall.x1
        wy1 = wall.y1
        wx2 = wall.x2
        wy2 = wall.y2
        rx3 = self.x1
        ry3 = self.y1
        rx4 = self.x1 + self.dirX
        ry4 = self.y1 + self.dirY

        n = (wx1 - rx3) * (ry3 - ry4) - (wy1 - ry3) * (rx3 - rx4)
        d = (wx1 - wx2) * (ry3 - ry4) - (wy1 - wy2) * (rx3 - rx4)

        if d == 0:
            return False

        t = n / d
        u = ((wx2 - wx1) * (wy1 - ry3) - (wy2 - wy1) * (wx1 - rx3)) / d

        if 0 < t < 1 and u > 0:
            px = wx1 + t * (wx2 - wx1)
            py = wy1 + t * (wx2 - wy1)
            return (px, py)
        return False


class Wall:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def show(self, surface):
        pygame.draw.line(
            surface, (255, 255, 255), (self.x1, self.y1), (self.x2, self.y2), 5
        )


class Light:
    def __init__(self, x1, y1, n):
        self.x1 = x1
        self.y1 = y1
        self.rays = []
        self.n = n
        for i in range(0, 360, int(360 / n)):
            self.rays.append(
                Ray(
                    self.x1,
                    self.y1,
                    math.cos(math.radians(i)),
                    math.sin(math.radians(i)),
                )
            )

    def show(self, surface, walls):
        for ray in self.rays:
            ray.x1 = self.x1
            ray.y1 = self.y1
            closest = 1000000
            closestPoint = None
            for wall in walls:
                intersection = ray.collide(wall)
                if intersection:
                    distance = math.sqrt(
                        (ray.x1 - intersection[0]) ** 2
                        + (ray.y1 - intersection[1]) ** 2
                    )
                    if distance < closest:
                        closest = distance
                        closestPoint = intersection

            if closestPoint:
                pygame.draw.line(
                    surface, (255, 255, 255), (ray.x1, ray.y1), closestPoint
                )


def draw_projector(x, y):
    pygame.draw.rect(screen, (50, 50, 50), (x, y, 250, 150), border_radius=15)
    pygame.draw.rect(screen, (70, 70, 70), (x + 15, y + 35, 220, 80), border_radius=10)
    pygame.draw.circle(screen, (255, 200, 0), (x + 250, y + 75), 40)
    pygame.draw.circle(screen, (0, 0, 0), (x + 250, y + 75), 30)
    pygame.draw.circle(screen, (0, 200, 200), (x + 250, y + 75), 20)

    colors = [(200, 0, 0), (0, 200, 0), (0, 0, 200)]
    positions = [(x + 40, y + 40), (x + 90, y + 40), (x + 140, y + 40)]
    for pos, color in zip(positions, colors):
        pygame.draw.rect(screen, color, (pos[0], pos[1], 30, 30), border_radius=5)


active_splashes = []


def create_splash():
    if len(active_splashes) < MAX_SPLASHES:
        paint_colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        color = random.choice(paint_colors)
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        radius = random.randint(20, 50)
        active_splashes.append((color, (x, y), radius))


def draw_splashes():
    for color, (x, y), radius in active_splashes:
        pygame.draw.circle(screen, color, (x, y), radius)
        for _ in range(10):
            offset_x = random.randint(-radius, radius)
            offset_y = random.randint(-radius, radius)
            pygame.draw.circle(screen, color, (x + offset_x, y + offset_y), radius // 2)


walls = [
    Wall(0, 0, WIDTH - 1, 0),
    Wall(0, 0, 0, HEIGHT - 1),
    Wall(0, HEIGHT - 1, WIDTH - 1, HEIGHT - 1),
    Wall(WIDTH - 1, 0, WIDTH - 1, HEIGHT - 1),
]

for i in range(random.randint(0, 10)):
    walls.append(
        Wall(
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT),
            random.randint(0, WIDTH),
            random.randint(0, HEIGHT),
        )
    )

l = Light(500, 500, NUM_RAYS)


class Button:
    def __init__(self, text, x, y, width, height):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.color = BUTTON_COLOR

    def draw(self):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=10)
        text_surface = font.render(self.text, True, TEXT_COLOR)
        text_rect = text_surface.get_rect(center=self.rect.center)
        screen.blit(text_surface, text_rect)

    def click(self):
        mouse_pos = pygame.mouse.get_pos()
        return self.rect.collidepoint(mouse_pos)


start_button = Button("Start", WIDTH // 2 - 100, HEIGHT // 2 + 100, 200, 70)

splash_timer = 0
splash_interval = 500

running = True
while running:
    screen.fill(BACKGROUND_COLOR)

    draw_projector(WIDTH // 2 - 125, HEIGHT // 4)

    current_time = pygame.time.get_ticks()
    if current_time - splash_timer > splash_interval:
        create_splash()
        splash_timer = current_time

    draw_splashes()

    title_surface = font.render("Projector Palette", True, (0, 0, 0))
    title_rect = title_surface.get_rect(center=(WIDTH // 2, HEIGHT // 8))
    screen.blit(title_surface, title_rect)

    for wall in walls:
        wall.show(screen)

    l.x1, l.y1 = pygame.mouse.get_pos()
    l.show(screen, walls)

    start_button.draw()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if start_button.click():
                pygame.quit()
                import project

    pygame.display.flip()

pygame.quit()
sys.exit()
