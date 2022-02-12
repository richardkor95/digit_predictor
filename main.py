import numpy as np
import pygame 

# settings 
grid_size = 28           #   28x28 Grid 
RED = (255, 0, 0)
GRAY = (100, 100, 100)
BLACK = (0, 0 ,0)

pygame.init()
pygame.font.init()
myfont = pygame.font.SysFont('Arial', 19)

def main():
    screen = pygame.display.set_mode((560, 600), pygame.RESIZABLE)
    clock = pygame.time.Clock()

    matrix = np.zeros([grid_size, grid_size], dtype='float32') # matrix with indizes 

    prev_idx, prev_idy = -1, -1         # to check if mouse left the canvas
    key_pressed = False                 # to click on key only once
    mouse_press_prev = False 

    while True: 
        events = pygame.event.get()
        keys = pygame.key.get_pressed()

        for event in events:
            if event.type == pygame.QUIT:
                return 
            if event.type == pygame.VIDEORESIZE:
                pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                number_vector = np.zeros(grid_size**2)
                screen.fill(BLACK)
        idx, idy = get_cell_index()
        if idx >= 0 and idx <= 27 and idy >= 0 and idy <= 27:
            if pygame.mouse.get_pressed()[0]:
                if not mouse_press_prev:
                    mouse_press_prev = True
                    matrix = update_matrix(matrix, idx, idy)
                else:
                    if not (prev_idx == idx and prev_idy == idy):
                        update_matrix(matrix, idx, idy)
            else:
                mouse_press_prev = False

        if keys[pygame.K_p] and not key_pressed:
            pass
        if keys[pygame.K_c]:
            matrix = np.zeros([grid_size, grid_size], dtype='float32') # matrix with indizes 

        prev_idx, prev_idy = idx, idy
        draw_num_on_screen(screen, matrix)
        # guess number 

        make_grid(screen, grid_size)

        pygame.display.flip()
        clock.tick(60)

            

def get_square_size():
    width_screen = pygame.display.Info().current_w
    height_screen = pygame.display.Info().current_h
    screen_size = min(width_screen, height_screen)
    square_size = int(screen_size/grid_size)
    return square_size


def make_grid(screen, grid_size):  
    square_size = get_square_size()
    # draw of the grid
    grid_thickness = 1
    for i in range(grid_size+1):
        pygame.draw.rect(screen, GRAY, (i*square_size - grid_thickness, 0, 2*grid_thickness, square_size*grid_size))
        pygame.draw.rect(screen, GRAY, (0, i*square_size - grid_thickness, square_size*grid_size, 2*grid_thickness))

    # Display message under the grid
    textsurface = myfont.render('Press <c> to clear, Press <p> to predict', False, RED)
    screen.blit(textsurface,(0, (grid_size)*square_size))


def update_matrix(matrix, idx, idy):  # could be better by not only use 0 or 1 but nevermind...
    for i in range(-1, 2):
        for j in range(-1, 2):
            if idx + i > -1 and idx + i < 28 and idy + j > -1 and idy + j < 28:
                matrix[idx + i,idy + j] += 0.25
                if matrix[idx + i,idy + j] > 1:
                    matrix[idx + i,idy + j] = 1

    matrix[idx, idy] = 1
    return matrix


def draw_num_on_screen(screen, matrix):
    square_size = get_square_size()
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            col = (matrix[i, j]*255, matrix[i, j]*255, matrix[i, j]*255) 
            pygame.draw.rect(screen, col, (i*square_size, j*square_size, square_size, square_size))


def print_matrix(matrix):
    print(matrix)
    print('#################################################')
            

def draw_number(screen, idx, idy):
    square_size = get_square_size()
    pygame.draw.rect(screen, RED, (idx*square_size, idy*square_size, square_size, square_size)) 


def get_cell_index():
    mx, my = pygame.mouse.get_pos()
    square_size = get_square_size()
    idx, idy = int(mx / square_size), int(my / square_size)
    return [idx, idy]


if __name__ == '__main__':
    main()

