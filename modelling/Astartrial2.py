import math
import heapq
import pygame
import sys

# ==============================================================================
# --- 1. The Treasure Map (Simple 6-Row x 10-Column Grid) ---
# ==============================================================================
# This map uses the simple (row, column) format you requested.

MAZE_MAP = {
    # --- Column 0 ---
    (0, 0): 'entry', (1, 0): 'empty', (2, 0): 'empty', (3, 0): 'obstacle', (4, 0): 'empty', (5, 0): 'empty',


    # --- Column 1 ---
    (0, 1): 'empty', (1, 1): 'trap2', (2, 1): 'empty', (3, 1): 'reward1', (4, 1): 'empty', (5, 1): 'empty',


    # --- Column 2 ---
    (0, 2): 'empty', (1, 2): 'empty', (2, 2): 'obstacle', (3, 2): 'empty', (4, 2): 'trap2', (5, 2): 'empty',


    # --- Column 3 ---
    (0, 3): 'empty', (1, 3): 'trap4', (2, 3): 'empty', (3, 3): 'obstacle', (4, 3): 'treasure', (5, 3): 'empty',


    # --- Column 4 ---
    (0, 4): 'reward1', (1, 4): 'treasure', (2, 4): 'obstacle', (3, 4): 'empty', (4, 4): 'obstacle', (5, 4): 'empty',

    # --- Column 5 ---
    (0, 5): 'empty', (1, 5): 'empty', (2, 5): 'empty', (3, 5): 'trap3', (4, 5): 'empty', (5, 5): 'reward2',


    # --- Column 6 ---
    (0, 6): 'empty', (1, 6): 'trap3', (2, 6): 'empty', (3, 6): 'obstacle', (4, 6): 'obstacle', (5, 6): 'empty',


    # --- Column 7 ---
    (0, 7): 'empty', (1, 7): 'empty', (2, 7): 'reward2', (3, 7): 'treasure', (4, 7): 'obstacle', (5, 7): 'empty',


    # --- Column 8 ---
    (0, 8): 'empty', (1, 8): 'obstacle', (2, 8): 'trap1', (3, 8): 'empty', (4, 8): 'empty', (5, 8): 'empty',

    
    # --- Column 9 ---
    (0, 9): 'empty', (1, 9): 'empty', (2, 9): 'empty', (3, 9): 'treasure', (4, 9): 'empty', (5, 9): 'empty',
}


# ==============================================================================
# --- 2. State and Node Representation for A* Search ---
# ==============================================================================
# These classes represent the state of the agent and nodes in the search tree.

class State:
    """Represents a unique configuration of the agent and the game world."""
    def __init__(self, location, uncollected_treasures, step_multiplier=1.0,
                 energy_multiplier=1.0, last_move_direction=None, visited_specials=frozenset()):
        self.location = location
        self.uncollected_treasures = frozenset(uncollected_treasures)
        self.step_multiplier = step_multiplier
        self.energy_multiplier = energy_multiplier
        self.last_move_direction = last_move_direction
        self.visited_specials = frozenset(visited_specials)

    def __eq__(self, other):
        return (self.location == other.location and
                self.uncollected_treasures == other.uncollected_treasures and
                self.step_multiplier == other.step_multiplier and
                self.energy_multiplier == other.energy_multiplier and
                self.last_move_direction == other.last_move_direction and
                self.visited_specials == other.visited_specials)

    def __hash__(self):
        return hash((self.location, self.uncollected_treasures, self.step_multiplier,
                     self.energy_multiplier, self.last_move_direction, self.visited_specials))

    def __repr__(self):
        return (f"State(Loc={self.location}, TreasLeft={len(self.uncollected_treasures)}, "
                f"StepX={self.step_multiplier:.2f}, EnergyX={self.energy_multiplier:.2f})")

class Node:
    """Represents a node in the A* search tree."""
    def __init__(self, state, parent=None, g_cost=0, h_cost=0):
        self.state = state
        self.parent = parent
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.f_cost = g_cost + h_cost

    def __lt__(self, other):
        # Tie-breaking for heapq stability
        return self.f_cost < other.f_cost

# ==============================================================================
# --- 3. Heuristic and Movement for the New Grid ---
# ==============================================================================

def grid_to_cube(row, col):
    x = col
    z = row - (col + (col & 1)) / 2  # "odd-q" vertical layout
    y = -x - z
    return x, y, z

def heuristic_distance(pos1, pos2):
    x1, y1, z1 = grid_to_cube(pos1[0], pos1[1])
    x2, y2, z2 = grid_to_cube(pos2[0], pos2[1])
    return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) / 2

def heuristic(state):
    if not state.uncollected_treasures:
        return 0
    max_dist = max(heuristic_distance(state.location, t_loc) for t_loc in state.uncollected_treasures)
    min_cost_per_step = 0.25 # Minimum cost assuming both rewards are collected
    return max_dist * min_cost_per_step

def get_valid_neighbors(row, col):
    """ 'odd-q' vertical layout neighbors """
    potential_moves = []
    if col % 2 == 1:  # ODD columns
        move_directions = [(-1, 0), (1, 0), (0, 1), (1, 1), (0, -1), (-1, -1)]
    else:  # EVEN columns
        move_directions = [(-1, 0), (1, 0), (-1, 1), (0, 1), (0, -1), (1, -1)]

    for dr, dc in move_directions:
        neighbor_coord = (row + dr, col + dc)
        if neighbor_coord in MAZE_MAP and MAZE_MAP.get(neighbor_coord, 'obstacle') != 'obstacle':
            potential_moves.append(((dr, dc), neighbor_coord))
    return potential_moves

# ==============================================================================
# --- 4. A* Search Algorithm (Updated for new rules and Visualization) ---
# ==============================================================================

def a_star_search(start_state, visualize=False):
    history = []
    open_set = []
    node_id_counter = 0
    start_node = Node(start_state, g_cost=0, h_cost=heuristic(start_state))
    heapq.heappush(open_set, (start_node.f_cost, node_id_counter, start_node))
    node_id_counter += 1

    closed_set = set()
    cost_so_far = {start_state: 0}

    while open_set:
        _, _, current_node = heapq.heappop(open_set)
        current_state = current_node.state

        if current_state in closed_set:
            continue
        closed_set.add(current_state)

        if visualize:
            # For visualization, we only care about the location part of the sets
            open_locs = {node.state.location for _, _, node in open_set}
            closed_locs = {state.location for state in closed_set}
            history.append({
                'open': open_locs,
                'closed': closed_locs,
                'current': current_state.location
            })

        if not current_state.uncollected_treasures:
            path = []
            temp = current_node
            while temp:
                path.append(temp.state)
                temp = temp.parent
            return path[::-1], current_node.g_cost, history

        row, col = current_state.location
        for move_direction, next_loc in get_valid_neighbors(row, col):
            cell_type = MAZE_MAP.get(next_loc, '').lower()
            if cell_type == 'trap4': # Rule: Trap4 is forbidden
                continue

            new_params = {
                'location': next_loc,
                'uncollected_treasures': set(current_state.uncollected_treasures),
                'step_multiplier': current_state.step_multiplier,
                'energy_multiplier': current_state.energy_multiplier,
                'last_move_direction': move_direction,
                'visited_specials': set(current_state.visited_specials)
            }
            move_cost = 1.0 * current_state.step_multiplier * current_state.energy_multiplier
            has_been_visited = next_loc in current_state.visited_specials

            # Process special tiles only if they haven't been visited before in this path
            if not has_been_visited:
                if cell_type == 'treasure':
                    if next_loc in new_params['uncollected_treasures']:
                        new_params['uncollected_treasures'].remove(next_loc)
                elif cell_type == 'reward1':
                    new_params['energy_multiplier'] /= 2
                    new_params['visited_specials'].add(next_loc)
                elif cell_type == 'reward2':
                    new_params['step_multiplier'] /= 2
                    new_params['visited_specials'].add(next_loc)
                elif cell_type == 'trap1':
                    new_params['energy_multiplier'] *= 2
                    new_params['visited_specials'].add(next_loc)
                elif cell_type == 'trap2':
                    new_params['step_multiplier'] *= 2
                    new_params['visited_specials'].add(next_loc)
                elif cell_type == 'trap3':
                    d_row, d_col = move_direction
                    final_landing_loc = (next_loc[0] + d_row, next_loc[1] + d_col)
                    if final_landing_loc not in MAZE_MAP or MAZE_MAP.get(final_landing_loc) == 'obstacle':
                        continue
                    new_params['location'] = final_landing_loc
                    new_params['visited_specials'].add(next_loc)

            next_state = State(**new_params)
            new_g_cost = current_node.g_cost + move_cost

            if new_g_cost < cost_so_far.get(next_state, float('inf')):
                cost_so_far[next_state] = new_g_cost
                h = heuristic(next_state)
                new_node = Node(next_state, current_node, new_g_cost, h)
                heapq.heappush(open_set, (new_node.f_cost, node_id_counter, new_node))
                node_id_counter += 1

    return None, None, history


# ==============================================================================
# --- 5. Pygame Visualization Setup ---
# ==============================================================================
pygame.init()

# --- Constants ---
HEX_RADIUS = 30
HEX_WIDTH = math.sqrt(3) * HEX_RADIUS
HEX_HEIGHT = 2 * HEX_RADIUS
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 700
ANIM_FPS = 15 # Speed of the animation

# --- Colors ---
C_WHITE = (255, 255, 255)
C_BLACK = (0, 0, 0)
C_GREY = (50, 50, 50)
C_LIGHT_GREY = (180, 180, 180)
C_GREEN = (50, 200, 50)
C_BLUE = (100, 100, 255)
C_RED = (255, 50, 50)
C_YELLOW = (255, 255, 0)
C_PURPLE = (160, 32, 240)
C_ORANGE = (255, 165, 0)
C_CYAN = (0, 255, 255)
C_PINK = (255, 105, 180)
C_DARK_GREEN = (0, 100, 0)
C_DARK_RED = (139, 0, 0)
C_AGENT = (255, 0, 255) # Magenta

TILE_VISUALS = {
    'entry':    {'color': C_WHITE, 'text': 'E'},
    'empty':    {'color': C_WHITE, 'text': ''},
    'obstacle': {'color': C_BLACK, 'text': ''},
    'treasure': {'color': C_YELLOW, 'text': 'T'},
    'reward1':  {'color': C_GREEN, 'text': 'R1'},
    'reward2':  {'color': C_GREEN, 'text': 'R2'},
    'trap1':    {'color': C_PURPLE, 'text': 'T1'},
    'trap2':    {'color': C_PURPLE, 'text': 'T2'},
    'trap3':    {'color': C_PURPLE, 'text': 'T3'},
    'trap4':    {'color': C_PURPLE, 'text': 'T4'},
}

# --- Pygame Screen and Font ---
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("A* Treasure Hunt Visualization")
clock = pygame.time.Clock()
FONT = pygame.font.SysFont('Arial', 16)
FONT_BIG = pygame.font.SysFont('Arial', 24, bold=True)

def grid_to_pixel(row, col):
    """ Converts 'odd-q' vertical grid coordinates to pixel coordinates. """
    x_offset = 60
    y_offset = 60
    pixel_x = x_offset + col * HEX_WIDTH * 0.75
    pixel_y = y_offset + row * HEX_HEIGHT
    if col % 2 == 1:
        pixel_y += HEX_HEIGHT / 2
    return int(pixel_x), int(pixel_y)

def draw_hexagon(surface, color, center_pixel, radius):
    points = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.pi / 180 * angle_deg
        points.append((center_pixel[0] + radius * math.cos(angle_rad),
                       center_pixel[1] + radius * math.sin(angle_rad)))
    pygame.draw.polygon(surface, color, points)
    pygame.draw.polygon(surface, C_GREY, points, 2) # Border

def draw_map_and_search_state(map_data, search_state=None, path_locs=None, agent_loc=None):
    # Draw base map tiles
    for loc, tile_type in map_data.items():
        center_pixel = grid_to_pixel(loc[0], loc[1])
        visuals = TILE_VISUALS.get(tile_type, {'color': C_WHITE, 'text': '?'})
        
        # Overlay search state colors
        draw_color = visuals['color']
        if search_state:
            if loc in search_state.get('closed', set()):
                draw_color = (120, 120, 120) # Gray for closed
            if loc in search_state.get('open', set()):
                draw_color = C_CYAN # Cyan for open
            if loc == search_state.get('current', None):
                draw_color = C_DARK_GREEN # Bright green for current
        
        draw_hexagon(screen, draw_color, center_pixel, HEX_RADIUS)
        
        # Draw tile text
        if visuals['text']:
            text_surf = FONT.render(visuals['text'], True, C_BLACK)
            text_rect = text_surf.get_rect(center=center_pixel)
            screen.blit(text_surf, text_rect)

    # Draw final path line
    if path_locs:
        pixel_path = [grid_to_pixel(r, c) for r,c in path_locs]
        if len(pixel_path) > 1:
            pygame.draw.lines(screen, C_AGENT, False, pixel_path, 5)

    # Draw agent
    if agent_loc:
        center_pixel = grid_to_pixel(agent_loc[0], agent_loc[1])
        pygame.draw.circle(screen, C_AGENT, center_pixel, HEX_RADIUS // 2)

# ==============================================================================
# --- 6. Main Execution Block ---
# ==============================================================================

if __name__ == "__main__":
    print("--- A* Treasure Hunt Solver (Rules from Image) ---")
    
    entry_point = None
    initial_treasures = []
    for loc, type in MAZE_MAP.items():
        if type == 'entry':
            entry_point = loc
        elif type == 'treasure':
            initial_treasures.append(loc)
    
    if not entry_point:
        raise ValueError("Entry point not found in MAZE_MAP!")
    
    start_state = State(
        location=entry_point,
        uncollected_treasures=initial_treasures
    )
    
    print(f"\nStarting at entry point: {entry_point}")
    print(f"Treasures to collect: {len(initial_treasures)} at {initial_treasures}")
    print("\nInitiating A* Search...")
    
    path_states, total_cost, search_history = a_star_search(start_state, visualize=True)
    
    print("-" * 50)
    
    if path_states:
        print(" Solution Found! Optimal Path:")
        for i, state in enumerate(path_states):
            # Console output remains the same...
            action_desc = ""
            if i == 0:
                action_desc = f"Start at {state.location}."
            else:
                prev_state = path_states[i-1]
                if prev_state.last_move_direction is not None and heuristic_distance(prev_state.location, state.location) > 1.5:
                    trap_loc_row = prev_state.location[0] + prev_state.last_move_direction[0]
                    trap_loc_col = prev_state.location[1] + prev_state.last_move_direction[1]
                    trap_loc = (trap_loc_row, trap_loc_col)
                    action_desc = f"Moved towards {trap_loc}, hit Trap 3, and landed at {state.location}."
                else:
                    action_desc = f"Moved to {state.location}."
                
                treasures_collected = prev_state.uncollected_treasures - state.uncollected_treasures
                newly_visited_specials = state.visited_specials - prev_state.visited_specials

                if treasures_collected:
                    action_desc += f" (Collected Treasure at {list(treasures_collected)[0]})"
                elif newly_visited_specials:
                    triggered_loc = list(newly_visited_specials)[0]
                    action_desc += f" (Triggered {MAZE_MAP[triggered_loc]} at {triggered_loc})"

            print(f"Step {i:02d}: {action_desc:<75} | State: {state}")
            
        print("-" * 50)
        print(f"\nTotal optimal path cost: {total_cost:.2f}")
        
        # --- Pygame Main Loop ---
        running = True
        anim_stage = "SEARCH" # Stages: SEARCH -> PATH -> DONE
        frame_index = 0
        
        path_locations = [state.location for state in path_states]

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            screen.fill(C_GREY)
            
            # --- Animation Logic ---
            if anim_stage == "SEARCH":
                if frame_index < len(search_history):
                    current_search_state = search_history[frame_index]
                    draw_map_and_search_state(MAZE_MAP, search_state=current_search_state)
                    
                    status_text = FONT_BIG.render(f"Animating Search... Step: {frame_index}/{len(search_history)}", True, C_WHITE)
                    screen.blit(status_text, (SCREEN_WIDTH - 450, 20))
                    
                    frame_index += 1
                else:
                    anim_stage = "PATH"
                    frame_index = 0

            elif anim_stage == "PATH":
                if frame_index < len(path_locations):
                    # Show the final state of the search (all closed nodes)
                    final_search_state = search_history[-1] if search_history else None
                    draw_map_and_search_state(MAZE_MAP, search_state=final_search_state)
                    
                    # Draw path and agent up to current frame
                    current_path_segment = path_locations[:frame_index + 1]
                    agent_pos = path_locations[frame_index]
                    draw_map_and_search_state(MAZE_MAP, path_locs=current_path_segment, agent_loc=agent_pos)

                    status_text = FONT_BIG.render(f"Animating Final Path... Step: {frame_index}/{len(path_locations)-1}", True, C_WHITE)
                    screen.blit(status_text, (SCREEN_WIDTH - 450, 20))

                    frame_index += 1
                else:
                    anim_stage = "DONE"
            
            elif anim_stage == "DONE":
                # Draw the final static scene
                final_search_state = search_history[-1] if search_history else None
                draw_map_and_search_state(MAZE_MAP, search_state=final_search_state, path_locs=path_locations, agent_loc=path_locations[-1])
                
                status_text = FONT_BIG.render(f"Done! Final Cost: {total_cost:.2f}", True, C_GREEN)
                screen.blit(status_text, (SCREEN_WIDTH - 450, 20))

            pygame.display.flip()
            clock.tick(ANIM_FPS)
            
        pygame.quit()
        sys.exit()

    else:
        print("\n No solution found. It's impossible to collect all treasures.")