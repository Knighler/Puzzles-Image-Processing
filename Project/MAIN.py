import cv2
import numpy as np
import os
import copy

#BASE_DIR = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\preprocessing_results\\2_denoised'
BASE_DIR = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls'
INPUT_ROOT = BASE_DIR
OUTPUT_ROOT = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\Paper_Results_flattening'
SUBFOLDERS = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']


W_COLOR = 1
W_GRADIENT = 0.2
W_EDGE_CONTINUITY = 10 


def smart_crop(img):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    threshold = 5.0 

    top = 0
    for r in range(height):
        if np.std(gray[r, :]) > threshold: top = r; break
    bottom = height
    for r in range(height-1, -1, -1):
        if np.std(gray[r, :]) > threshold: bottom = r + 1; break
    left = 0
    for c in range(width):
        if np.std(gray[:, c]) > threshold: left = c; break
    right = width
    for c in range(width-1, -1, -1):
        if np.std(gray[:, c]) > threshold: right = c + 1; break
    
    if (right - left) < (width * 0.5) or (bottom - top) < (height * 0.5): return img
    return img[top:bottom, left:right]


def get_canny_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def get_match_cost(piece_a, piece_b, edge_a_map, edge_b_map, relation):
    
    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    # Horizontal
    if relation == 0: 
        p_edge_a = A[:, -1, :]
        p_edge_b = B[:, 0, :]
        p_inner_a = A[:, -2, :] 
        bin_a= edge_a_map[:, -1]
        bin_b = edge_b_map[:, 0]
    #Vertical
    else: 
        p_edge_a=A[-1, :, :]
        p_edge_b = B[0, :, :]
        p_inner_a = A[-2, :, :] 
        bin_a = edge_a_map[-1, :]
        bin_b = edge_b_map[0, :]

    # Penalize black
    if np.std(p_edge_a[:, 0]) < 3 and np.std(p_edge_b[:, 0]) < 3:
        return 100000.0 

    # 2 Color Differevertical_neighbore
    color_diff = np.sum(np.abs(p_edge_a - p_edge_b))

    # 3 Gradient Continuity Prediction 
    trend = p_edge_a - p_inner_a
    expected_b = p_edge_a + trend
    grad_cost = np.sum(np.abs(expected_b - p_edge_b))

    # 4. Edge continuity 
    b_a = (bin_a > 0).astype(int)
    b_b = (bin_b > 0).astype(int)
    
    edge_cost = np.sum(np.bitwise_xor(b_a, b_b))
    matches = np.sum(np.bitwise_and(b_a, b_b))

    if matches > 0: 
        edge_cost -= (matches * 5.0) 

    total_cost = (W_COLOR * color_diff) + (W_GRADIENT * grad_cost) + (W_EDGE_CONTINUITY * edge_cost)
    return total_cost


#Best Buddies

def calculate_best_buddies_score(grid, costs):

    height, width = grid.shape
    buddies = 0
    total_joints = 0
    
    for r in range(height):
        for c in range(width):
            current_piece = grid[r, c]
            if current_piece == -1: continue
            # Check Right
            if c + 1 < width:
                neighbour_piece = grid[r, c+1]
                if neighbour_piece != -1:
                    total_joints += 1
                    best_for_current_piece = np.argmin(costs[current_piece, :, 0])
                    best_for_neighbour_piece = np.argmin(costs[neighbour_piece, :, 2])
                    if best_for_current_piece == neighbour_piece and best_for_neighbour_piece == current_piece:
                        buddies += 1

            # Check Down
            if r + 1 < height:
                neighbour_piece = grid[r+1, c]
                if neighbour_piece != -1:
                    total_joints += 1
                    best_for_current_piece = np.argmin(costs[current_piece, :, 1])
                    best_for_neighbour_piece = np.argmin(costs[neighbour_piece, :, 3])
                    if best_for_current_piece == neighbour_piece and best_for_neighbour_piece == current_piece:
                        buddies += 1

    if  total_joints > 0:
        return buddies / total_joints       
    else:
        0


def solve_single_run(pieces, dimension, base_costs, use_noise=False):
    num_pieces = len(pieces)
    edge_maps = [get_canny_edges(p) for p in pieces]
    activity_scores = [np.sum(e)/255.0 for e in edge_maps]
    avg_activity = np.mean(activity_scores)

    # Stochastic Greedy
    costs = base_costs.copy()
    if use_noise:
        noise = np.random.normal(0, 0.1, costs.shape) * costs
        costs += noise

    #Seed
    best_seed_val = np.inf
    seed_pair = (0, 1, 0)
    
    for i in range(num_pieces):
        # Award pieces with lots of details
        if activity_scores[i] < avg_activity: continue
        best_right = np.argmin(costs[i, :, 0])
        if np.argmin(costs[best_right, :, 2]) == i:
            if costs[i, best_right, 0] < best_seed_val:
                best_seed_val = costs[i, best_right, 0]
                seed_pair = (i, best_right, 0)

        best_bottom = np.argmin(costs[i, :, 1])
        if np.argmin(costs[best_bottom, :, 3]) == i:
            if costs[i, best_bottom, 1] < best_seed_val:
                best_seed_val = costs[i, best_bottom, 1]
                seed_pair = (i, best_bottom, 1)

    # Initialize Grid
    grid_size = dimension * 3
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    seed_y=dimension
    seed_x = dimension
    p1, p2, rel = seed_pair
    grid[seed_y, seed_x] = p1
    if rel == 0: 
        grid[seed_y, seed_x+1] = p2
    else: 
        grid[seed_y+1, seed_x] = p2
    placed = {p1, p2}

    # Greedy 
    while len(placed) < num_pieces:
        rows, cols = np.where(grid != -1)
        possible_slots = set()
        
        # Find all empty neighbors
        for r, c in zip(rows, cols):
            for  horizontal_offset, vertical_offset in [(0,1), (1,0), (0,-1), (-1,0)]:
                horizontal_neighbor, vertical_neighbor = r+ horizontal_offset, c+vertical_offset
                if grid[horizontal_neighbor, vertical_neighbor] == -1:
                    horizontal_span = max(rows.max(), horizontal_neighbor) - min(rows.min(), horizontal_neighbor) + 1
                    width_span = max(cols.max(), vertical_neighbor) - min(cols.min(), vertical_neighbor) + 1
                    if horizontal_span <= dimension and width_span <= dimension:
                        possible_slots.add((horizontal_neighbor, vertical_neighbor))

        best_move = None
        best_conf_score = -1.0 
        for (horizontal_neighbor, vertical_neighbor) in possible_slots:
            slot_costs = []
            for current_piece in range(num_pieces):
                if current_piece in placed: continue
                
                current_cost = 0
                count = 0
        
                if grid[horizontal_neighbor-1, vertical_neighbor] != -1: 
                    current_cost += costs[grid[horizontal_neighbor-1, vertical_neighbor], current_piece, 1]
                    count += 1
                if grid[horizontal_neighbor+1, vertical_neighbor] != -1: 
                    current_cost += costs[grid[horizontal_neighbor+1, vertical_neighbor], current_piece, 3]
                    count += 1
                if grid[horizontal_neighbor, vertical_neighbor-1] != -1: 
                    current_cost += costs[grid[horizontal_neighbor, vertical_neighbor-1], current_piece, 0]
                    count += 1
                if grid[horizontal_neighbor, vertical_neighbor+1] != -1: 
                    current_cost += costs[grid[horizontal_neighbor, vertical_neighbor+1], current_piece, 2]
                    count += 1
                
                if count > 0:
                    slot_costs.append((current_cost/count, current_piece))

            if not slot_costs: continue
            slot_costs.sort(key=lambda x: x[0])
            
            # Confidence vertical_neighbore 
            best_c, best_p = slot_costs[0]
            if len(slot_costs) > 1:
                second_c = slot_costs[1][0]
                ratio = second_c / (best_c + 1e-5) 
            else:
                ratio = 100.0 

            # Choose most confident
            if ratio > best_conf_score:
                best_conf_score = ratio
                best_move = (horizontal_neighbor, vertical_neighbor, best_p)

        #choose best move for that slot
        if best_move:
            grid[best_move[0], best_move[1]] = best_move[2]
            placed.add(best_move[2])
        else:
            break 

    # Crop Result
    rows = np.any(grid != -1, axis=1)
    cols = np.any(grid != -1, axis=0)
    return grid[rows][:, cols]

def solve_puzzle(pieces, dimension):
    num_pieces = len(pieces)
    edge_maps = [get_canny_edges(p) for p in pieces]
    costs = np.full((num_pieces, num_pieces, 4), np.inf)
    
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j: continue
            c_right = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 0)
            c_down  = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 1)
            costs[i, j, 0] = c_right
            costs[j, i, 2] = c_right
            costs[i, j, 1] = c_down
            costs[j, i, 3] = c_down

    # 2. Multi-Start Loop
    best_grid = None
    best_score = -1
    
    # Deterministic
    grid = solve_single_run(pieces, dimension, costs, use_noise=False)
    if grid is not None and grid.shape == (dimension, dimension):
        score = calculate_best_buddies_score(grid, costs)
        best_grid = grid
        best_score = score

    for i in range(100):
        try:
            grid = solve_single_run(pieces, dimension, costs, use_noise=True)
            if grid is not None and grid.shape == (dimension, dimension):
                score = calculate_best_buddies_score(grid, costs)
                if score > best_score:
                    best_score = score
                    best_grid = grid
        except:
            continue

    return best_grid


def process_file(path, fname, out_folder, dimension):
    img = cv2.imread(path)
    if img is None: return
    #img = smart_crop(img) 
    h, w, _ = img.shape
    piece_height, piece_width = h // dimension, w // dimension
    
    pieces = []
    for r in range(dimension):
        for c in range(dimension):
            y, x = r * piece_height, c * piece_width
            pieces.append(img[y:y+piece_height, x:x+piece_width])

    final_map = solve_puzzle(pieces, dimension)
        
    if final_map is None:
        return

    r_lim, c_lim = final_map.shape
    res = np.zeros((r_lim*piece_height, c_lim*piece_width, 3), dtype=np.uint8)
    for r in range(r_lim):
        for c in range(c_lim):
            current_piece = final_map[r, c]
            if current_piece != -1:
                y, x = r*piece_height, c*piece_width
                res[y:y+piece_height, x:x+piece_width] = pieces[current_piece]
                
    cv2.imwrite(os.path.join(out_folder, fname), res)

def run():
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    for folder in SUBFOLDERS:
        src = os.path.join(INPUT_ROOT, folder)
        dst = os.path.join(OUTPUT_ROOT, folder)
        if not os.path.exists(src): print("where")
        if not os.path.exists(dst): os.makedirs(dst)
        
        try: dimension = int(folder.split('x')[0].split('_')[-1])
        except: continue
        
        files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','png'))]
        for f in files:
            process_file(os.path.join(src, f), f, dst, dimension)

if __name__ == "__main__":
    run()
