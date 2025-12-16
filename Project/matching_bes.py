import cv2
import numpy as np
import os
import heapq

# ==========================================
# CONFIGURATION
# ==========================================
#BASE_DIR = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\\preprocessing_results\\2_denoised'
BASE_DIR = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls'
INPUT_ROOT = BASE_DIR
OUTPUT_ROOT = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\Processed_Current_Best'
SUBFOLDERS = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']

#potentoial error

W_COLOR = 1
W_GRADIENT = 0.2
W_EDGE_CONTINUITY = 5

# ==========================================
# 1. HELPERS: CROP & EDGE DETECTION
# ==========================================
def smart_crop(img):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    threshold = 5.0 

    top = 0
    for r in range(h):
        if np.std(gray[r, :]) > threshold: top = r; break

    bottom = h
    for r in range(h-1, -1, -1):
        if np.std(gray[r, :]) > threshold: bottom = r + 1; break

    left = 0
    for c in range(w):
        if np.std(gray[:, c]) > threshold: left = c; break
    right = w
    for c in range(w-1, -1, -1):
        if np.std(gray[:, c]) > threshold: right = c + 1; break

    
    if (right - left) < (w * 0.5) or (bottom - top) < (h * 0.5): return img
    return img[top:bottom, left:right]

def get_canny_edges(img):
    """
    Computes the binary edge map of a piece.
    Used to detect outlines that must continue across pieces.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blur slightly to remove compression noise
    # may remove it
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # Thresholds: 50/150 are standard for detecting strong lines
    edges = cv2.Canny(gray, 50, 200)
    return edges

# ==========================================
# 2. MATCHING METRIC (COLOR + EDGES)
# ==========================================

#older BEST
def get_match_cost(piece_a, piece_b, edge_a_map, edge_b_map, relation):
    # Relation 0: A|B, Relation 1: A/B
    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    if relation == 0: # Left | Right
        # Color info for grad difference
        p_edge_a, p_edge_b = A[:, -1, :], B[:, 0, :]
        p_inner_a = A[:, -2, :] 
        p_inner_b = B[:,1,:]

        # Binary Edges
        bin_a, bin_b = edge_a_map[:, -1], edge_b_map[:, 0]

    else: # Top / Bottom
        # Color info for grad difference
        p_edge_a, p_edge_b = A[-1, :, :], B[0, :, :]
        p_inner_a = A[-2, :, :] #
        p_inner_b = B[1:,:,:]

        # Binary Edges
        bin_a, bin_b = edge_a_map[-1, :], edge_b_map[0, :]

    # 1. Black Penalty (Heuristic for borders) Potential error
    if np.mean(p_edge_a[:, 0]) < 15 and np.mean(p_edge_b[:, 0]) < 15:
        return 100000.0 # high cost if both edges are black

    # 2. LAB Color Difference
    color_diff = np.sum(np.abs(p_edge_a - p_edge_b))

    # 3. Gradient Continuity 
    trend = p_edge_a - p_inner_a
    expected_b = p_edge_a + trend # we expect B to match this
    grad_cost = np.sum(np.abs(  expected_b - p_edge_b))

    # [cite_start]4. EDGE CONTINUITY (New Logic) [cite: 24, 25]
    # If A has an edge (255) and B doesn't (0), that's a "Broken Line".
    # XOR gives us the mismatches. AND gives us the matches.
    
    # Normalize binary maps to 0-1 
    b_a = (bin_a > 0).astype(int)
    b_b = (bin_b > 0).astype(int)
    
    edge_cost = np.sum(np.bitwise_xor(b_a, b_b))
    matches = np.sum(np.bitwise_and(b_a, b_b))
    
    #potential error
    if matches > 0: 
        # Bonus: If lines actually connect, reduce cost significantly
        edge_cost -= (matches * 2.0)

    total_cost=(W_COLOR * color_diff)+(W_GRADIENT * grad_cost)+(W_EDGE_CONTINUITY * edge_cost)
                 
    return total_cost


#old BEST
def get_match_cost_old(piece_a, piece_b, edge_a_map, edge_b_map, relation):
    # Relation 0: A|B, Relation 1: A/B
    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    border_num= 5 #experimental

    if relation == 0: 
        p_edge_a=A[:, -border_num:, :]
        p_edge_b= B[:, :border_num, :]
        p_inner_a=A[:, -(border_num*2):-(border_num), :] 
        p_inner_b=B[:,border_num:border_num*2,:]

        # Binary Edges
        bin_a=edge_a_map[:,  -border_num:] 
        bin_b=edge_b_map[:,  :border_num]

    else: # Top / Bottom
        # Color info for grad difference
        p_edge_a, p_edge_b = A[-border_num:, :, :], B[:border_num, :, :]
        p_inner_a = A[-(border_num*2):-(border_num), :, :] 
        p_inner_b = B[border_num:border_num*2,:,:]

        # Binary Edges
        bin_a= edge_a_map[ -border_num:, :] 
        bin_b = edge_b_map[:border_num, :]


    # LAB Color Difference Euclidean distance
    difference=(p_edge_a-p_edge_b).reshape(-1, 3)
    color_diff = np.mean(np.linalg.norm((difference),axis=1))

    #  Gradient Continuity 
    a_difference = p_edge_a - p_inner_a
    b_difference =  p_edge_b -p_inner_b 
    grad_diff = (a_difference - b_difference).reshape(-1, 3)
    grad_cost = np.mean(np.linalg.norm(grad_diff, axis=1))

    # Normalize binary maps to 0-1 
    b_a = (bin_a > 0).astype(int)
    b_b = (bin_b > 0).astype(int)
    
    edge_cost = np.sum(np.abs(b_a- b_b))
   
 
    total_cost=(W_COLOR * color_diff)+(W_GRADIENT * grad_cost)+(W_EDGE_CONTINUITY * edge_cost)
                 
    return total_cost

#flat penalty
def get_match_cost_flat_penalty(piece_a, piece_b, edge_a_map, edge_b_map, relation, border_num=4):
    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(np.float32)

    if relation == 0:  # A | B
        p_edge_a = A[:, -border_num:, :]
        p_edge_b = B[:, :border_num, :]
        p_inner_a = A[:, -2*border_num:-border_num, :]
        p_inner_b = B[:, border_num:2*border_num, :]
        bin_a = edge_a_map[:, -border_num:] > 0
        bin_b = edge_b_map[:, :border_num] > 0
    else:  # A / B
        p_edge_a = A[-border_num:, :, :]
        p_edge_b = B[:border_num, :, :]
        p_inner_a = A[-2*border_num:-border_num, :, :]
        p_inner_b = B[border_num:2*border_num, :, :]
        bin_a = edge_a_map[-border_num:, :] > 0
        bin_b = edge_b_map[:border_num, :] > 0

    # --------------------------------------------------
    # 1. EDGE CONTINUITY (PRIMARY)
    # --------------------------------------------------
    edge_mismatch = np.sum(bin_a ^ bin_b)
    edge_strength = np.sum(bin_a | bin_b)

    if edge_strength > 0:
        edge_cost = edge_mismatch / edge_strength
    else:
        edge_cost = 0.0  # No edges → handled later

    # --------------------------------------------------
    # 2. COLOR DIFFERENCE (SECONDARY)
    # --------------------------------------------------
    color_diff = np.mean(np.linalg.norm(p_edge_a - p_edge_b, axis=2))

    # --------------------------------------------------
    # 3. GRADIENT MAGNITUDE CONSISTENCY (WEAK)
    # --------------------------------------------------
    grad_a = np.linalg.norm(p_edge_a - p_inner_a, axis=2)
    grad_b = np.linalg.norm(p_edge_b - p_inner_b, axis=2)
    grad_cost = np.mean(np.abs(grad_a - grad_b))

    # --------------------------------------------------
    # 4. FLAT REGION PENALTY (CRITICAL FOR CARTOONS)
    # --------------------------------------------------
    flatness = np.mean(grad_a) + np.mean(grad_b)

    flat_penalty = 0.0
    if flatness < 5.0 and edge_strength == 0:
        # Flat–flat borders are ambiguous → penalize
        flat_penalty = 10.0

    # --------------------------------------------------
    # TOTAL COST
    # --------------------------------------------------
    total_cost = (
        5.0 * edge_cost +
        1.0 * color_diff +
        0.5 * grad_cost +
        flat_penalty
    )

    return total_cost

#newest
def get_match_cost_newest(piece_a, piece_b, edge_a_map, edge_b_map, relation, border=3, p=0.3):
    """
    Cartoon-robust compatibility metric inspired by:
    Pomeranz et al. 'A Fully Automated Square Jigsaw Puzzle Solver'
    """

    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(np.float32)

    if relation == 0:  # A | B
        ea, eb = A[:, -border:], B[:, :border]
        ia, ib = A[:, -2*border:-border], B[:, border:2*border]
        ba = edge_a_map[:, -border:] > 0
        bb = edge_b_map[:, :border] > 0
    else:  # A / B
        ea, eb = A[-border:, :], B[:border, :]
        ia, ib = A[-2*border:-border, :], B[border:2*border, :]
        ba = edge_a_map[-border:, :] > 0
        bb = edge_b_map[:border, :] > 0

    # -------------------------------------------------
    # 1. Fractional Lp color difference
    # -------------------------------------------------
    color_diff = np.mean(np.sum(np.abs(ea - eb) ** p, axis=2))

    # -------------------------------------------------
    # 2. Symmetric boundary prediction (Eq. 6)
    # -------------------------------------------------
    pred_b = 2 * ea - ia
    pred_a = 2 * eb - ib

    grad_cost = (
        np.mean(np.sum(np.abs(pred_b - eb) ** p, axis=2)) +
        np.mean(np.sum(np.abs(pred_a - ea) ** p, axis=2))
    )

    # -------------------------------------------------
    # 3. Edge continuity (additive, cartoon-safe)
    # -------------------------------------------------
    edge_mismatch = np.sum(ba ^ bb)
    edge_match = np.sum(ba & bb)
    edge_cost = edge_mismatch - 2.0 * edge_match

    # -------------------------------------------------
    # 4. Flat-region ambiguity penalty (soft)
    # -------------------------------------------------
    flatness = np.mean(np.abs(ea - ia)) + np.mean(np.abs(eb - ib))
    flat_penalty = 3.0 if flatness < 4.0 and edge_match == 0 else 0.0

    # -------------------------------------------------
    # TOTAL COST (weights tuned for cartoons)
    # -------------------------------------------------
    return (
        1.0 * color_diff +
        0.4 * grad_cost +
        0.6 * edge_cost 
    )





# Solver

# local first fone
def solve_puzzle_confidence(pieces, grid_n):
    num_pieces = len(pieces)
    if num_pieces != grid_n * grid_n: return None

    # Pre-compute edges for all pieces once
    edge_maps = [get_canny_edges(p) for p in pieces]
    
    # Calculate "Activity Score" for seeding (High variance = distinct piece)
    activity_scores = [np.sum(e)/255.0 for e in edge_maps]

    # --- A. Compatibility Matrix ---
    costs = np.full((num_pieces, num_pieces, 4), np.inf)

    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j: continue
            
            # Pass edge maps to the cost function
            c_right = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 0)
            c_down  = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 1)
            
            costs[i, j, 0] = c_right # i Left of j
            costs[j, i, 2] = c_right # j Right of i
            costs[i, j, 1] = c_down  # i Top of j
            costs[j, i, 3] = c_down  # j Bottom of i

    # --- B. Smart Seeding ---
    # Instead of just the lowest cost, look for the lowest cost among "Active" pieces.
    # This prevents the solver from starting in a flat black/sky area.
    
    best_seed_val = np.inf
    seed_pair = (0, 1, 0)
    
    # We filter candidates: Only consider pieces with average or higher activity
    avg_activity = np.mean(activity_scores)
    
    for i in range(num_pieces):
        # Find best Right match
        best_r = np.argmin(costs[i, :, 0])
        val_r = costs[i, best_r, 0]
        
        # Mutual check + Activity bias
        if np.argmin(costs[best_r, :, 2]) == i:
            # Prefer active pieces for the seed
            if activity_scores[i] > avg_activity or activity_scores[best_r] > avg_activity:
                # Artificial boost to score if pieces are active
                if val_r < best_seed_val:
                    best_seed_val = val_r
                    seed_pair = (i, best_r, 0)
                
        # Find best Bottom match
        best_b = np.argmin(costs[i, :, 1])
        val_b = costs[i, best_b, 1]
        
        if np.argmin(costs[best_b, :, 3]) == i:
            if activity_scores[i] > avg_activity or activity_scores[best_b] > avg_activity:
                if val_b < best_seed_val:
                    best_seed_val = val_b
                    seed_pair = (i, best_b, 1)

    # Place seed
    grid_size = grid_n * 3
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    
    sy, sx = grid_n, grid_n
    p1, p2, rel = seed_pair
    grid[sy, sx] = p1
    if rel == 0: grid[sy, sx+1] = p2
    else: grid[sy+1, sx] = p2
    
    placed = {p1, p2}
    
    # --- C. Greedy Fill ---
    while len(placed) < num_pieces:
        candidates = []
        rows, cols = np.where(grid != -1)
        min_r, max_r = np.min(rows), np.max(rows)
        min_c, max_c = np.min(cols), np.max(cols)
        
        # Find open slots
        possible_slots = set()
        for r, c in zip(rows, cols):
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r+dr, c+dc
                if not (0 <= nr < grid_size and 0 <= nc < grid_size): continue
                if grid[nr, nc] != -1: continue 
                
                h_span = max(max_r, nr) - min(min_r, nr) + 1
                w_span = max(max_c, nc) - min(min_c, nc) + 1
                if h_span > grid_n or w_span > grid_n: continue
                
                possible_slots.add((nr, nc))

        best_move = None
        best_move_ratio = 1.0
        
        for (nr, nc) in possible_slots:
            slot_costs = []
            
            for pid in range(num_pieces):
                if pid in placed: continue
                
                total_cost = 0
                neighbors_found = 0
                
                # Check neighbors (Top, Bottom, Left, Right)
                # Note: Costs index: 0=Left->Right, 1=Top->Bottom, 2=Right->Left, 3=Bottom->Top
                if grid[nr-1, nc] != -1: # Top Neighbor
                    neighbors_found += 1
                    total_cost += costs[grid[nr-1, nc], pid, 1] 
                if grid[nr+1, nc] != -1: # Bottom Neighbor
                    neighbors_found += 1
                    total_cost += costs[grid[nr+1, nc], pid, 3]
                if grid[nr, nc-1] != -1: # Left Neighbor
                    neighbors_found += 1
                    total_cost += costs[grid[nr, nc-1], pid, 0]
                if grid[nr, nc+1] != -1: # Right Neighbor
                    neighbors_found += 1
                    total_cost += costs[grid[nr, nc+1], pid, 2]
                
                if neighbors_found > 0:
                    # Average the cost
                    slot_costs.append((total_cost / neighbors_found, pid))
            
            if not slot_costs: continue
            slot_costs.sort(key=lambda x: x[0])
            
            # Confidence Ratio
            best_match_cost = slot_costs[0][0]
            best_pid = slot_costs[0][1]
            
            if len(slot_costs) > 1:
                second_best = slot_costs[1][0]
                ratio = best_match_cost / (second_best + 1e-5)
            else:
                ratio = 0.0
                
            if ratio < best_move_ratio:
                best_move_ratio = ratio
                best_move = (nr, nc, best_pid)

        if best_move:
            grid[best_move[0], best_move[1]] = best_move[2]
            placed.add(best_move[2])
        else:
            break

    rows = np.any(grid != -1, axis=1)
    cols = np.any(grid != -1, axis=0)
    return grid[rows][:, cols]


#global second one
def solve_puzzle_confidence_global(pieces, grid_n):
    #greedy favors overconfidence

    num_pieces = len(pieces)
    if num_pieces != grid_n * grid_n: return None

    edge_maps = [get_canny_edges(p) for p in pieces] 
    
    activity_scores = [np.sum(e)/255.0 for e in edge_maps]

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

    confidence = np.full_like(costs, np.inf)
    for i in range(num_pieces):
        for d in range(4):
            sorted_costs = np.sort(costs[i, :, d])
            if sorted_costs[1] > 0:
                confidence[i, :, d] = costs[i, :, d] / sorted_costs[1]

    best_conf= np.inf
    seed_pair = None
    
 
    #avg_activity = np.mean(activity_scores)
    
    
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j: continue
            for d in [0,1]:  
                conf = confidence[i, j, d]
                if conf < best_conf:
                    best_conf = conf
                    seed_pair = (i, j, d)


    grid_size = grid_n * 3
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    
    sy, sx = grid_n, grid_n
    p1, p2, rel = seed_pair
    grid[sy, sx] = p1
    if rel == 0: grid[sy, sx+1] = p2
    else: grid[sy+1, sx] = p2
    
    placed = {p1, p2}
    
    while len(placed) < num_pieces:

        # STEP 1: Build all possible relations
        relations = []

        for placed_pid in placed:
            r, c = np.where(grid == placed_pid)
            r, c = r[0], c[0]

            for pid in range(num_pieces):
                if pid in placed:
                    continue

                for d, (dr, dc) in enumerate([(0,1),(1,0),(0,-1),(-1,0)]):
                    nr, nc = r + dr, c + dc

                    if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                        continue
                    if grid[nr, nc] != -1:
                        continue

                    # bounding box constraint
                    rows, cols = np.where(grid != -1)
                    min_r, max_r = rows.min(), rows.max()
                    min_c, max_c = cols.min(), cols.max()

                    h_span = max(max_r, nr) - min(min_r, nr) + 1
                    w_span = max(max_c, nc) - min(min_c, nc) + 1
                    if h_span > grid_n or w_span > grid_n:
                        continue

                    conf = confidence[placed_pid, pid, d]
                    relations.append((conf, placed_pid, pid, d, nr, nc))

        # STEP 2: Choose the most confident relation globally
        if not relations:
            break

        relations.sort(key=lambda x: x[0])  # lowest confidence ratio = best

        # STEP 3: Apply the first VALID relation
        placed_this_round = False

        for conf, src, pid, d, nr, nc in relations:
            # Check consistency with existing neighbors
            valid = True
            if d == 0 and grid[nr, nc-1] != src: valid = False
            if d == 2 and grid[nr, nc+1] != src: valid = False
            if d == 1 and grid[nr-1, nc] != src: valid = False
            if d == 3 and grid[nr+1, nc] != src: valid = False

            if not valid:
                continue

            grid[nr, nc] = pid
            placed.add(pid)
            placed_this_round = True
            break

        # STEP 4: Fail safely if no placement possible
        if not placed_this_round:
            break

    rows = np.any(grid != -1, axis=1)
    cols = np.any(grid != -1, axis=0)
    return grid[rows][:, cols]


# local non lock
def solve_puzzle_confidence_nonlock(pieces, grid_n):
    num_pieces = len(pieces)
    if num_pieces != grid_n * grid_n:
        return None

    # ----------------------------------
    # Precompute edges and activity
    # ----------------------------------
    edge_maps = [get_canny_edges(p) for p in pieces]
    activity_scores = [np.sum(e) / 255.0 for e in edge_maps]
    avg_activity = np.mean(activity_scores)

    # ----------------------------------
    # Compatibility matrix
    # ----------------------------------
    costs = np.full((num_pieces, num_pieces, 4), np.inf)

    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j:
                continue
            cr = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 0)
            cd = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 1)

            costs[i, j, 0] = cr
            costs[j, i, 2] = cr
            costs[i, j, 1] = cd
            costs[j, i, 3] = cd

    # ----------------------------------
    # Smart seed selection
    # ----------------------------------
    best_seed_val = np.inf
    seed_pair = (0, 1, 0)

    for i in range(num_pieces):
        r = np.argmin(costs[i, :, 0])
        if np.argmin(costs[r, :, 2]) == i:
            if activity_scores[i] > avg_activity or activity_scores[r] > avg_activity:
                if costs[i, r, 0] < best_seed_val:
                    best_seed_val = costs[i, r, 0]
                    seed_pair = (i, r, 0)

        b = np.argmin(costs[i, :, 1])
        if np.argmin(costs[b, :, 3]) == i:
            if activity_scores[i] > avg_activity or activity_scores[b] > avg_activity:
                if costs[i, b, 1] < best_seed_val:
                    best_seed_val = costs[i, b, 1]
                    seed_pair = (i, b, 1)

    # ----------------------------------
    # Grid initialization
    # ----------------------------------
    grid_size = grid_n * 3
    grid = np.full((grid_size, grid_size), -1, dtype=int)

    sy = sx = grid_n
    a, b, rel = seed_pair
    grid[sy, sx] = a
    if rel == 0:
        grid[sy, sx + 1] = b
    else:
        grid[sy + 1, sx] = b

    placed = {a, b}

    # ----------------------------------
    # Local greedy fill (FIXED)
    # ----------------------------------
    while len(placed) < num_pieces:
        rows, cols = np.where(grid != -1)
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()

        best_move = None
        best_ratio = 1.0
        best_priority = -1

        possible_slots = set()
        for r, c in zip(rows, cols):
            for dr, dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                    continue
                if grid[nr, nc] != -1:
                    continue

                h = max(max_r, nr) - min(min_r, nr) + 1
                w = max(max_c, nc) - min(min_c, nc) + 1
                if h <= grid_n and w <= grid_n:
                    possible_slots.add((nr, nc))

        for nr, nc in possible_slots:
            slot_costs = []

            neighbors_found = 0
            structural_neighbors = 0

            # Count neighbors ONCE per slot
            if grid[nr-1, nc] != -1:
                neighbors_found += 1
                if activity_scores[grid[nr-1, nc]] > avg_activity:
                    structural_neighbors += 1
            if grid[nr+1, nc] != -1:
                neighbors_found += 1
                if activity_scores[grid[nr+1, nc]] > avg_activity:
                    structural_neighbors += 1
            if grid[nr, nc-1] != -1:
                neighbors_found += 1
                if activity_scores[grid[nr, nc-1]] > avg_activity:
                    structural_neighbors += 1
            if grid[nr, nc+1] != -1:
                neighbors_found += 1
                if activity_scores[grid[nr, nc+1]] > avg_activity:
                    structural_neighbors += 1

            for pid in range(num_pieces):
                if pid in placed:
                    continue

                total = 0
                count = 0

                if grid[nr-1, nc] != -1:
                    total += costs[grid[nr-1, nc], pid, 1]
                    count += 1
                if grid[nr+1, nc] != -1:
                    total += costs[grid[nr+1, nc], pid, 3]
                    count += 1
                if grid[nr, nc-1] != -1:
                    total += costs[grid[nr, nc-1], pid, 0]
                    count += 1
                if grid[nr, nc+1] != -1:
                    total += costs[grid[nr, nc+1], pid, 2]
                    count += 1

                if count > 0:
                    slot_costs.append((total / count, pid))

            if not slot_costs:
                continue

            slot_costs.sort()
            best_cost, best_pid = slot_costs[0]

            if len(slot_costs) > 1:
                ratio = best_cost / (slot_costs[1][0] + 1e-5)
            else:
                ratio = 0.0

            # Flat-region safety (EARLY ONLY)
            if neighbors_found == 1 and structural_neighbors == 0 and len(placed) < num_pieces * 0.4:
                continue

            if ratio > 0.85:
                continue

            priority = neighbors_found + 0.5 * structural_neighbors

            if (
                best_move is None or
                priority > best_priority or
                (priority == best_priority and ratio < best_ratio)
            ):
                best_move = (nr, nc, best_pid)
                best_ratio = ratio
                best_priority = priority

        if best_move:
            r, c, pid = best_move
            grid[r, c] = pid
            placed.add(pid)
        else:
            break

    rows = np.any(grid != -1, axis=1)
    cols = np.any(grid != -1, axis=0)
    return grid[rows][:, cols]

def solve_puzzle_confidence_newest(pieces, grid_n):

    num_pieces = len(pieces)
    if num_pieces != grid_n * grid_n:
        return None

    # --------------------------------------------------
    # 1. Precompute edges + costs
    # --------------------------------------------------
    edge_maps = [get_canny_edges(p) for p in pieces]

    costs = np.full((num_pieces, num_pieces, 4), np.inf)
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j:
                continue
            cr = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 0)
            cd = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 1)
            costs[i, j, 0] = cr
            costs[j, i, 2] = cr
            costs[i, j, 1] = cd
            costs[j, i, 3] = cd

    # --------------------------------------------------
    # 2. Directional confidence (paper)
    # --------------------------------------------------
    confidence = np.full_like(costs, np.inf)
    for i in range(num_pieces):
        for d in range(4):
            sorted_costs = np.sort(costs[i, :, d])
            if sorted_costs[1] > 0:
                confidence[i, :, d] = costs[i, :, d] / sorted_costs[1]

    # --------------------------------------------------
    # 3. Global relation queue (THIS IS THE KEY)
    # --------------------------------------------------
    pq = []
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j:
                continue
            for d in (0, 1):  # right, down (avoid duplicates)
                heapq.heappush(pq, (confidence[i, j, d], i, j, d))

    # --------------------------------------------------
    # 4. Placement state (graph-based)
    # --------------------------------------------------
    pos = {}          # piece -> (r, c)
    grid = {}         # (r, c) -> piece

    def can_place(pid, r, c):
        if (r, c) in grid:
            return False

        for dr, dc, dcheck in [(0,1,0),(1,0,1),(0,-1,2),(-1,0,3)]:
            nb = grid.get((r+dr, c+dc))
            if nb is not None:
                # MUST be mutually compatible
                if costs[nb, pid, dcheck] > costs[nb].min() * 1.3:
                    return False
        return True


    # --------------------------------------------------
    # 5. Global greedy placement
    # --------------------------------------------------
    while pq and len(pos) < num_pieces:
        conf, a, b, d = heapq.heappop(pq)

        if a in pos and b in pos:
            continue

        if a not in pos and b not in pos:
            pos[a] = (0, 0)
            grid[(0, 0)] = a

        if a in pos and b not in pos:
            r, c = pos[a]
            dr, dc = [(0,1),(1,0),(0,-1),(-1,0)][d]
            nr, nc = r + dr, c + dc
            if can_place(b, nr, nc):
                pos[b] = (nr, nc)
                grid[(nr, nc)] = b

        elif b in pos and a not in pos:
            r, c = pos[b]
            dr, dc = [(0,-1),(-1,0),(0,1),(1,0)][d]
            nr, nc = r + dr, c + dc
            if can_place(a, nr, nc):
                pos[a] = (nr, nc)
                grid[(nr, nc)] = a

    # --------------------------------------------------
# 5b. FORCE PLACE REMAINING PIECES (CRITICAL)
# --------------------------------------------------
    unplaced = [i for i in range(num_pieces) if i not in pos]

    for pid in unplaced:
        best = None
        best_score = np.inf

        for (r, c) in list(grid.keys()):
            for d, (dr, dc) in enumerate([(0,1),(1,0),(0,-1),(-1,0)]):
                nr, nc = r + dr, c + dc
                if (nr, nc) in grid:
                    continue

                if not can_place(pid, nr, nc):
                    continue

                score = 0
                count = 0
                for dr2, dc2, dcheck in [(0,1,0),(1,0,1),(0,-1,2),(-1,0,3)]:
                    nb = grid.get((nr+dr2, nc+dc2))
                    if nb is not None:
                        score += costs[nb, pid, dcheck]
                        count += 1

                if count > 0:
                    score /= count
                    if score < best_score:
                        best_score = score
                        best = (nr, nc)

        if best is not None:
            pos[pid] = best
            grid[best] = pid


    # --------------------------------------------------
    # 6. Normalize to grid
    # --------------------------------------------------
    rows = [r for r, c in pos.values()]
    cols = [c for r, c in pos.values()]

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    h = max_r - min_r + 1
    w = max_c - min_c + 1

    # HARD SAFETY: reject impossible shapes
    if h > grid_n or w > grid_n:
        # force compact shift (best effort)
        # center the bounding box
        pass

    final = np.full((h, w), -1, dtype=int)

    for pid, (r, c) in pos.items():
        final[r - min_r, c - min_c] = pid

    return final

# ==========================================
# 4. RUNNER
# ==========================================
def process_file(path, fname, out_folder, grid_n):
    img = cv2.imread(path)
    if img is None: return
    img = smart_crop(img) # potential error

    h, w, _ = img.shape
    ph=h // grid_n
    pw =w // grid_n
    
    #splitting
    pieces = []
    for r in range(grid_n):
        for c in range(grid_n):
            y, x = r * ph, c * pw
            pieces.append(img[y:y+ph, x:x+pw])


            
    final_map = solve_puzzle_confidence(pieces, grid_n)
        
    if final_map is None:
        print(f"   [!] Solver failed: {fname}")
        return

    # Stitch
    r_lim, c_lim = final_map.shape
    r_lim = min(r_lim, grid_n)
    c_lim = min(c_lim, grid_n)
    
    res = np.zeros((r_lim*ph, c_lim*pw, 3), dtype=np.uint8)
    for r in range(r_lim):
        for c in range(c_lim):
            pid = final_map[r, c]
            if pid != -1:
                y, x = r*ph, c*pw
                res[y:y+ph, x:x+pw] = pieces[pid]
                
    cv2.imwrite(os.path.join(out_folder, fname), res)

def run():
    if not os.path.exists(OUTPUT_ROOT): os.makedirs(OUTPUT_ROOT)
    for folder in SUBFOLDERS:
        src = os.path.join(INPUT_ROOT, folder)
        dst = os.path.join(OUTPUT_ROOT, folder)
        if not os.path.exists(src): continue
        if not os.path.exists(dst): os.makedirs(dst)
        
        try: grid_n = int(folder.split('x')[0].split('_')[-1])
        except: continue
        
        print(f"Processing {folder} (Grid: {grid_n}x{grid_n})...")
        files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.png'))]
        for i, f in enumerate(files):
            process_file(os.path.join(src, f), f, dst, grid_n)
            if (i+1) % 10 == 0: print(f"  {i+1}/{len(files)}")

if __name__ == "__main__":
    run()