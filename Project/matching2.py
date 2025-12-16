import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls"
INPUT_ROOT = BASE_DIR
OUTPUT_ROOT = os.path.join(BASE_DIR, "Processed")
SUBFOLDERS = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']

# ==========================================
# 1. SMART CROP (CRITICAL FOR BLACK BORDERS)
# ==========================================
def smart_crop(img):
    """
    Aggressively removes black letterboxing borders.
    """
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Tolerance for "Black". Increased to 40 to catch compression artifacts.
    limit = 40 

    # 1. Top
    top = 0
    for r in range(h):
        if np.mean(gray[r, :]) > limit:
            top = r
            break
    # 2. Bottom
    bottom = h
    for r in range(h-1, -1, -1):
        if np.mean(gray[r, :]) > limit:
            bottom = r + 1
            break
    # 3. Left
    left = 0
    for c in range(w):
        if np.mean(gray[:, c]) > limit:
            left = c
            break
    # 4. Right
    right = w
    for c in range(w-1, -1, -1):
        if np.mean(gray[:, c]) > limit:
            right = c + 1
            break

    # Safety: if we cropped too much (image is basically empty), return original
    if (right - left) < (w * 0.5) or (bottom - top) < (h * 0.5):
        return img

    return img[top:bottom, left:right]

# ==========================================
# 2. MATCHING METRICS (LAB + GRADIENT)
# ==========================================
def get_match_cost(piece_a, piece_b, relation):
    """
    Calculates how well two pieces fit. 
    Relation 0: A | B (A is Left)
    Relation 1: A / B (A is Top)
    """
    # Convert to LAB (Human perceptual color space)
    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(float)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(float)
    
    if relation == 0: # A Left, B Right
        # Pixels at the boundary
        edge_a = A[:, -1, :] 
        edge_b = B[:, 0, :]
        # Pixels one step inward (for gradient)
        inner_a = A[:, -2, :] 
        inner_b = B[:, 1, :]
    else: # A Top, B Bottom
        edge_a = A[-1, :, :]
        edge_b = B[0, :, :]
        inner_a = A[-2, :, :]
        inner_b = B[1, :, :]

    # 1. Color Difference
    diff = np.abs(edge_a - edge_b)
    color_cost = np.sum(diff)

    # 2. Gradient Continuity
    # The "slope" of color change should be consistent across the seam
    trend_a = edge_a - inner_a
    # We expect the transition edge_a -> edge_b to follow trend_a
    # So expected_b = edge_a + trend_a
    # Error = |expected_b - actual_b|
    #       = |(edge_a + trend_a) - edge_b|
    #       = |trend_a - (edge_b - edge_a)|
    
    seam_derivative = edge_b - edge_a
    grad_cost = np.sum(np.abs(trend_a - seam_derivative))

    return color_cost + (2.5 * grad_cost)

# ==========================================
# 3. CONFIDENCE-BASED SOLVER
# ==========================================
def solve_puzzle_confidence(pieces, grid_n):
    num_pieces = len(pieces)
    if num_pieces != grid_n * grid_n: return None

    # --- A. Pre-Calculate All Costs ---
    # costs[i, j, rel] = cost of putting piece i related to j by rel
    # rel 0: i is LEFT of j
    # rel 1: i is TOP of j
    # rel 2: i is RIGHT of j
    # rel 3: i is BOTTOM of j
    costs = np.full((num_pieces, num_pieces, 4), np.inf)

    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j: continue
            
            # Calculate base costs
            c_right = get_match_cost(pieces[i], pieces[j], 0) # i | j
            c_down  = get_match_cost(pieces[i], pieces[j], 1) # i / j
            
            costs[i, j, 0] = c_right # i Left of j
            costs[j, i, 2] = c_right # j Right of i
            costs[i, j, 1] = c_down  # i Top of j
            costs[j, i, 3] = c_down  # j Bottom of i

    # --- B. Initialize Grid & Confidence Logic ---
    grid_size = grid_n * 3
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    
    # Find the ABSOLUTE best match in the entire set to start
    best_seed_val = np.inf
    seed_pair = (0, 1, 0) # p1, p2, rel
    
    for i in range(num_pieces):
        # Find best Right match
        best_r = np.argmin(costs[i, :, 0])
        val_r = costs[i, best_r, 0]
        
        # Mutual check: Does best_r prefer i on its Left?
        if np.argmin(costs[best_r, :, 2]) == i:
            if val_r < best_seed_val:
                best_seed_val = val_r
                seed_pair = (i, best_r, 0)
                
        # Find best Bottom match
        best_b = np.argmin(costs[i, :, 1])
        val_b = costs[i, best_b, 1]
        
        # Mutual check
        if np.argmin(costs[best_b, :, 3]) == i:
            if val_b < best_seed_val:
                best_seed_val = val_b
                seed_pair = (i, best_b, 1)

    # Place seed
    sy, sx = grid_n, grid_n
    p1, p2, rel = seed_pair
    grid[sy, sx] = p1
    if rel == 0: grid[sy, sx+1] = p2
    else: grid[sy+1, sx] = p2
    
    placed = {p1, p2}
    
    # --- C. The Loop: Pick the "Most Obvious" Move ---
    while len(placed) < num_pieces:
        candidates = []
        
        rows, cols = np.where(grid != -1)
        min_r, max_r = np.min(rows), np.max(rows)
        min_c, max_c = np.min(cols), np.max(cols)
        
        # 1. Identify all empty slots next to existing pieces
        possible_slots = set()
        for r, c in zip(rows, cols):
            # Check 4 neighbors
            for dr, dc, rel_from_placed in [(0,1,0), (1,0,1), (0,-1,2), (-1,0,3)]:
                nr, nc = r+dr, c+dc
                
                # Bounds check
                if not (0 <= nr < grid_size and 0 <= nc < grid_size): continue
                if grid[nr, nc] != -1: continue # Already filled
                
                # Shape check (Prevent growing larger than N x N)
                h_span = max(max_r, nr) - min(min_r, nr) + 1
                w_span = max(max_c, nc) - min(min_c, nc) + 1
                if h_span > grid_n or w_span > grid_n: continue
                
                possible_slots.add((nr, nc))

        # 2. Evaluate every possible piece for every possible slot
        best_move = None
        best_move_ratio = 1.0 # 0.0 is perfect confidence, 1.0 is pure guess
        
        for (nr, nc) in possible_slots:
            # We need to find the best piece for this specific slot (nr, nc)
            # This slot might have multiple neighbors (e.g., a Top and a Left neighbor)
            
            slot_costs = []
            
            for pid in range(num_pieces):
                if pid in placed: continue
                
                total_cost = 0
                neighbors_found = 0
                
                # Check all 4 directions around this slot
                # Top Neighbor?
                if grid[nr-1, nc] != -1:
                    neighbors_found += 1
                    # Neighbor is Top of pid
                    total_cost += costs[grid[nr-1, nc], pid, 1] 
                    
                # Bottom Neighbor?
                if grid[nr+1, nc] != -1:
                    neighbors_found += 1
                    # Neighbor is Bottom of pid
                    total_cost += costs[grid[nr+1, nc], pid, 3]

                # Left Neighbor?
                if grid[nr, nc-1] != -1:
                    neighbors_found += 1
                    # Neighbor is Left of pid
                    total_cost += costs[grid[nr, nc-1], pid, 0]

                # Right Neighbor?
                if grid[nr, nc+1] != -1:
                    neighbors_found += 1
                    # Neighbor is Right of pid
                    total_cost += costs[grid[nr, nc+1], pid, 2]
                
                if neighbors_found > 0:
                    slot_costs.append((total_cost, pid))
            
            if not slot_costs: continue

            # Sort pieces by how well they fit this slot
            slot_costs.sort(key=lambda x: x[0])
            
            best_match_cost = slot_costs[0][0]
            best_pid = slot_costs[0][1]
            
            # CONFIDENCE CALCULATION
            # If we have a second option, compare best vs second best.
            # If best is 100 and second is 102, confidence is low (Ratio ~ 0.98)
            # If best is 100 and second is 1000, confidence is high (Ratio ~ 0.1)
            
            if len(slot_costs) > 1:
                second_best_cost = slot_costs[1][0]
                # Avoid division by zero
                if second_best_cost < 1e-5: ratio = 1.0
                else: ratio = best_match_cost / second_best_cost
            else:
                # Only one piece fits? Maximum confidence!
                ratio = 0.0
                
            # We want the move with the LOWEST ratio (Highest distinctiveness)
            if ratio < best_move_ratio:
                best_move_ratio = ratio
                best_move = (nr, nc, best_pid)

        # 3. Execute the most confident move
        if best_move:
            grid[best_move[0], best_move[1]] = best_move[2]
            placed.add(best_move[2])
        else:
            # Dead end? This shouldn't happen unless puzzle is broken
            break

    # Extract result
    rows = np.any(grid != -1, axis=1)
    cols = np.any(grid != -1, axis=0)
    return grid[rows][:, cols]

# ==========================================
# 4. RUNNER
# ==========================================
def process_file(path, fname, out_folder, grid_n):
    # 1. Load & Smart Crop
    img = cv2.imread(path)
    if img is None: return
    img = smart_crop(img)

    h, w, _ = img.shape
    ph, pw = h // grid_n, w // grid_n
    
    # 2. Slice
    pieces = []
    for r in range(grid_n):
        for c in range(grid_n):
            y, x = r * ph, c * pw
            pieces.append(img[y:y+ph, x:x+pw])
            
    # 3. Solve (Using Confidence Logic for all sizes)
    final_map = solve_puzzle_confidence(pieces, grid_n)
        
    if final_map is None:
        print(f"   [!] Solver failed: {fname}")
        return

    # 4. Stitch
    r_lim, c_lim = final_map.shape
    # Ensure we don't crash if solver returned odd shape
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