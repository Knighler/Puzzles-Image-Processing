import cv2
import numpy as np
import os
import copy

BASE_DIR = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls'
INPUT_ROOT = BASE_DIR
OUTPUT_ROOT = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\Paper_Results_Updated'
SUBFOLDERS = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']


W_COLOR = 1
W_GRADIENT = 0.2
W_EDGE_CONTINUITY = 15 


def smart_crop(img):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    threshold = 5.0 
    top = 0; bottom = h; left = 0; right = w
    for r in range(h):
        if np.std(gray[r, :]) > threshold: top = r; break
    for r in range(h-1, -1, -1):
        if np.std(gray[r, :]) > threshold: bottom = r + 1; break
    for c in range(w):
        if np.std(gray[:, c]) > threshold: left = c; break
    for c in range(w-1, -1, -1):
        if np.std(gray[:, c]) > threshold: right = c + 1; break
    if (right - left) < (w * 0.5) or (bottom - top) < (h * 0.5): return img
    return img[top:bottom, left:right]

def get_canny_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 50, 150)
    return edges


def get_match_cost(piece_a, piece_b, edge_a_map, edge_b_map, relation):
    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    if relation == 0: 
        p_edge_a, p_edge_b = A[:, -1, :], B[:, 0, :]
        p_inner_a = A[:, -2, :] 
        bin_a, bin_b = edge_a_map[:, -1], edge_b_map[:, 0]
    else: 
        p_edge_a, p_edge_b = A[-1, :, :], B[0, :, :]
        p_inner_a = A[-2, :, :] 
        bin_a, bin_b = edge_a_map[-1, :], edge_b_map[0, :]

    if np.mean(p_edge_a[:, 0]) < 15 and np.mean(p_edge_b[:, 0]) < 15:
        return 100000.0 

    color_diff = np.sum(np.abs(p_edge_a - p_edge_b))
    trend = p_edge_a - p_inner_a
    expected_b = p_edge_a + trend
    grad_cost = np.sum(np.abs(expected_b - p_edge_b))

    b_a = (bin_a > 0).astype(int)
    b_b = (bin_b > 0).astype(int)
    edge_mismatch = np.sum(np.bitwise_xor(b_a, b_b))
    matches = np.sum(np.bitwise_and(b_a, b_b))
    edge_cost = edge_mismatch
    if matches > 0: 
        edge_cost -= (matches * 5.0)

    total_cost = (W_COLOR * color_diff) + (W_GRADIENT * grad_cost) + (W_EDGE_CONTINUITY * edge_cost)
    return total_cost


def calculate_best_buddies_score(grid, costs):
    h, w = grid.shape
    buddies = 0
    total_joints = 0
    
    for r in range(h):
        for c in range(w):
            pid = grid[r, c]
            if pid == -1: continue
            
            # Check Right
            if c + 1 < w:
                nid = grid[r, c+1]
                if nid != -1:
                    total_joints += 1
                    best_for_pid = np.argmin(costs[pid, :, 0])
                    best_for_nid = np.argmin(costs[nid, :, 2])
                    if best_for_pid == nid and best_for_nid == pid: buddies += 1

            # Check Down
            if r + 1 < h:
                nid = grid[r+1, c]
                if nid != -1:
                    total_joints += 1
                    best_for_pid = np.argmin(costs[pid, :, 1])
                    best_for_nid = np.argmin(costs[nid, :, 3])
                    if best_for_pid == nid and best_for_nid == pid: buddies += 1
                        
    return buddies / total_joints if total_joints > 0 else 0


def solve_single_run(pieces, grid_n, costs, initial_grid=None, use_noise=False):
    num_pieces = len(pieces)
    
 
    run_costs = costs.copy()
    if use_noise:
        noise = np.random.normal(0, 0.05, run_costs.shape) * run_costs
        run_costs += noise

 
    grid_size = grid_n * 3
    grid = np.full((grid_size, grid_size), -1, dtype=int)
    placed = set()

    if initial_grid is not None:

        rows, cols = np.where(initial_grid != -1)
        h_seg = rows.max() - rows.min() + 1
        w_seg = cols.max() - cols.min() + 1
        
        offset_y = (grid_size - h_seg) // 2 - rows.min()
        offset_x = (grid_size - w_seg) // 2 - cols.min()
        
        for r, c in zip(rows, cols):
            new_r, new_c = r + offset_y, c + offset_x
            grid[new_r, new_c] = initial_grid[r, c]
            placed.add(initial_grid[r, c])
    else:

        best_seed_val = np.inf
        seed_pair = (0, 1, 0)
        edge_maps = [get_canny_edges(p) for p in pieces] 
        activity = [np.sum(e)/255.0 for e in edge_maps]
        avg_act = np.mean(activity)

        for i in range(num_pieces):
            if activity[i] < avg_act: continue
            best_r = np.argmin(run_costs[i, :, 0])
            if np.argmin(run_costs[best_r, :, 2]) == i:
                if run_costs[i, best_r, 0] < best_seed_val:
                    best_seed_val = run_costs[i, best_r, 0]; seed_pair = (i, best_r, 0)
            best_b = np.argmin(run_costs[i, :, 1])
            if np.argmin(run_costs[best_b, :, 3]) == i:
                if run_costs[i, best_b, 1] < best_seed_val:
                    best_seed_val = run_costs[i, best_b, 1]; seed_pair = (i, best_b, 1)

        sy, sx = grid_n, grid_n
        p1, p2, rel = seed_pair
        grid[sy, sx] = p1
        if rel == 0: grid[sy, sx+1] = p2
        else: grid[sy+1, sx] = p2
        placed = {p1, p2}


    while len(placed) < num_pieces:
        rows, cols = np.where(grid != -1)
        possible_slots = set()
        
        for r, c in zip(rows, cols):
            for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                nr, nc = r+dr, c+dc
                if grid[nr, nc] == -1:
                    h_span = max(rows.max(), nr) - min(rows.min(), nr) + 1
                    w_span = max(cols.max(), nc) - min(cols.min(), nc) + 1
                    if h_span <= grid_n and w_span <= grid_n:
                        possible_slots.add((nr, nc))

        best_move = None
        best_conf_score = -1.0 

        for (nr, nc) in possible_slots:
            slot_costs = []
            for pid in range(num_pieces):
                if pid in placed: continue
                current_cost = 0; count = 0
                if grid[nr-1, nc] != -1: current_cost += run_costs[grid[nr-1, nc], pid, 1]; count += 1
                if grid[nr+1, nc] != -1: current_cost += run_costs[grid[nr+1, nc], pid, 3]; count += 1
                if grid[nr, nc-1] != -1: current_cost += run_costs[grid[nr, nc-1], pid, 0]; count += 1
                if grid[nr, nc+1] != -1: current_cost += run_costs[grid[nr, nc+1], pid, 2]; count += 1
                
                if count > 0: slot_costs.append((current_cost/count, pid))

            if not slot_costs: continue
            slot_costs.sort(key=lambda x: x[0])
            
            best_c, best_p = slot_costs[0]
            if len(slot_costs) > 1:
                ratio = slot_costs[1][0] / (best_c + 1e-5)
            else:
                ratio = 100.0
            
            if ratio > best_conf_score:
                best_conf_score = ratio
                best_move = (nr, nc, best_p)

        if best_move:
            grid[best_move[0], best_move[1]] = best_move[2]
            placed.add(best_move[2])
        else:
            break


    rows = np.any(grid != -1, axis=1)
    cols = np.any(grid != -1, axis=0)
    return grid[rows][:, cols]


def find_largest_consistent_segment(grid, costs):

    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    max_segment = []
    
    for r in range(h):
        for c in range(w):
            if grid[r, c] == -1 or visited[r, c]: continue
            
            # BFS to find segment
            queue = [(r, c)]
            current_segment = []
            visited[r, c] = True
            
            while queue:
                curr_r, curr_c = queue.pop(0)
                pid = grid[curr_r, curr_c]
                current_segment.append((curr_r, curr_c, pid))
                
                # Check neighbors for Best Buddy Status
                # Right
                if curr_c + 1 < w and grid[curr_r, curr_c+1] != -1:
                    nid = grid[curr_r, curr_c+1]
                    
                    bp = np.argmin(costs[pid, :, 0])
                    bn = np.argmin(costs[nid, :, 2])
                    if bp == nid and bn == pid and not visited[curr_r, curr_c+1]:
                        visited[curr_r, curr_c+1] = True
                        queue.append((curr_r, curr_c+1))
                
                # Down
                if curr_r + 1 < h and grid[curr_r+1, curr_c] != -1:
                    nid = grid[curr_r+1, curr_c]
                    bp = np.argmin(costs[pid, :, 1])
                    bn = np.argmin(costs[nid, :, 3])
                    if bp == nid and bn == pid and not visited[curr_r+1, curr_c]:
                        visited[curr_r+1, curr_c] = True
                        queue.append((curr_r+1, curr_c))
            
            if len(current_segment) > len(max_segment):
                max_segment = current_segment
                
    # Create a grid containing max segment
    if not max_segment: return None
    
    # Normalize coordinates
    rs = [x[0] for x in max_segment]; cs = [x[1] for x in max_segment]
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)
    
    seg_h = max_r - min_r + 1
    seg_w = max_c - min_c + 1
    
    segment_grid = np.full((seg_h, seg_w), -1, dtype=int)
    for (r, c, pid) in max_segment:
        segment_grid[r - min_r, c - min_c] = pid
        
    return segment_grid

def run_shifter_phase(pieces, grid_n, initial_best_grid, costs):

    current_grid = initial_best_grid
    current_score = calculate_best_buddies_score(current_grid, costs)

    
    for i in range(10): 
        segment = find_largest_consistent_segment(current_grid, costs)
        if segment is None: break
        count_placed = np.sum(segment != -1)
        if count_placed == len(pieces): 
            break
            
        
        new_grid = solve_single_run(pieces, grid_n, costs, initial_grid=segment, use_noise=False)
        
        if new_grid is None or new_grid.shape[0] > grid_n or new_grid.shape[1] > grid_n:
            continue

        new_score = calculate_best_buddies_score(new_grid, costs)
        
        if new_score > current_score:
            print(f"    [Shifter] Improved! {current_score:.2f} -> {new_score:.2f} (Segment size: {count_placed})")
            current_grid = new_grid
            current_score = new_score
        else:
            print("    [Shifter] No improvement.")
            break
            
    return current_grid


def solve_puzzle_hybrid(pieces, grid_n):
    num_pieces = len(pieces)
    edge_maps = [get_canny_edges(p) for p in pieces]
    
    print("  Calculating Cost Matrix...")
    costs = np.full((num_pieces, num_pieces, 4), np.inf)
    for i in range(num_pieces):
        for j in range(num_pieces):
            if i == j: continue
            c_right = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 0)
            c_down  = get_match_cost(pieces[i], pieces[j], edge_maps[i], edge_maps[j], 1)
            costs[i, j, 0] = c_right; costs[j, i, 2] = c_right
            costs[i, j, 1] = c_down;  costs[j, i, 3] = c_down

   
    best_grid = None
    best_score = -1
    
    print("  Phase 1: Stochastic Search...")
    
    for run in range(100):
        use_noise = (run > 0)
        try:
            grid = solve_single_run(pieces, grid_n, costs, initial_grid=None, use_noise=use_noise)
            if grid is not None and grid.shape == (grid_n, grid_n):
                score = calculate_best_buddies_score(grid, costs)
                if score > best_score:
                    best_score = score
                    best_grid = grid
                    print(f"    Run {run}: Found new best score {score:.2f}")
        except: continue

    if best_grid is None:
        print("  Failed to find valid grid.")
        return None

 
    print("  Phase 2: Shifter Optimization...")
    final_grid = run_shifter_phase(pieces, grid_n, best_grid, costs)
    
    return final_grid


def process_file(path, fname, out_folder, grid_n):
    img = cv2.imread(path)
    if img is None: return
    #img = smart_crop(img) 
    h, w, _ = img.shape
    ph, pw = h // grid_n, w // grid_n
    
    pieces = []
    for r in range(grid_n):
        for c in range(grid_n):
            y, x = r * ph, c * pw
            pieces.append(img[y:y+ph, x:x+pw])

    final_map = solve_puzzle_hybrid(pieces, grid_n)
        
    if final_map is None:
        print(f"   [!] Solver failed: {fname}")
        return

    r_lim, c_lim = final_map.shape
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
        
        print(f"\nProcessing {folder} (Grid: {grid_n}x{grid_n})...")
        files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg','.png'))]
        for f in files:
            print(f" File: {f}")
            process_file(os.path.join(src, f), f, dst, grid_n)

if __name__ == "__main__":
    run()