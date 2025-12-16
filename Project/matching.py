import cv2
import numpy as np
import os
import heapq

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = r"E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls"
INPUT_ROOT = BASE_DIR
OUTPUT_ROOT = os.path.join(BASE_DIR, "Processed")
SUBFOLDERS = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']

# ==========================================
# 1. HELPER: SMART CROP
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

# ==========================================
# 2. MATCHING METRIC
# ==========================================
def get_match_cost(piece_a, piece_b, relation):
    # Relation 0: A|B, Relation 1: A/B
    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    if relation == 0:
        edge_a, edge_b = A[:, -1, :], B[:, 0, :]
        inner_a = A[:, -2, :]
    else:
        edge_a, edge_b = A[-1, :, :], B[0, :, :]
        inner_a = A[-2, :, :]

    # Black Penalty
    if np.mean(edge_a[:, 0]) < 15 and np.mean(edge_b[:, 0]) < 15:
        return 100000.0

    diff = np.sum(np.abs(edge_a - edge_b))
    trend = edge_a - inner_a
    grad_cost = np.sum(np.abs((edge_a + trend) - edge_b))
    
    return diff + (2.0 * grad_cost)

# ==========================================
# 3. CORE LOGIC CLASSES
# ==========================================

class PuzzleSolver:
    def __init__(self, pieces, grid_n):
        self.pieces = pieces
        self.grid_n = grid_n
        self.num_pieces = len(pieces)
        self.grid_size = grid_n * 3
        
        # Precompute Costs
        self.costs = np.full((self.num_pieces, self.num_pieces, 4), np.inf)
        self.buddies = set() # Stores (p1, p2, rel) if they are Best Buddies
        
        self._calculate_costs()
        self._find_best_buddies()

    def _calculate_costs(self):
        for i in range(self.num_pieces):
            for j in range(self.num_pieces):
                if i == j: continue
                c_right = get_match_cost(self.pieces[i], self.pieces[j], 0)
                c_down = get_match_cost(self.pieces[i], self.pieces[j], 1)
                
                self.costs[i, j, 0] = c_right
                self.costs[j, i, 2] = c_right
                self.costs[i, j, 1] = c_down
                self.costs[j, i, 3] = c_down

    def _find_best_buddies(self):
        """
        Identify pairs that mutually agree they are the best match.
        Used for the Segmentation Phase[cite: 159, 209].
        """
        for i in range(self.num_pieces):
            # Check Right (0)
            best_r = np.argmin(self.costs[i, :, 0])
            if np.argmin(self.costs[best_r, :, 2]) == i:
                self.buddies.add((i, best_r, 0))
                
            # Check Down (1)
            best_d = np.argmin(self.costs[i, :, 1])
            if np.argmin(self.costs[best_d, :, 3]) == i:
                self.buddies.add((i, best_d, 1))

    # --- PHASE 1: THE PLACER ---
    def run_placer(self, seed_grid=None):
        """
        Greedy placer. Can start from scratch or from a 'seed' segment (shifting).
        """
        grid = np.full((self.grid_size, self.grid_size), -1, dtype=int)
        placed = set()
        pq = [] # Priority Queue

        # Initialize Grid
        if seed_grid is None:
            # Start from absolute best buddy
            sorted_buddies = sorted(list(self.buddies), key=lambda x: self.costs[x[0], x[1], x[2]])
            if not sorted_buddies: return None
            
            p1, p2, rel = sorted_buddies[0]
            sy, sx = self.grid_n, self.grid_n
            grid[sy, sx] = p1
            placed.add(p1)
            
            if rel == 0: grid[sy, sx+1] = p2; placed.add(p2)
            else: grid[sy+1, sx] = p2; placed.add(p2)
        else:
            # Copy seed grid
            grid = seed_grid.copy()
            placed = set(grid[grid != -1])

        # Helper to add moves
        def add_candidates(r, c):
            pid = grid[r, c]
            for dr, dc, rel in [(0,1,0), (1,0,1), (0,-1,2), (-1,0,3)]:
                nr, nc = r+dr, c+dc
                if grid[nr, nc] != -1: continue

                # Find best unplaced piece
                best_cost = np.inf
                best_cand = -1
                
                for cand in range(self.num_pieces):
                    if cand not in placed:
                        # Base cost
                        cost = self.costs[pid, cand, rel]
                        
                        # Context Bonus (Check other neighbors)
                        if rel == 0 and grid[nr-1, nc] != -1: # Check Top
                            cost += self.costs[grid[nr-1, nc], cand, 1]
                        elif rel == 1 and grid[nr, nc-1] != -1: # Check Left
                            cost += self.costs[grid[nr, nc-1], cand, 0]
                        elif rel == 2 and grid[nr-1, nc] != -1: # Check Top
                            cost += self.costs[grid[nr-1, nc], cand, 1]
                        elif rel == 3 and grid[nr, nc-1] != -1: # Check Left
                            cost += self.costs[grid[nr, nc-1], cand, 0]
                            
                        if cost < best_cost:
                            best_cost = cost
                            best_cand = cand
                
                if best_cand != -1:
                    heapq.heappush(pq, (best_cost, nr, nc, best_cand))

        # Initialize PQ
        rows, cols = np.where(grid != -1)
        for r, c in zip(rows, cols): add_candidates(r, c)

        # Fill Loop
        while len(placed) < self.num_pieces and pq:
            cost, r, c, cand = heapq.heappop(pq)
            if cand in placed or grid[r, c] != -1: continue
            
            # Shape Constraint
            curr_rows, curr_cols = np.where(grid != -1)
            min_r, max_r = min(np.min(curr_rows), r), max(np.max(curr_rows), r)
            min_c, max_c = min(np.min(curr_cols), c), max(np.max(curr_cols), c)
            
            if (max_r - min_r + 1) > self.grid_n or (max_c - min_c + 1) > self.grid_n: continue
            
            grid[r, c] = cand
            placed.add(cand)
            add_candidates(r, c)
            
        return grid

    # --- PHASE 2: THE SEGMENTER ---
    def run_segmenter(self, grid):
        """
        Breaks the grid into chunks. Two pieces stay connected ONLY if
        they are 'Best Buddies'. Otherwise, the link is broken[cite: 209].
        """
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        segments = []

        for r in range(h):
            for c in range(w):
                if grid[r, c] != -1 and not visited[r, c]:
                    # Start new segment
                    current_segment = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    
                    while stack:
                        curr_r, curr_c = stack.pop()
                        pid = grid[curr_r, curr_c]
                        current_segment.append((curr_r, curr_c, pid))
                        
                        # Check neighbors
                        # Down (1)
                        if curr_r + 1 < h and grid[curr_r+1, curr_c] != -1:
                            nid = grid[curr_r+1, curr_c]
                            # Check if Best Buddies (pid, nid, 1)
                            if (pid, nid, 1) in self.buddies and not visited[curr_r+1, curr_c]:
                                visited[curr_r+1, curr_c] = True
                                stack.append((curr_r+1, curr_c))
                                
                        # Right (0)
                        if curr_c + 1 < w and grid[curr_r, curr_c+1] != -1:
                            nid = grid[curr_r, curr_c+1]
                            # Check if Best Buddies (pid, nid, 0)
                            if (pid, nid, 0) in self.buddies and not visited[curr_r, curr_c+1]:
                                visited[curr_r, curr_c+1] = True
                                stack.append((curr_r, curr_c+1))
                                
                        # Up (Check if nid matches pid as bottom)
                        if curr_r - 1 >= 0 and grid[curr_r-1, curr_c] != -1:
                            nid = grid[curr_r-1, curr_c]
                            # Is nid the top buddy of pid? (nid, pid, 1) in buddies
                            if (nid, pid, 1) in self.buddies and not visited[curr_r-1, curr_c]:
                                visited[curr_r-1, curr_c] = True
                                stack.append((curr_r-1, curr_c))

                        # Left (Check if nid matches pid as left)
                        if curr_c - 1 >= 0 and grid[curr_r, curr_c-1] != -1:
                            nid = grid[curr_r, curr_c-1]
                            if (nid, pid, 0) in self.buddies and not visited[curr_r, curr_c-1]:
                                visited[curr_r, curr_c-1] = True
                                stack.append((curr_r, curr_c-1))

                    segments.append(current_segment)
        return segments

    # --- PHASE 3: THE SHIFTER ---
    def run_shifter(self):
        """
        Iterative loop: Place -> Segment -> Keep Largest -> Repeat[cite: 215].
        """
        # Initial greedy placement
        current_grid = self.run_placer()
        if current_grid is None: return None
        
        # Iteration Loop (Try to improve 5 times)
        for _ in range(5):
            segments = self.run_segmenter(current_grid)
            if not segments: break
            
            # Find Largest Segment (S_max)
            largest_segment = max(segments, key=len)
            
            # If largest segment is the whole puzzle, we are done
            if len(largest_segment) == self.num_pieces:
                break
                
            # Create a new grid with JUST the largest segment (others are removed)
            seed_grid = np.full_like(current_grid, -1)
            
            # Re-center the segment to avoid bounds issues
            # Calculate shift to center
            rows = [x[0] for x in largest_segment]
            cols = [x[1] for x in largest_segment]
            min_r, min_c = min(rows), min(cols)
            
            offset_r = (self.grid_size // 2) - ((min_r + max(rows)) // 2)
            offset_c = (self.grid_size // 2) - ((min_c + max(cols)) // 2)
            
            for r, c, pid in largest_segment:
                new_r, new_c = r + offset_r, c + offset_c
                if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                    seed_grid[new_r, new_c] = pid
            
            # Re-run Placer using this segment as the fixed seed
            current_grid = self.run_placer(seed_grid=seed_grid)
            
        return current_grid

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def process_file(path, fname, out_folder, grid_n):
    img = cv2.imread(path)
    if img is None: return
    img = smart_crop(img)

    h, w, _ = img.shape
    ph, pw = h // grid_n, w // grid_n
    
    pieces = []
    for r in range(grid_n):
        for c in range(grid_n):
            y, x = r * ph, c * pw
            pieces.append(img[y:y+ph, x:x+pw])
            
    # SOLVE
    solver = PuzzleSolver(pieces, grid_n)
    final_map = solver.run_shifter() # Executes Place -> Segment -> Shift loop
        
    if final_map is None: return

    # Stitch
    rows = np.any(final_map != -1, axis=1)
    cols = np.any(final_map != -1, axis=0)
    trimmed_map = final_map[rows][:, cols]
    
    r_lim, c_lim = trimmed_map.shape
    r_lim = min(r_lim, grid_n)
    c_lim = min(c_lim, grid_n)
    
    res = np.zeros((r_lim*ph, c_lim*pw, 3), dtype=np.uint8)
    for r in range(r_lim):
        for c in range(c_lim):
            pid = trimmed_map[r, c]
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