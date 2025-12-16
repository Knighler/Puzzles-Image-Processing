import cv2
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_ROOT = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\preprocessing_results\\3_enhanced'
OUTPUT_ROOT = 'E:\\ASU\\Fall 25\\Image\\Project\\Raw Images\\Gravity Falls\\processed'
SUBFOLDERS = ['puzzle_2x2', 'puzzle_4x4', 'puzzle_8x8']

# ==========================================
# 1. PREPROCESSING
# ==========================================
def smart_crop(img, black_limit=40):
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    top, bottom, left, right = 0, h, 0, w
    for r in range(h):
        if np.mean(gray[r, :]) > black_limit: top = r; break
    for r in range(h-1, -1, -1):
        if np.mean(gray[r, :]) > black_limit: bottom = r+1; break
    for c in range(w):
        if np.mean(gray[:, c]) > black_limit: left = c; break
    for c in range(w-1, -1, -1):
        if np.mean(gray[:, c]) > black_limit: right = c+1; break
    if (right-left)<(w*0.5) or (bottom-top)<(h*0.5): return img
    return img[top:bottom, left:right]

# ==========================================
# 2. MATCHING METRIC (With Dynamic Threshold)
# ==========================================
def get_match_cost(piece_a, piece_b, relation, penalty_threshold):
    """
    Cost function with ADAPTIVE Penalty.
    penalty_threshold: Calculated dynamically per puzzle (Median Sigma).
    """
    A = cv2.cvtColor(piece_a, cv2.COLOR_BGR2LAB).astype(np.float32)
    B = cv2.cvtColor(piece_b, cv2.COLOR_BGR2LAB).astype(np.float32)

    if relation == 0: # A Left, B Right
        edge_a = A[:, -1, :]
        edge_b = B[:, 0, :]
        inner_a = A[:, -2, :]
    else: # A Top, B Bottom
        edge_a = A[-1, :, :]
        edge_b = B[0, :, :]
        inner_a = A[-2, :, :]

    diff = np.sum(np.abs(edge_a - edge_b))

    trend = edge_a - inner_a
    grad_cost = np.sum(np.abs((edge_a + trend) - edge_b))

    base_cost = diff + (3.0 * grad_cost)

    # --- ADAPTIVE ACTIVITY PENALTY ---
    # Use the dynamic threshold passed from the solver
    sigma_a = np.mean(np.std(edge_a, axis=0))
    sigma_b = np.mean(np.std(edge_b, axis=0))

    # If edge is significantly smoother than the puzzle's average...
    if sigma_a < (penalty_threshold * 0.5) or sigma_b < (penalty_threshold * 0.5):
        return base_cost * 5.0 # Massive penalty for boring edges

    return base_cost

# ==========================================
# 3. ROBUST SOLVER (With Texture Seeding)
# ==========================================
class PuzzleSolver:
    def __init__(self, pieces, grid_n):
        self.pieces = pieces
        self.grid_n = grid_n
        self.num_pieces = len(pieces)
        self.grid_size = grid_n * 3

        # 1. CALCULATE GLOBAL TEXTURE STATS (Dynamic Thresholding)
        # We calculate the std dev of every piece to find the "Median Activity"
        self.piece_textures = []
        for p in pieces:
            gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
            self.piece_textures.append(np.std(gray))

        self.median_texture = np.median(self.piece_textures)
        # Ensure threshold isn't too low for pure black images
        self.penalty_threshold = max(self.median_texture, 5.0)

        self.costs = np.full((self.num_pieces, self.num_pieces, 4), np.inf)
        self.buddies = set()

        self._calculate_costs()
        self._find_best_buddies()

    def _calculate_costs(self):
        for i in range(self.num_pieces):
            for j in range(self.num_pieces):
                if i == j: continue
                # Pass the dynamic threshold
                c_right = get_match_cost(self.pieces[i], self.pieces[j], 0, self.penalty_threshold)
                c_down = get_match_cost(self.pieces[i], self.pieces[j], 1, self.penalty_threshold)

                self.costs[i, j, 0] = c_right
                self.costs[j, i, 2] = c_right
                self.costs[i, j, 1] = c_down
                self.costs[j, i, 3] = c_down

    def _find_best_buddies(self):
        RATIO_THRESHOLD = 0.75
        for i in range(self.num_pieces):
            # Check Right
            costs_r = self.costs[i, :, 0]
            sorted_idx = np.argsort(costs_r)
            best, second = sorted_idx[0], sorted_idx[1]
            if costs_r[best] > 20000: continue
            ratio = costs_r[best] / (costs_r[second] + 1e-5)
            if ratio < RATIO_THRESHOLD:
                if np.argmin(self.costs[best, :, 2]) == i:
                    self.buddies.add((i, best, 0))

            # Check Down
            costs_d = self.costs[i, :, 1]
            sorted_idx = np.argsort(costs_d)
            best, second = sorted_idx[0], sorted_idx[1]
            if costs_d[best] > 20000: continue
            ratio = costs_d[best] / (costs_d[second] + 1e-5)
            if ratio < RATIO_THRESHOLD:
                if np.argmin(self.costs[best, :, 3]) == i:
                    self.buddies.add((i, best, 1))

    def run_placer(self, seed_grid=None):
        grid = np.full((self.grid_size, self.grid_size), -1, dtype=int)
        placed = set()

        # --- IMPROVED SEED SELECTION (Texture-Based) ---
        if seed_grid is not None:
            grid = seed_grid.copy()
            placed = set(grid[grid != -1])
        else:
            # Filter buddies: Only consider pairs where pieces have HIGH TEXTURE
            # This prevents starting with two black/flat pieces.
            rich_buddies = []
            for (p1, p2, rel) in self.buddies:
                # Combined texture score of the pair
                tex_score = self.piece_textures[p1] + self.piece_textures[p2]
                rich_buddies.append((tex_score, p1, p2, rel))

            if rich_buddies:
                # Sort by Texture Score (Descending), NOT by Cost
                # We want the most detailed pair, not the "cheapest" one.
                rich_buddies.sort(key=lambda x: x[0], reverse=True)
                _, p1, p2, rel = rich_buddies[0]

                sy, sx = self.grid_n, self.grid_n
                grid[sy, sx] = p1
                placed.add(p1)
                if rel == 0: grid[sy, sx+1] = p2; placed.add(p2)
                else: grid[sy+1, sx] = p2; placed.add(p2)
            else:
                # Fallback: Just pick the single most textured piece
                best_start = np.argmax(self.piece_textures)
                grid[self.grid_n, self.grid_n] = best_start
                placed.add(best_start)

        # Placement Loop (Robust Global Search)
        while len(placed) < self.num_pieces:
            candidates = set()
            rows, cols = np.where(grid != -1)
            min_r, max_r = np.min(rows), np.max(rows)
            min_c, max_c = np.min(cols), np.max(cols)

            for r, c in zip(rows, cols):
                for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                    nr, nc = r+dr, c+dc
                    if grid[nr, nc] == -1: candidates.add((nr, nc))

            if not candidates: break

            best_move = None
            best_global_cost = np.inf

            for (r, c) in candidates:
                new_min_r, new_max_r = min(min_r, r), max(max_r, r)
                new_min_c, new_max_c = min(min_c, c), max(max_c, c)
                if (new_max_r - new_min_r + 1) > self.grid_n: continue
                if (new_max_c - new_min_c + 1) > self.grid_n: continue

                neighbor_reqs = []
                if grid[r-1, c] != -1: neighbor_reqs.append((grid[r-1, c], 1))
                if grid[r+1, c] != -1: neighbor_reqs.append((grid[r+1, c], 3))
                if grid[r, c-1] != -1: neighbor_reqs.append((grid[r, c-1], 0))
                if grid[r, c+1] != -1: neighbor_reqs.append((grid[r, c+1], 2))

                if not neighbor_reqs: continue

                local_best_pid = -1
                local_best_cost = np.inf

                for pid in range(self.num_pieces):
                    if pid in placed: continue
                    total_cost = 0
                    for (n_pid, rel) in neighbor_reqs:
                        total_cost += self.costs[n_pid, pid, rel]

                    avg_cost = total_cost / len(neighbor_reqs)
                    if len(neighbor_reqs) > 1: avg_cost *= 0.6

                    if avg_cost < local_best_cost:
                        local_best_cost = avg_cost
                        local_best_pid = pid

                if local_best_pid != -1 and local_best_cost < best_global_cost:
                    best_global_cost = local_best_cost
                    best_move = (r, c, local_best_pid)

            if best_move:
                grid[best_move[0], best_move[1]] = best_move[2]
                placed.add(best_move[2])
            else:
                break
        return grid

    def force_fill_holes(self, grid):
        placed = set(grid[grid != -1])
        remaining = list(set(range(self.num_pieces)) - placed)
        if not remaining: return grid

        rows, cols = np.where(grid != -1)
        min_r, max_r = np.min(rows), np.max(rows)
        min_c, max_c = np.min(cols), np.max(cols)

        empty_slots = []
        for r in range(min_r, max_r+1):
            for c in range(min_c, max_c+1):
                if grid[r, c] == -1: empty_slots.append((r, c))

        for (r, c) in empty_slots:
            if not remaining: break
            best_pid = -1
            best_score = np.inf
            neighbor_reqs = []
            if grid[r-1, c] != -1: neighbor_reqs.append((grid[r-1, c], 1))
            if grid[r, c-1] != -1: neighbor_reqs.append((grid[r, c-1], 0))
            if grid[r+1, c] != -1: neighbor_reqs.append((grid[r+1, c], 3))
            if grid[r, c+1] != -1: neighbor_reqs.append((grid[r, c+1], 2))

            for pid in remaining:
                score = 0; count = 0
                for (n_pid, rel) in neighbor_reqs:
                    score += self.costs[n_pid, pid, rel]
                    count += 1
                if count > 0: score /= count
                if score < best_score:
                    best_score = score
                    best_pid = pid

            if best_pid != -1:
                grid[r, c] = best_pid
                remaining.remove(best_pid)
        return grid

    def run_segmenter(self, grid):
        h, w = grid.shape
        visited = np.zeros_like(grid, dtype=bool)
        segments = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != -1 and not visited[r, c]:
                    seg = []
                    stack = [(r, c)]
                    visited[r, c] = True
                    while stack:
                        curr_r, curr_c = stack.pop()
                        pid = grid[curr_r, curr_c]
                        seg.append((curr_r, curr_c, pid))
                        for dr, dc, rel in [(1,0,1), (0,1,0), (-1,0,3), (0,-1,2)]:
                            nr, nc = curr_r+dr, curr_c+dc
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != -1:
                                nid = grid[nr, nc]
                                if (pid, nid, rel) in self.buddies and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    stack.append((nr, nc))
                    segments.append(seg)
        return segments

    def run_shifter(self):
        current_grid = self.run_placer()
        if current_grid is None: return None
        for i in range(5):
            segments = self.run_segmenter(current_grid)
            if not segments: break
            largest_segment = max(segments, key=len)
            if len(largest_segment) == self.num_pieces: break
            seed_grid = np.full_like(current_grid, -1)
            rows = [x[0] for x in largest_segment]
            cols = [x[1] for x in largest_segment]
            offset_r = (self.grid_size // 2) - ((min(rows) + max(rows)) // 2)
            offset_c = (self.grid_size // 2) - ((min(cols) + max(cols)) // 2)
            for r, c, pid in largest_segment:
                new_r, new_c = r + offset_r, c + offset_c
                if 0 <= new_r < self.grid_size and 0 <= new_c < self.grid_size:
                    seed_grid[new_r, new_c] = pid
            current_grid = self.run_placer(seed_grid=seed_grid)
        current_grid = self.force_fill_holes(current_grid)
        return current_grid

# ==========================================
# 4. RUNNER
# ==========================================
def run_solver_for_folder(folder_name):
    full_path = os.path.join(INPUT_ROOT, folder_name)
    output_path = os.path.join(OUTPUT_ROOT, folder_name)
    if not os.path.exists(full_path): return
    if not os.path.exists(output_path): os.makedirs(output_path)
    try:
        part = folder_name.split('x')[0]
        grid_n = int(part.split('_')[-1] if '_' in part else part)
    except: return

    all_files = sorted([f for f in os.listdir(full_path) if f.endswith('.png')])
    puzzles = {}
    for f in all_files:
        if '_piece_' in f:
            pid = f.split('_piece_')[0]
            if pid not in puzzles: puzzles[pid] = []
            puzzles[pid].append(f)

    for pid, files in puzzles.items():
        if len(files) != grid_n * grid_n: continue
        pieces = []
        for f in files:
            img = cv2.imread(os.path.join(full_path, f))
            if img is not None: pieces.append(img)
        if not pieces: continue

        # Resize
        max_h = max(p.shape[0] for p in pieces)
        max_w = max(p.shape[1] for p in pieces)
        resized = [cv2.resize(p, (max_w, max_h)) for p in pieces]

        print(f"Solving {pid} (Grid: {grid_n}x{grid_n})...")
        solver = PuzzleSolver(resized, grid_n)
        final_grid = solver.run_shifter()

        if final_grid is None: continue

        rows = np.any(final_grid != -1, axis=1)
        cols = np.any(final_grid != -1, axis=0)
        trimmed_map = final_grid[rows][:, cols]
        r_lim, c_lim = min(trimmed_map.shape[0], grid_n), min(trimmed_map.shape[1], grid_n)

        res = np.zeros((r_lim*max_h, c_lim*max_w, 3), dtype=np.uint8)
        for r in range(r_lim):
            for c in range(c_lim):
                idx = trimmed_map[r, c]
                if idx != -1:
                    y, x = r*max_h, c*max_w
                    res[y:y+max_h, x:x+max_w] = resized[idx]

        save_path = os.path.join(output_path, f"{pid}_solved.png")
        cv2.imwrite(save_path, res)
        print(f"Saved: {save_path}")

if __name__=="__main__":
    if not os.path.exists(INPUT_ROOT): print("Error: Input root not found")
    else:
        for folder in SUBFOLDERS: run_solver_for_folder(folder)