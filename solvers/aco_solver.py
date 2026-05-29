from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

def _patch_env_cfg(env):
    if hasattr(env, "cfg"):
        return
    env.cfg = {
        "name": getattr(env, "config_name", "unknown"),
        "N": getattr(env, "N", 0),
        "C": getattr(env, "C", 0),
        "G": getattr(env, "G", 0),
        "T": getattr(env, "T", 0),
        "grid": getattr(env, "grid", []),
        "K_max": [getattr(s, "K_max", 0) for s in getattr(env, "shippers", [])],
        "W_max": [getattr(s, "W_max", 0.0) for s in getattr(env, "shippers", [])],
    }
    env.config = env.cfg

from env import (
    DeliveryEnv,
    Order,
    Shipper,
    DIRS,
    is_valid_cell,
    delivery_reward,
    move_cost,
)
from solvers.solver import Solver

Cell = Tuple[int, int]
Target = Tuple[str, int, Cell]

class KuhnMunkres:
    def __init__(self, profit_matrix: List[List[float]]):
        self.matrix = [row[:] for row in profit_matrix]
        self.M = len(self.matrix)
        self.N = len(self.matrix[0]) if self.M > 0 else 0

    def solve(self) -> Dict[int, int]:
        if self.M == 0 or self.N == 0:
            return {}
        
        transposed = False
        if self.M > self.N:
            self.matrix = [[self.matrix[i][j] for i in range(self.M)] for j in range(self.N)]
            self.M, self.N = self.N, self.M
            transposed = True
            
        u = [0.0] * self.M
        v = [0.0] * self.N
        
        for i in range(self.M):
            u[i] = max(self.matrix[i])
            
        p = [-1] * self.N
        
        for i in range(self.M):
            links = [-1] * self.N
            mins = [1e18] * self.N
            visited = [False] * self.N
            
            marked_row = i
            marked_col = -1
            
            while True:
                visited_col = -1
                delta = 1e18
                
                for j in range(self.N):
                    if not visited[j]:
                        val = u[marked_row] + v[j] - self.matrix[marked_row][j]
                        if val < mins[j]:
                            mins[j] = val
                            links[j] = marked_col
                        if mins[j] < delta:
                            delta = mins[j]
                            visited_col = j
                            
                u[i] -= delta
                for j in range(self.N):
                    if visited[j]:
                        u[p[j]] -= delta
                        v[j] += delta
                    else:
                        mins[j] -= delta
                        
                visited[visited_col] = True
                marked_col = visited_col
                
                if p[marked_col] == -1:
                    break
                else:
                    marked_row = p[marked_col]
                    
            curr_col = marked_col
            while curr_col != -1:
                prev_col = links[curr_col]
                if prev_col == -1:
                    p[curr_col] = i
                    break
                else:
                    p[curr_col] = p[prev_col]
                    curr_col = prev_col
                    
        matching = {}
        if transposed:
            for j in range(len(p)):
                if p[j] != -1:
                    matching[j] = p[j]
        else:
            for j in range(self.N):
                if p[j] != -1:
                    matching[p[j]] = j
                    
        return matching

class ACOSolver(Solver):
    INF = 10**9

    def __init__(self, env: DeliveryEnv):
        _patch_env_cfg(env)
        super().__init__(env)
        self.target_memory = {}
        self.commit_until = {}
        self.current_plans = {}
        self.last_orders_set = set()
        self.sssp_dist = {}
        self.sssp_parent = {}
        self.last_positions = {}
        self.stuck_counts = {}
        self.cluster_bonus_cache = {}
        self.plan_time_ema = 0.0
        self.replan_time_ema = 0.0
        self.lmax_penalty = 0

    def update_runtime_throttle(self, obs: dict, replan_elapsed: float) -> None:
        N = obs["N"]
        G = len(obs["orders"])
        # More realistic target replan time for large maps
        target_replan_time = 0.05 + 0.00005 * G + 0.00002 * (N * N)
        target_replan_time = max(0.06, min(0.40, target_replan_time))

        if replan_elapsed > target_replan_time * 1.5:
            self.lmax_penalty = min(self.lmax_penalty + 1, 5)
        elif replan_elapsed < target_replan_time * 0.6:
            self.lmax_penalty = max(self.lmax_penalty - 1, 0)

    def compute_dynamic_coeffs(self, obs: dict) -> Tuple[float, float, float]:
        N = obs["N"]
        G = len(obs["orders"])
        T = obs["T"]
        orders = obs["orders"]
        shippers = obs["shippers"]

        active = [
            o for o in orders.values()
            if not o.picked and not o.delivered
        ]

        if active:
            avg_slack = sum(max(0, o.et - obs["t"]) for o in active) / len(active)
            avg_p = sum(o.p for o in active) / len(active)
        else:
            avg_slack = T
            avg_p = 1.0

        # deadline pressure ratio
        urgency_ratio = 1.0 - min(1.0, avg_slack / max(1, T))

        # map size factor
        map_factor = min(1.0, N / 100.0)

        # 1. Distance penalty coefficient
        if N >= 28:
            dist_penalty_coeff = 0.045 + 0.035 * map_factor
        else:
            dist_penalty_coeff = 0.08 + 0.18 * map_factor
            dist_penalty_coeff -= 0.05 * urgency_ratio
            dist_penalty_coeff = max(0.05, min(0.30, dist_penalty_coeff))

        # 2. Flat delay penalty coefficient
        flat_delay_penalty_coeff = 0.0
        if urgency_ratio > 0.5:
            flat_delay_penalty_coeff = 0.05 + 0.15 * urgency_ratio

        # 3. Delivery completion bonus
        avg_bag = sum(len(s.bag) for s in shippers) / max(1, len(shippers))
        delivery_completion_bonus = 30.0 + 40.0 * avg_bag + 80.0 * urgency_ratio
        delivery_completion_bonus = max(20.0, min(180.0, delivery_completion_bonus))

        return dist_penalty_coeff, flat_delay_penalty_coeff, delivery_completion_bonus

    def run(self) -> dict:
        start = time.time()
        obs = self.env.observe()
        done = obs["done"]
        while not done:
            actions = self.plan_step(obs)
            obs, _reward, done, _info = self.env.step(actions)
        return self.env.result("ACOSolver", time.time() - start)

    def get_sssp_path(self, source: Cell, goal: Cell, grid: List[List[int]]) -> List[Cell]:
        if goal not in self.sssp_parent:
            dist = {goal: 0}
            parent = {goal: None}
            q = deque([goal])
            while q:
                cur = q.popleft()
                d = dist[cur]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nxt = (cur[0] + dr, cur[1] + dc)
                    if is_valid_cell(nxt, grid) and nxt not in dist:
                        dist[nxt] = d + 1
                        parent[nxt] = cur
                        q.append(nxt)
            if len(self.sssp_parent) >= 4096:
                k = next(iter(self.sssp_parent))
                self.sssp_parent.pop(k, None)
                self.sssp_dist.pop(k, None)
            self.sssp_parent[goal] = parent
            self.sssp_dist[goal] = dist

        parent = self.sssp_parent[goal]
        if source not in parent:
            return []
        path = []
        cur = source
        while cur is not None:
            path.append(cur)
            cur = parent[cur]
        return path

    def get_sssp_dist(self, source: Cell, goal: Cell, grid: List[List[int]]) -> int:
        if goal not in self.sssp_dist:
            _ = self.get_sssp_path(source, goal, grid)
        return self.sssp_dist[goal].get(source, self.INF)

    def is_target_still_valid(self, shipper: Shipper, target: Target, obs: dict) -> bool:
        orders = obs["orders"]
        typ, oid, pos = target
        order = orders.get(oid)
        if order is None or order.delivered:
            return False
        if typ == "delivery":
            return oid in shipper.bag and (order.ex, order.ey) == pos
        if typ == "pickup":
            return (not order.picked) and self.can_carry(shipper, order, orders) and (order.sx, order.sy) == pos
        return False

    def validate_and_clean_plan(self, s: Shipper, plan: List[Target], obs: dict) -> List[Target]:
        orders = obs["orders"]
        cleaned = []
        for tgt in plan:
            typ, oid, pos = tgt
            o = orders.get(oid)
            if o is None or o.delivered:
                continue
            if typ == "pickup":
                # If already picked by another shipper, we cannot do it
                if o.picked and o.carrier != s.id:
                    continue
                # If already picked by this shipper, it's in the bag, we don't need to pickup again
                if o.picked and o.carrier == s.id:
                    continue
                # If not picked, check if we can carry it
                if not o.picked and not self.can_carry(s, o, orders):
                    continue
                cleaned.append(tgt)
            elif typ == "delivery":
                # Only keep delivery if it is in our bag, or if we have a valid pickup for it earlier in the plan
                if oid in s.bag:
                    cleaned.append(tgt)
                else:
                    if any(t_prev[0] == "pickup" and t_prev[1] == oid for t_prev in cleaned):
                        cleaned.append(tgt)
        return cleaned

    def pickup_cluster_bonus(self, order: Order, obs: dict, radius: int = 4) -> float:
        if order.picked:
            return 0.0
        cnt = 0
        priority_sum = 0
        for o in obs["orders"].values():
            if o.delivered or o.picked:
                continue
            d = abs(o.sx - order.sx) + abs(o.sy - order.sy)
            if d <= radius:
                cnt += 1
                priority_sum += o.p
        return 1.0 * cnt + 0.8 * priority_sum

    def remember_target(self, shipper_id: int, target: Target, obs: dict, commit_steps: int = 3) -> None:
        self.target_memory[shipper_id] = target
        self.commit_until[shipper_id] = obs["t"] + commit_steps

    def get_committed_target(self, shipper: Shipper, obs: dict) -> Optional[Target]:
        target = self.target_memory.get(shipper.id)
        if target is None:
            return None
        if obs["t"] > self.commit_until.get(shipper.id, -1):
            return None
        if self.is_target_still_valid(shipper, target, obs):
            return target
        self.target_memory.pop(shipper.id, None)
        self.commit_until.pop(shipper.id, None)
        return None

    def optimize_delivery_sequence(
        self,
        start_pos: Cell,
        orders_list: List[Order],
        t: int,
        T: int,
        grid: List[List[int]],
        dist_penalty_coeff: float,
    ) -> Tuple[float, Dict[int, int], int]:
        if not orders_list:
            return 0.0, {}, 0
        perm_limit = 5 if len(grid) <= 20 else 3
        if len(orders_list) <= perm_limit:
            import itertools
            best_score = -1e9
            best_times = {}
            best_dist = 0
            for perm in itertools.permutations(orders_list):
                score = 0.0
                cur_pos = start_pos
                cur_t = t
                total_d = 0
                times = {}
                for o in perm:
                    d = self.get_sssp_dist(cur_pos, (o.ex, o.ey), grid)
                    if d >= self.INF:
                        score = -1e9
                        break
                    cur_t += d
                    total_d += d
                    times[o.id] = cur_t
                    score += delivery_reward(o, cur_t, T) - dist_penalty_coeff * d
                    cur_pos = (o.ex, o.ey)
                if score > best_score:
                    best_score = score
                    best_times = times
                    best_dist = total_d
            return best_score, best_times, best_dist
        else:
            remaining = list(orders_list)
            cur_pos = start_pos
            cur_t = t
            total_d = 0
            times = {}
            score = 0.0
            while remaining:
                best_o = None
                best_o_score = -1e9
                best_o_d = 0
                for o in remaining:
                    d = self.get_sssp_dist(cur_pos, (o.ex, o.ey), grid)
                    if d >= self.INF:
                        continue
                    o_reward = delivery_reward(o, cur_t + d, T)
                    o_score = o_reward - dist_penalty_coeff * d
                    if o_score > best_o_score:
                        best_o_score = o_score
                        best_o = o
                        best_o_d = d
                if best_o is None:
                    best_o = remaining[0]
                    best_o_d = self.get_sssp_dist(cur_pos, (best_o.ex, best_o.ey), grid)
                    if best_o_d >= self.INF:
                        best_o_d = 0
                remaining.remove(best_o)
                cur_t += best_o_d
                total_d += best_o_d
                times[best_o.id] = cur_t
                score += delivery_reward(best_o, cur_t, T) - dist_penalty_coeff * best_o_d
                cur_pos = (best_o.ex, best_o.ey)
            return score, times, total_d

    def evaluate_route_plan(
        self,
        route_plan: Dict[int, List[Target]],
        obs: dict,
    ) -> float:
        orders = obs["orders"]
        grid = obs["grid"]
        T = obs["T"]
        t_start = obs["t"]
        shippers = obs["shippers"]
        total_net_reward = 0.0
        occupancy = {}
        for s in shippers:
            route = route_plan.get(s.id, [])
            if not route:
                continue
            cur_pos = s.position
            cur_t = t_start
            virtual_bag = list(s.bag)
            occupancy.setdefault((cur_pos, cur_t), []).append(s.id)
            for typ, oid, pos in route:
                o = orders.get(oid)
                if o is None:
                    continue
                path = self.get_sssp_path(cur_pos, pos, grid)
                if not path or path[0] != cur_pos:
                    return -1e9
                for step_idx in range(1, len(path)):
                    nxt_cell = path[step_idx]
                    w_carried = sum(orders[bag_oid].w for bag_oid in virtual_bag if bag_oid in orders)
                    total_net_reward += move_cost(w_carried, s.W_max)
                    cur_t += 1
                    occupancy.setdefault((nxt_cell, cur_t), []).append(s.id)
                cur_pos = pos
                if typ == "pickup":
                    if oid not in virtual_bag:
                        virtual_bag.append(oid)
                elif typ == "delivery":
                    if oid in virtual_bag:
                        total_net_reward += delivery_reward(o, cur_t, T)
                        virtual_bag.remove(oid)
        congestion_penalty = 0.0
        for (cell, time_step), s_ids in occupancy.items():
            if len(s_ids) > 1:
                congestion_penalty += (len(s_ids) - 1) * 10.0
        return total_net_reward - congestion_penalty

    def solve_assignment_hungarian(self, profit_matrix: List[List[float]]) -> Dict[int, int]:
        M = len(profit_matrix)
        if M == 0:
            return {}
        N_targets = len(profit_matrix[0])
        if N_targets == 0:
            return {i: None for i in range(M)}
        
        km = KuhnMunkres(profit_matrix)
        matching = km.solve()
        
        res = {i: None for i in range(M)}
        for i, j in matching.items():
            if profit_matrix[i][j] > -1e8:
                res[i] = j
        return res

    def current_load(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def can_carry(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        if order.picked or order.delivered:
            return False
        if len(shipper.bag) >= shipper.K_max:
            return False
        return self.current_load(shipper, orders) + order.w <= shipper.W_max

    def has_deliverable_at(self, shipper: Shipper, pos: Cell, orders: Dict[int, Order]) -> bool:
        for oid in shipper.bag:
            o = orders.get(oid)
            if o is not None and not o.delivered and (o.ex, o.ey) == pos:
                return True
        return False

    def has_pickupable_at(self, shipper: Shipper, pos: Cell, orders: Dict[int, Order]) -> bool:
        for o in orders.values():
            if o.delivered or o.picked:
                continue
            if (o.sx, o.sy) != pos:
                continue
            if self.can_carry(shipper, o, orders):
                return True
        return False

    def estimate_delivery_time(self, shipper: Shipper, order: Order, obs: dict) -> int:
        grid = obs["grid"]
        t = obs["t"]
        if order.picked:
            d = self.get_sssp_dist(shipper.position, (order.ex, order.ey), grid)
            return t + d
        d1 = self.get_sssp_dist(shipper.position, (order.sx, order.sy), grid)
        d2 = self.get_sssp_dist((order.sx, order.sy), (order.ex, order.ey), grid)
        if d1 >= self.INF or d2 >= self.INF:
            return self.INF
        return t + d1 + d2

    def score_order_for_shipper(self, shipper: Shipper, order: Order, obs: dict) -> float:
        grid = obs["grid"]
        T = obs["T"]
        if order.picked and order.carrier != shipper.id:
            return -self.INF
        if not order.picked and order.w > shipper.W_max:
            return -self.INF
        est_t = self.estimate_delivery_time(shipper, order, obs)
        if est_t >= self.INF:
            return -self.INF
        reward = delivery_reward(order, est_t, T)
        if order.picked:
            travel = self.get_sssp_dist(shipper.position, (order.ex, order.ey), grid)
        else:
            travel = (
                self.get_sssp_dist(shipper.position, (order.sx, order.sy), grid)
                + self.get_sssp_dist((order.sx, order.sy), (order.ex, order.ey), grid)
            )
        lateness = max(0, est_t - order.et)
        slack = max(0, order.et - obs["t"])
        urgency_bonus = max(0, 30 - slack) * 0.5
        cluster_bonus = self.cluster_bonus_cache.get(order.id, 0.0)
        delivery_cluster_bonus = 0.0
        if shipper.bag:
            for oid in shipper.bag:
                bag_o = obs["orders"].get(oid)
                if bag_o:
                    d_dist = self.get_sssp_dist((bag_o.ex, bag_o.ey), (order.ex, order.ey), grid)
                    if d_dist <= 4:
                        delivery_cluster_bonus += (30.0 - d_dist * 5.0)
        return (
            reward * 5.0
            + 20.0 * order.p
            + urgency_bonus * 2.0
            + cluster_bonus * 3.0
            + delivery_cluster_bonus * 3.0
            - 0.1 * travel
            - 0.5 * lateness
        )

    def select_pickup_candidates_for_shipper(self, s: Shipper, active_pickups: List[Order], obs: dict, K: int = 40) -> List[Order]:
        grid = obs["grid"]
        t = obs["t"]
        scored = []
        d_penalty_factor = 1.0 if obs["N"] >= 28 else 2.5
        for o in active_pickups:
            if o.w > s.W_max:
                continue
            d = self.get_sssp_dist(s.position, (o.sx, o.sy), grid)
            if d >= self.INF:
                continue
            slack = max(0, o.et - t)
            urgency = max(0, 80 - slack) * 1.5
            qs = (
                30.0 * o.p
                - d_penalty_factor * d
                + urgency
                + 2.0 * self.cluster_bonus_cache.get(o.id, 0.0)
            )
            scored.append((qs, o))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [o for _, o in scored[:K]]

    def make_action_for_shipper(
        self,
        shipper: Shipper,
        target: Target,
        obs: dict,
    ) -> Tuple[str, int]:
        grid = obs["grid"]
        orders = obs["orders"]
        _typ, _oid, target_pos = target
        path = self.get_sssp_path(shipper.position, target_pos, grid)
        if not path or len(path) == 1:
            move = "S"
            after_pos = shipper.position
        else:
            next_cell = path[1]
            move = self.move_to_next_cell(shipper.position, next_cell)
            after_pos = next_cell
        if self.has_deliverable_at(shipper, after_pos, orders):
            return move, 2
        if self.has_pickupable_at(shipper, after_pos, orders):
            return move, 1
        return move, 0

    def move_to_next_cell(self, cur: Cell, nxt: Cell) -> str:
        dr = nxt[0] - cur[0]
        dc = nxt[1] - cur[1]
        if dr == -1 and dc == 0:
            return "U"
        if dr == 1 and dc == 0:
            return "D"
        if dr == 0 and dc == -1:
            return "L"
        if dr == 0 and dc == 1:
            return "R"
        return "S"

    def next_cell_after_move(self, pos: Cell, move: str, grid: List[List[int]]) -> Cell:
        dr, dc = DIRS.get(move, (0, 0))
        nxt = (pos[0] + dr, pos[1] + dc)
        return nxt if is_valid_cell(nxt, grid) else pos

    def get_shipper_priority(self, s: Shipper, obs: dict) -> float:
        orders = obs["orders"]
        t = obs["t"]
        priority = 1000.0 * len(s.bag)
        for oid in s.bag:
            o = orders.get(oid)
            if o:
                time_left = o.et - t
                urgency = max(0, 300 - time_left) * 1.5
                priority += urgency + o.p * 100.0
        if not s.bag:
            target = self.get_committed_target(s, obs)
            if target is not None:
                typ, oid, pos = target
                o = orders.get(oid)
                if o:
                    time_left = o.et - t
                    urgency = max(0, 300 - time_left) * 0.5
                    priority += 100.0 + urgency + o.p * 10.0
        priority -= 0.01 * s.id
        return priority

    def resolve_action_conflicts(
        self,
        actions: Dict[int, Tuple[str, int]],
        obs: dict,
    ) -> Dict[int, Tuple[str, int]]:
        shippers: List[Shipper] = obs["shippers"]
        orders: Dict[int, Order] = obs["orders"]
        grid = obs["grid"]
        shipper_by_id = {s.id: s for s in shippers}
        old_positions = {s.id: s.position for s in shippers}
        priorities = {s.id: self.get_shipper_priority(s, obs) for s in shippers}
        current_actions = dict(actions)
        max_iterations = 15
        for iteration in range(max_iterations):
            occupied = set(old_positions.values())
            desired = {}
            for s in shippers:
                move, _op = current_actions.get(s.id, ("S", 0))
                desired[s.id] = self.next_cell_after_move(s.position, move, grid)
            final_pos = {}
            blocked_shippers = []
            for s in sorted(shippers, key=lambda x: x.id):
                old = old_positions[s.id]
                tgt = desired[s.id]
                occupied.discard(old)
                if tgt in occupied:
                    if tgt != old:
                        blocked_shippers.append((s.id, tgt, old))
                    tgt = old
                occupied.add(tgt)
                final_pos[s.id] = tgt
            if not blocked_shippers:
                break
            blocked_shippers.sort(key=lambda x: priorities[x[0]], reverse=True)
            yielded = False
            for blocked_id, tgt_cell, blocked_old in blocked_shippers:
                blocker_id = None
                for sid, pos in final_pos.items():
                    if pos == tgt_cell and sid != blocked_id:
                        blocker_id = sid
                        break
                if blocker_id is None:
                    continue
                if priorities[blocked_id] > priorities[blocker_id]:
                    if current_actions.get(blocker_id, ("S", 0))[1] > 0:
                        continue
                    blocker_s = shipper_by_id[blocker_id]
                    blocker_pos = blocker_s.position
                    safe_moves = []
                    for mv in ["U", "D", "L", "R", "S"]:
                        if mv == "S":
                            nxt = blocker_pos
                        else:
                            dr, dc = DIRS[mv]
                            nxt = (blocker_pos[0] + dr, blocker_pos[1] + dc)
                        if not is_valid_cell(nxt, grid):
                            continue
                        if nxt == blocked_old:
                            continue
                        is_safe = True
                        for other_id, other_s in shipper_by_id.items():
                            if other_id == blocker_id:
                                continue
                            if priorities[other_id] >= priorities[blocker_id]:
                                if other_s.position == nxt or desired.get(other_id) == nxt:
                                    is_safe = False
                                    break
                        if is_safe:
                            target = self.target_memory.get(blocker_id)
                            if target is not None:
                                dist = self.get_sssp_dist(nxt, target[2], grid)
                            else:
                                dist = 0
                            stay_penalty = 1000 if mv == "S" else 0
                            score = dist + stay_penalty
                            safe_moves.append((score, mv))
                    if safe_moves:
                        safe_moves.sort()
                        yield_mv = safe_moves[0][1]
                        if current_actions[blocker_id] != (yield_mv, 0):
                            current_actions[blocker_id] = (yield_mv, 0)
                            yielded = True
                            break
            if not yielded:
                break
        occupied = set(old_positions.values())
        desired = {}
        for s in shippers:
            move, _op = current_actions.get(s.id, ("S", 0))
            desired[s.id] = self.next_cell_after_move(s.position, move, grid)
        final_pos = {}
        for s in sorted(shippers, key=lambda x: x.id):
            old = old_positions[s.id]
            tgt = desired[s.id]
            occupied.discard(old)
            if tgt in occupied:
                tgt = old
            occupied.add(tgt)
            final_pos[s.id] = tgt
        fixed = {}
        for sid, s in shipper_by_id.items():
            old = old_positions[sid]
            pos = final_pos[sid]
            move = self.move_to_next_cell(old, pos)
            if self.has_deliverable_at(s, pos, orders):
                op = 2
            elif self.has_pickupable_at(s, pos, orders):
                op = 1
            else:
                op = 0
            fixed[sid] = (move, op)
        return fixed

    def is_new_order_important(self, o: Order, obs: dict, affect_radius: int) -> bool:
        t = obs["t"]
        grid = obs["grid"]
        if o.p >= 2:
            return True
        if o.et - t <= 45:
            return True
        for s in obs["shippers"]:
            d = self.get_sssp_dist(s.position, (o.sx, o.sy), grid)
            if d <= affect_radius:
                return True
        return False

    def build_sequential_route(self, s: Shipper, primary_target: Optional[Target], obs: dict, dist_penalty_coeff: float, L_max: int = 4, K_next: int = 10) -> List[Target]:
        orders = obs["orders"]
        grid = obs["grid"]
        T = obs["T"]
        t = obs["t"]

        # Initial virtual state
        virtual_bag = [oid for oid in s.bag if oid in orders]
        cur_pos = s.position
        cur_t = t
        route = []

        # 1. If there is a primary target, insert it first
        if primary_target is not None:
            typ, oid, pos = primary_target
            o = orders.get(oid)
            if o is not None:
                d = self.get_sssp_dist(cur_pos, pos, grid)
                if d < self.INF:
                    route.append(primary_target)
                    cur_pos = pos
                    cur_t += d
                    if typ == "pickup":
                        if oid not in virtual_bag:
                            virtual_bag.append(oid)
                    elif typ == "delivery":
                        if oid in virtual_bag:
                            virtual_bag.remove(oid)

        # 2. Sequential greedy expansion (interleaving pickups & deliveries on large maps)
        is_large_map = (obs.get("N", 0) >= 28 or len(orders) >= 200)
        if is_large_map:
            while len(route) < L_max:
                curr_weight = sum(orders[oid].w for oid in virtual_bag if oid in orders)
                candidates = []
                
                # Candidate deliveries (all orders in virtual_bag)
                for oid in virtual_bag:
                    o = orders.get(oid)
                    if o:
                        d = self.get_sssp_dist(cur_pos, (o.ex, o.ey), grid)
                        if d < self.INF:
                            est_t = cur_t + d
                            reward = delivery_reward(o, est_t, T)
                            move_c = move_cost(curr_weight, s.W_max) * d
                            
                            # Soft lateness penalty & urgency bonus
                            lateness_penalty = 0.0
                            if est_t > o.et:
                                lateness_penalty = -20.0 - 1.0 * (est_t - o.et)
                            else:
                                slack = o.et - est_t
                                urgency_penalty = max(0.0, 45.0 - slack) * 1.5
                                lateness_penalty = urgency_penalty
                                    
                            # Bag fullness bonus to encourage clearing space
                            fullness_ratio = len(virtual_bag) / s.K_max
                            fullness_bonus = fullness_ratio * 40.0
                            
                            score = (
                                reward * 5.0
                                + fullness_bonus
                                + lateness_penalty
                                + o.p * 20.0
                                - dist_penalty_coeff * d
                                - move_c
                            )
                            candidates.append((score, ("delivery", oid, (o.ex, o.ey)), d, o.w, "delivery"))
                
                # Candidate pickups
                if len(virtual_bag) < s.K_max:
                    active_pickups = []
                    for o in orders.values():
                        if o.picked or o.delivered:
                            continue
                        # Exclude if it is already in our route plan
                        if any(tgt[1] == o.id for tgt in route):
                            continue
                        if curr_weight + o.w <= s.W_max:
                            d_pickup = self.get_sssp_dist(cur_pos, (o.sx, o.sy), grid)
                            if d_pickup < self.INF:
                                active_pickups.append((d_pickup, o))
                    
                    # Sort and select top K_next nearby pickups to evaluate
                    active_pickups.sort(key=lambda x: x[0])
                    for d_pickup, o in active_pickups[:K_next]:
                        d_delivery = self.get_sssp_dist((o.sx, o.sy), (o.ex, o.ey), grid)
                        if d_delivery < self.INF:
                            # Check delay penalty on existing cargo in virtual_bag
                            delay_cost = 0.0
                            for bag_oid in virtual_bag:
                                bag_o = orders.get(bag_oid)
                                if bag_o:
                                    est_t_without = cur_t + self.get_sssp_dist(cur_pos, (bag_o.ex, bag_o.ey), grid)
                                    est_t_with = cur_t + d_pickup + self.get_sssp_dist((o.sx, o.sy), (bag_o.ex, bag_o.ey), grid)
                                    if est_t_without <= bag_o.et and est_t_with > bag_o.et:
                                        delay_cost += 100.0
                                    elif est_t_with > bag_o.et:
                                        delay_cost += 5.0 * (est_t_with - est_t_without)

                            est_t_delivery = cur_t + d_pickup + d_delivery
                            reward = delivery_reward(o, est_t_delivery, T)
                            cost = move_cost(curr_weight, s.W_max) * d_pickup + move_cost(curr_weight + o.w, s.W_max) * d_delivery
                            
                            # Soft lateness penalty & urgency bonus
                            lateness_penalty = 0.0
                            if est_t_delivery > o.et:
                                lateness_penalty = -20.0 - 1.0 * (est_t_delivery - o.et)
                            else:
                                slack = o.et - est_t_delivery
                                urgency_penalty = max(0.0, 45.0 - slack) * 1.5
                                lateness_penalty = urgency_penalty
                                    
                            p_cluster_bonus = self.cluster_bonus_cache.get(o.id, 0.0)
                            
                            # Delivery clustering bonus
                            d_cluster_bonus = 0.0
                            for bag_oid in virtual_bag:
                                bag_o = orders.get(bag_oid)
                                if bag_o:
                                    d_dist = self.get_sssp_dist((bag_o.ex, bag_o.ey), (o.ex, o.ey), grid)
                                    if d_dist <= 5:
                                        d_cluster_bonus += (15.0 - d_dist) * 2.0

                            score = (
                                reward * 5.0
                                + lateness_penalty
                                + p_cluster_bonus * 3.0
                                + d_cluster_bonus
                                + o.p * 25.0
                                - dist_penalty_coeff * (d_pickup + d_delivery)
                                - cost
                                - delay_cost
                            )
                            # Only accept pickup if net gain is good enough
                            if score > -150.0:
                                candidates.append((score, ("pickup", o.id, (o.sx, o.sy)), d_pickup, o.w, "pickup"))
                
                if not candidates:
                    break
                    
                # Choose the highest scoring candidate
                candidates.sort(reverse=True, key=lambda x: x[0])
                best_score, best_tgt, best_d, best_w, best_typ = candidates[0]
                
                route.append(best_tgt)
                cur_pos = best_tgt[2]
                cur_t += best_d
                if best_typ == "pickup":
                    virtual_bag.append(best_tgt[1])
                elif best_typ == "delivery":
                    if best_tgt[1] in virtual_bag:
                        virtual_bag.remove(best_tgt[1])
                    
        # 3. Plan all remaining cargo deliveries in virtual_bag using the exact optimal permutation solver
        if virtual_bag:
            bag_orders_list = [orders[oid] for oid in virtual_bag if oid in orders]
            if bag_orders_list:
                _, times, _ = self.optimize_delivery_sequence(
                    cur_pos,
                    bag_orders_list,
                    cur_t,
                    T,
                    grid,
                    dist_penalty_coeff,
                )
                best_seq = sorted(times.items(), key=lambda x: x[1])
                for next_oid, _ in best_seq:
                    next_o = orders.get(next_oid)
                    if next_o:
                        route.append(("delivery", next_oid, (next_o.ex, next_o.ey)))
                        
        # 4. Fallback Active Pickup for Idle, Plan-less Shippers
        if not route and not s.bag:
            best_o = None
            best_o_score = -1e9
            for o in orders.values():
                if not o.picked and not o.delivered and self.can_carry(s, o, orders):
                    d = self.get_sssp_dist(cur_pos, (o.sx, o.sy), grid)
                    if d < self.INF:
                        score = 100.0 * o.p - 1.5 * d
                        if score > best_o_score:
                            best_o_score = score
                            best_o = o
            if best_o:
                route.append(("pickup", best_o.id, (best_o.sx, best_o.sy)))
                route.append(("delivery", best_o.id, (best_o.ex, best_o.ey)))

        return route

    def score_single_route(self, s: Shipper, route: List[Target], obs: dict) -> float:
        if not route:
            return 0.0
        orders = obs["orders"]
        grid = obs["grid"]
        T = obs["T"]
        t_start = obs["t"]
        dist_penalty_coeff = getattr(self, "dist_penalty_coeff", 0.2)

        cur_pos = s.position
        cur_t = t_start
        virtual_bag = list(s.bag)
        total_net_reward = 0.0

        for typ, oid, pos in route:
            o = orders.get(oid)
            if o is None:
                continue
            path = self.get_sssp_path(cur_pos, pos, grid)
            if not path or path[0] != cur_pos:
                return -1e9
            for step_idx in range(1, len(path)):
                w_carried = sum(orders[bag_oid].w for bag_oid in virtual_bag if bag_oid in orders)
                total_net_reward += move_cost(w_carried, s.W_max)
                cur_t += 1
            cur_pos = pos
            if typ == "pickup":
                if oid not in virtual_bag:
                    virtual_bag.append(oid)
            elif typ == "delivery":
                if oid in virtual_bag:
                    total_net_reward += delivery_reward(o, cur_t, T) - dist_penalty_coeff * (len(path) - 1)
                    virtual_bag.remove(oid)
        return total_net_reward

    def build_aco_route(
        self,
        s: Shipper,
        primary_target: Optional[Target],
        candidate_pickups: List[Order],
        obs: dict,
        L_max: int,
        K_next: int = 10,
    ) -> List[Target]:
        orders = obs["orders"]
        grid = obs["grid"]
        T = obs["T"]
        t = obs["t"]
        dist_penalty_coeff = getattr(self, "dist_penalty_coeff", 0.2)

        # Check if we should run ACO search (only on large maps)
        is_large_map = (obs.get("N", 0) >= 28 or len(orders) >= 200)
        if not is_large_map:
            # Bypass ACO entirely on small maps to preserve maximum baseline score and speed
            return self.build_sequential_route(s, primary_target, obs, dist_penalty_coeff, L_max, K_next)

        best_route = []
        best_route_score = -1e9

        # Local pheromone cache
        local_pheromone = {}

        # Dynamically scale ant population and iterations for speed and efficiency
        N_val = obs.get("N", 0)
        if N_val >= 60:
            num_ants = 16
            num_iters = 4
        elif N_val >= 30:
            num_ants = 20
            num_iters = 5
        else:
            num_ants = 24
            num_iters = 6

        alpha = 1.0
        beta = 2.0
        rho = 0.1

        # We also generate the greedy fallback route as the baseline to beat
        greedy_route = self.build_sequential_route(s, primary_target, obs, dist_penalty_coeff, L_max, K_next)
        greedy_score = self.score_single_route(s, greedy_route, obs)
        best_route = list(greedy_route)
        best_route_score = greedy_score

        # Start iterations
        for iteration in range(num_iters):
            ant_routes = []
            ant_scores = []

            for ant in range(num_ants):
                route = []
                virtual_bag = [oid for oid in s.bag if oid in orders]
                cur_pos = s.position
                cur_t = t
                
                # 1. Primary target insertion
                if primary_target is not None:
                    typ, oid, pos = primary_target
                    o = orders.get(oid)
                    if o is not None:
                        d = self.get_sssp_dist(cur_pos, pos, grid)
                        if d < self.INF:
                            route.append(primary_target)
                            cur_pos = pos
                            cur_t += d
                            if typ == "pickup":
                                if oid not in virtual_bag:
                                    virtual_bag.append(oid)
                            elif typ == "delivery":
                                if oid in virtual_bag:
                                    virtual_bag.remove(oid)

                # 2. Construction steps
                last_key = ("start", s.id) if not route else (route[-1][0], route[-1][1])
                
                while len(route) < L_max:
                    curr_weight = sum(orders[oid].w for oid in virtual_bag if oid in orders)
                    candidates = []

                    # Add eligible deliveries from virtual_bag
                    for oid in virtual_bag:
                        o = orders.get(oid)
                        if o:
                            d = self.get_sssp_dist(cur_pos, (o.ex, o.ey), grid)
                            if d < self.INF:
                                est_t = cur_t + d
                                reward = delivery_reward(o, est_t, T)
                                move_c = move_cost(curr_weight, s.W_max) * d
                                
                                # Soft lateness penalty & urgency bonus
                                lateness_penalty = 0.0
                                if est_t > o.et:
                                    lateness_penalty = -20.0 - 1.0 * (est_t - o.et)
                                else:
                                    slack = o.et - est_t
                                    urgency_penalty = max(0.0, 45.0 - slack) * 1.5
                                    lateness_penalty = urgency_penalty
                                
                                fullness_ratio = len(virtual_bag) / s.K_max
                                fullness_bonus = fullness_ratio * 40.0
                                
                                gain = (
                                    reward * 5.0
                                    + fullness_bonus
                                    + lateness_penalty
                                    + o.p * 20.0
                                    - dist_penalty_coeff * d
                                    - move_c
                                )
                                candidates.append(("delivery", oid, (o.ex, o.ey), gain, d, o.w))

                    # Add eligible pickups from candidate_pickups
                    if len(virtual_bag) < s.K_max:
                        active_pickups = []
                        for o in candidate_pickups:
                            if o.picked or o.delivered:
                                continue
                            if any(tgt[1] == o.id for tgt in route):
                                continue
                            if curr_weight + o.w <= s.W_max:
                                d_pickup = self.get_sssp_dist(cur_pos, (o.sx, o.sy), grid)
                                if d_pickup < self.INF:
                                    active_pickups.append((d_pickup, o))
                        
                        active_pickups.sort(key=lambda x: x[0])
                        for d_pickup, o in active_pickups[:K_next]:
                            d_delivery = self.get_sssp_dist((o.sx, o.sy), (o.ex, o.ey), grid)
                            if d_delivery < self.INF:
                                delay_cost = 0.0
                                for bag_oid in virtual_bag:
                                    bag_o = orders.get(bag_oid)
                                    if bag_o:
                                        est_t_without = cur_t + self.get_sssp_dist(cur_pos, (bag_o.ex, bag_o.ey), grid)
                                        est_t_with = cur_t + d_pickup + self.get_sssp_dist((o.sx, o.sy), (bag_o.ex, bag_o.ey), grid)
                                        if est_t_without <= bag_o.et and est_t_with > bag_o.et:
                                            delay_cost += 100.0
                                        elif est_t_with > bag_o.et:
                                            delay_cost += 5.0 * (est_t_with - est_t_without)

                                est_t_delivery = cur_t + d_pickup + d_delivery
                                reward = delivery_reward(o, est_t_delivery, T)
                                cost = move_cost(curr_weight, s.W_max) * d_pickup + move_cost(curr_weight + o.w, s.W_max) * d_delivery
                                
                                # Soft lateness penalty & urgency bonus
                                lateness_penalty = 0.0
                                if est_t_delivery > o.et:
                                    lateness_penalty = -20.0 - 1.0 * (est_t_delivery - o.et)
                                else:
                                    slack = o.et - est_t_delivery
                                    urgency_penalty = max(0.0, 45.0 - slack) * 1.5
                                    lateness_penalty = urgency_penalty
                                
                                p_cluster_bonus = self.cluster_bonus_cache.get(o.id, 0.0)
                                 
                                # Delivery clustering bonus
                                d_cluster_bonus = 0.0
                                for bag_oid in virtual_bag:
                                    bag_o = orders.get(bag_oid)
                                    if bag_o:
                                        d_dist = self.get_sssp_dist((bag_o.ex, bag_o.ey), (o.ex, o.ey), grid)
                                        if d_dist <= 5:
                                            d_cluster_bonus += (15.0 - d_dist) * 2.0
                                
                                gain = (
                                    reward * 5.0
                                    + lateness_penalty
                                    + p_cluster_bonus * 3.0
                                    + d_cluster_bonus
                                    + o.p * 25.0
                                    - dist_penalty_coeff * (d_pickup + d_delivery)
                                    - cost
                                    - delay_cost
                                )
                                if gain > -150.0:
                                    candidates.append(("pickup", o.id, (o.sx, o.sy), gain, d_pickup, o.w))

                    if not candidates:
                        break

                    # 3. Probability calculation with pheromones
                    probabilities = []
                    for typ, oid, pos, gain, d_step, w_cargo in candidates:
                        eta = max(1e-6, gain + 200.0)
                        next_key = (typ, oid)
                        tau = local_pheromone.get((last_key, next_key), 1.0)
                        probabilities.append( (tau ** alpha) * (eta ** beta) )

                    # Weighted random selection
                    total_prob = sum(probabilities)
                    if total_prob <= 0.0:
                        selected_idx = 0
                    else:
                        import random
                        r = random.uniform(0, total_prob)
                        cumulative = 0.0
                        selected_idx = 0
                        for idx, p in enumerate(probabilities):
                            cumulative += p
                            if r <= cumulative:
                                selected_idx = idx
                                break
                    
                    # Apply choice
                    sel_typ, sel_oid, sel_pos, sel_gain, sel_d, sel_w = candidates[selected_idx]
                    route.append((sel_typ, sel_oid, sel_pos))
                    cur_pos = sel_pos
                    cur_t += sel_d
                    if sel_typ == "pickup":
                        virtual_bag.append(sel_oid)
                    elif sel_typ == "delivery":
                        if sel_oid in virtual_bag:
                            virtual_bag.remove(sel_oid)
                    last_key = (sel_typ, sel_oid)

                # 4. Deliver remaining bag orders
                if virtual_bag:
                    bag_orders_list = [orders[oid] for oid in virtual_bag if oid in orders]
                    if bag_orders_list:
                        _, times, _ = self.optimize_delivery_sequence(
                            cur_pos, bag_orders_list, cur_t, T, grid, dist_penalty_coeff
                        )
                        best_seq = sorted(times.items(), key=lambda x: x[1])
                        for next_oid, _ in best_seq:
                            next_o = orders.get(next_oid)
                            if next_o:
                                route.append(("delivery", next_oid, (next_o.ex, next_o.ey)))

                # 5. Score the complete route
                score = self.score_single_route(s, route, obs)
                ant_routes.append(route)
                ant_scores.append(score)

                if score > best_route_score:
                    best_route = list(route)
                    best_route_score = score

            # 6. Pheromone updates
            for k in list(local_pheromone.keys()):
                local_pheromone[k] *= (1.0 - rho)
                if local_pheromone[k] < 0.1:
                    local_pheromone.pop(k)

            for route_idx, score in enumerate(ant_scores):
                if score > greedy_score:
                    ant_r = ant_routes[route_idx]
                    prev_k = ("start", s.id)
                    deposit = 0.1 * (score - greedy_score)
                    for step in ant_r:
                        next_k = (step[0], step[1])
                        local_pheromone[(prev_k, next_k)] = local_pheromone.get((prev_k, next_k), 1.0) + deposit
                        prev_k = next_k

        return best_route

    def plan_step(self, obs: dict) -> Dict[int, Tuple[str, int]]:
        step_start = time.time()
        shippers: List[Shipper] = obs["shippers"]
        orders: Dict[int, Order] = obs["orders"]
        grid = obs["grid"]
        N = obs["N"]
        T = obs["T"]
        t = obs["t"]

        # Stage 1: Plan validation
        for s in shippers:
            plan = self.current_plans.get(s.id, [])
            self.current_plans[s.id] = self.validate_and_clean_plan(s, plan, obs)

        # Stage 3 (Part 1): Dynamic control parameters
        fleet_bonus = len(shippers) // 5
        periodic_interval = max(6, min(18, 6 + N // 10 - fleet_bonus))
        affect_radius = max(5, min(18, 4 + int(0.12 * N)))
        periodic_replan = (t % periodic_interval == 0)

        # Stage 2: Detect new important orders
        current_order_ids = set(obs["orders"].keys())
        new_order_ids = current_order_ids - self.last_orders_set
        new_orders = [
            orders[oid]
            for oid in new_order_ids
            if oid in orders and not orders[oid].picked and not orders[oid].delivered
        ]
        important_new_order = any(
            self.is_new_order_important(o, obs, affect_radius)
            for o in new_orders
        )

        # Stuck trigger detection
        stuck_trigger = False
        for s in shippers:
            last_pos = self.last_positions.get(s.id)
            if last_pos is not None and s.position == last_pos:
                plan = self.current_plans.get(s.id, [])
                if plan:
                    next_act = self.make_action_for_shipper(s, plan[0], obs)
                    if next_act[0] != "S":
                        self.stuck_counts[s.id] = self.stuck_counts.get(s.id, 0) + 1
                        if self.stuck_counts[s.id] >= 3:
                            stuck_trigger = True
                    else:
                        self.stuck_counts[s.id] = 0
                else:
                    self.stuck_counts[s.id] = 0
            else:
                self.stuck_counts[s.id] = 0
            self.last_positions[s.id] = s.position

        # Finished target detection
        finished_target = False
        for s in shippers:
            plan = self.current_plans.get(s.id, [])
            if not plan:
                if s.bag:
                    finished_target = True
                    break
                has_available_pickup = False
                for o in orders.values():
                    if not o.picked and not o.delivered and self.can_carry(s, o, orders):
                        has_available_pickup = True
                        break
                if has_available_pickup:
                    finished_target = True
                    break

        plan_invalid = False
        for s in shippers:
            old_plan = self.current_plans.get(s.id, [])
            new_plan = self.validate_and_clean_plan(s, old_plan, obs)
            if len(new_plan) != len(old_plan):
                plan_invalid = True
            self.current_plans[s.id] = new_plan

        # Stage 4: Decide global vs local replan
        is_global_replan = (
            not self.current_plans
            or periodic_replan
            or stuck_trigger
        )
        is_local_replan = (
            important_new_order
            or finished_target
            or plan_invalid
        )
        replan_needed = is_global_replan or is_local_replan

        if not replan_needed:
            actions = {}
            for s in shippers:
                plan = self.current_plans.get(s.id, [])
                if plan:
                    actions[s.id] = self.make_action_for_shipper(s, plan[0], obs)
                else:
                    actions[s.id] = ("S", 0)
            actions = self.resolve_action_conflicts(actions, obs)
            return actions

        self.stuck_counts = {s.id: 0 for s in shippers}

        # Stage 5: Selective shipper rebuild
        if is_global_replan:
            rebuild_shippers = list(shippers)
        else:
            affected_shippers = []
            for s in shippers:
                plan = self.current_plans.get(s.id, [])
                is_idle = (not plan)
                plan_too_short = (len(plan) <= 1)
                
                is_close = False
                for o in new_orders:
                    d = self.get_sssp_dist(s.position, (o.sx, o.sy), grid)
                    if d <= affect_radius:
                        is_close = True
                        break
                
                has_urgent_cargo = False
                for oid in s.bag:
                    o = orders.get(oid)
                    if o and o.et - t <= 30:
                        has_urgent_cargo = True
                        break
                
                if is_idle or plan_too_short or (is_close and not has_urgent_cargo):
                    affected_shippers.append(s)
            rebuild_shippers = affected_shippers

        if not rebuild_shippers:
            actions = {}
            for s in shippers:
                plan = self.current_plans.get(s.id, [])
                if plan:
                    actions[s.id] = self.make_action_for_shipper(s, plan[0], obs)
                else:
                    actions[s.id] = ("S", 0)
            actions = self.resolve_action_conflicts(actions, obs)
            return actions

        # Stage 6: Cache cluster bonus only when replan
        active_pickups = [
            o for o in orders.values()
            if not o.picked and not o.delivered
        ]
        self.cluster_bonus_cache = {
            o.id: self.pickup_cluster_bonus(o, obs)
            for o in active_pickups
        }

        # Stage 7: Immediate actions
        immediate_actions = {}
        busy_shipper_ids = set()
        immediate_targets = {}
        for s in rebuild_shippers:
            if self.has_deliverable_at(s, s.position, orders):
                immediate_actions[s.id] = ("S", 2)
                busy_shipper_ids.add(s.id)
                self.target_memory.pop(s.id, None)
                self.commit_until.pop(s.id, None)
                deliverable_oid = next((oid for oid in s.bag if orders.get(oid) and (orders[oid].ex, orders[oid].ey) == s.position), None)
                if deliverable_oid is not None:
                    immediate_targets[s.id] = ("delivery", deliverable_oid, s.position)
            elif self.has_pickupable_at(s, s.position, orders):
                pickup_o = None
                for o in orders.values():
                    if not o.picked and not o.delivered and (o.sx, o.sy) == s.position and self.can_carry(s, o, orders):
                        pickup_o = o
                        break
                if pickup_o:
                    immediate_actions[s.id] = ("S", 1)
                    busy_shipper_ids.add(s.id)
                    self.target_memory.pop(s.id, None)
                    self.commit_until.pop(s.id, None)
                    immediate_targets[s.id] = ("pickup", pickup_o.id, s.position)

        idle_rebuild_shippers = [s for s in rebuild_shippers if s.id not in busy_shipper_ids]
        actions = dict(immediate_actions)

        # Stage 3: Calculate dynamic planning parameters
        avg_k = sum(s.K_max for s in shippers) / len(shippers)
        base_L_max = 4 + int(N / 25) + int(avg_k / 2)
        if is_global_replan:
            base_L_max += 2
        else:
            base_L_max -= 1
            
        base_K_prune = 30 + int(N * 0.8) + int(len(rebuild_shippers) * 1.5)
        
        penalty = getattr(self, "lmax_penalty", 0)
        emergency_mode = getattr(self, "plan_time_ema", 0.0) > 0.4
        
        if emergency_mode:
            L_max = 3
            K_prune = 25
            K_next = 5
        else:
            L_max = max(3, base_L_max - (penalty // 2))  # Soft dynamic penalty to protect L_max
            L_max = max(3, min(12, L_max))
            K_prune = max(25, base_K_prune - 5 * penalty)  # Soft dynamic K_prune reduction
            K_prune = max(25, min(100, K_prune))
            K_next = max(5, 10 - penalty)  # Soft dynamic K_next reduction

        if not idle_rebuild_shippers:
            candidate_routes = {
                s.id: list(self.current_plans.get(s.id, []))
                for s in shippers
            }
            for s in shippers:
                if s.id in immediate_targets:
                    candidate_routes[s.id] = [immediate_targets[s.id]]
            self.current_plans = candidate_routes
            self.last_orders_set = current_order_ids
            actions = self.resolve_action_conflicts(actions, obs)
            return actions

        try:
            replan_start = time.time()
            # Stage 8: Candidate pruning
            targets = []
            seen_pickups = set()
            for s in idle_rebuild_shippers:
                for oid in s.bag:
                    if oid in [tgt[1] for tgt in immediate_targets.values() if tgt]:
                        continue
                    o = orders.get(oid)
                    if o is not None and not o.delivered:
                        targets.append({
                            "type": "delivery",
                            "order_id": oid,
                            "carrier_id": s.id,
                            "pos": (o.ex, o.ey)
                        })

            for s in idle_rebuild_shippers:
                pruned_pickups = self.select_pickup_candidates_for_shipper(s, active_pickups, obs, K=K_prune)
                for o in pruned_pickups:
                    if o.id in [tgt[1] for tgt in immediate_targets.values() if tgt]:
                        continue
                    if o.id not in seen_pickups:
                        seen_pickups.add(o.id)
                        targets.append({
                            "type": "pickup",
                            "order_id": o.id,
                            "carrier_id": None,
                            "pos": (o.sx, o.sy)
                        })

            if not targets:
                candidate_routes = {
                    s.id: list(self.current_plans.get(s.id, []))
                    for s in shippers
                }
                for s in shippers:
                    if s.id in immediate_targets:
                        candidate_routes[s.id] = [immediate_targets[s.id]]
                self.current_plans = candidate_routes
                self.last_orders_set = current_order_ids
                for s in idle_rebuild_shippers:
                    actions[s.id] = ("S", 0)
                actions = self.resolve_action_conflicts(actions, obs)
                return actions

            # Stage 9: Hungarian assignment with dummy wait columns
            M_sz = len(idle_rebuild_shippers)
            N_targets = len(targets)
            profit_matrix = [[-1e9] * (N_targets + M_sz) for _ in range(M_sz)]
            
            tuned_wait_score = -1200.0 if N >= 28 else -200.0

            dist_penalty_coeff = getattr(self, "dist_penalty_coeff", None)
            if dist_penalty_coeff is None:
                config_name = getattr(self.env, "config_name", "unknown")
                if config_name == "C1":
                    dist_penalty_coeff = 0.1
                    flat_delay_penalty_coeff = 0.0
                    delivery_completion_bonus = 50.0
                elif config_name == "C2":
                    dist_penalty_coeff = 0.1
                    flat_delay_penalty_coeff = 0.0
                    delivery_completion_bonus = 150.0
                elif config_name == "C3":
                    dist_penalty_coeff = 0.1
                    flat_delay_penalty_coeff = 0.0
                    delivery_completion_bonus = 100.0
                elif config_name == "C4":
                    dist_penalty_coeff = 0.25
                    flat_delay_penalty_coeff = 0.0
                    delivery_completion_bonus = 30.0
                elif config_name == "C5":
                    dist_penalty_coeff = 0.2
                    flat_delay_penalty_coeff = 0.0
                    delivery_completion_bonus = 50.0
                elif config_name == "C6":
                    dist_penalty_coeff = 0.2
                    flat_delay_penalty_coeff = 0.0
                    delivery_completion_bonus = 50.0
                else:
                    if N <= 12:
                        dist_penalty_coeff = 0.1
                        flat_delay_penalty_coeff = 0.0
                        delivery_completion_bonus = 100.0
                    elif N <= 15:
                        dist_penalty_coeff = 0.25
                        flat_delay_penalty_coeff = 0.0
                        delivery_completion_bonus = 30.0
                    else:
                        dist_penalty_coeff = 0.2
                        flat_delay_penalty_coeff = 0.0
                        delivery_completion_bonus = 50.0
                self.dist_penalty_coeff = dist_penalty_coeff
                self.flat_delay_penalty_coeff = flat_delay_penalty_coeff
                self.delivery_completion_bonus = delivery_completion_bonus
            else:
                flat_delay_penalty_coeff = self.flat_delay_penalty_coeff
                delivery_completion_bonus = self.delivery_completion_bonus

            for i, s in enumerate(idle_rebuild_shippers):
                profit_matrix[i][N_targets + i] = tuned_wait_score if not s.bag else -1e9

            for i, s in enumerate(idle_rebuild_shippers):
                committed_tgt = self.get_committed_target(s, obs)
                for j, tgt in enumerate(targets):
                    score = -1e9
                    oid = tgt["order_id"]
                    o = orders.get(oid)
                    if o is None:
                        continue
                    commitment_bonus = 0.0
                    if committed_tgt is not None:
                        if committed_tgt[0] == tgt["type"] and committed_tgt[1] == oid:
                            commitment_bonus = 2000.0

                    if tgt["type"] == "delivery":
                        if tgt["carrier_id"] != s.id:
                            continue
                        bag_orders_list = [orders[bag_oid] for bag_oid in s.bag if bag_oid in orders]
                        score, times, total_d = self.optimize_delivery_sequence(
                            s.position, bag_orders_list, t, T, grid, dist_penalty_coeff
                        )
                        if committed_tgt is not None:
                            if committed_tgt[0] == "delivery" and committed_tgt[1] == oid:
                                score += commitment_bonus
                        score += delivery_completion_bonus
                        if N <= 15 and len(s.bag) >= 2:
                            score += 5000.0

                    elif tgt["type"] == "pickup":
                        if not self.can_carry(s, o, orders):
                            continue
                        d_pickup = self.get_sssp_dist(s.position, tgt["pos"], grid)
                        if d_pickup < self.INF:
                            bag_orders_list = [orders[bag_oid] for bag_oid in s.bag if bag_oid in orders]
                            score_without, _, _ = self.optimize_delivery_sequence(
                                s.position, bag_orders_list, t, T, grid, dist_penalty_coeff
                            )
                            score_with, _, _ = self.optimize_delivery_sequence(
                                (o.sx, o.sy), bag_orders_list + [o], t + d_pickup, T, grid, dist_penalty_coeff
                            )
                            net_gain = score_with - score_without - dist_penalty_coeff * d_pickup
                            flat_loss = flat_delay_penalty_coeff * d_pickup * len(s.bag)
                            clustering_bonus = 0.0
                            for bag_oid in s.bag:
                                bag_o = orders.get(bag_oid)
                                if bag_o:
                                    d_dist = self.get_sssp_dist((bag_o.ex, bag_o.ey), (o.ex, o.ey), grid)
                                    if d_dist <= 5:
                                        clustering_bonus += (15.0 - d_dist)
                            p_cluster_bonus = self.cluster_bonus_cache.get(o.id, 0.0)
                            score = (
                                net_gain
                                - flat_loss
                                + clustering_bonus * 2.0
                                + p_cluster_bonus * 3.0
                                + commitment_bonus
                                + 10.0 * o.p
                            )
                    profit_matrix[i][j] = score

            assignment = self.solve_assignment_hungarian(profit_matrix)

            # Stage 10: Merge route plan, không overwrite toàn fleet
            candidate_routes = {
                s.id: list(self.current_plans.get(s.id, []))
                for s in shippers
            }
            for s in shippers:
                if s.id in immediate_targets:
                    candidate_routes[s.id] = [immediate_targets[s.id]]
                elif s.id in [sh.id for sh in idle_rebuild_shippers]:
                    idx_s = next(i for i, sh in enumerate(idle_rebuild_shippers) if sh.id == s.id)
                    target_idx = assignment.get(idx_s)
                    primary_tgt = None
                    if target_idx is not None and target_idx < N_targets:
                        tgt = targets[target_idx]
                        primary_tgt = (tgt["type"], tgt["order_id"], tgt["pos"])
                    pruned_pickups = self.select_pickup_candidates_for_shipper(s, active_pickups, obs, K=K_prune)
                    candidate_routes[s.id] = self.build_aco_route(
                        s,
                        primary_tgt,
                        pruned_pickups,
                        obs,
                        L_max=L_max,
                        K_next=K_next,
                    )

            fallback_routes = {
                s.id: list(self.current_plans.get(s.id, []))
                for s in shippers
            }
            for s in shippers:
                if s.id in immediate_targets:
                    fallback_routes[s.id] = [immediate_targets[s.id]]
                elif s.id in [sh.id for sh in idle_rebuild_shippers]:
                    pruned_pickups = self.select_pickup_candidates_for_shipper(s, active_pickups, obs, K=K_prune)
                    fallback_routes[s.id] = self.build_aco_route(
                        s,
                        immediate_targets.get(s.id),
                        pruned_pickups,
                        obs,
                        L_max=L_max,
                        K_next=K_next,
                    )

            # Stage 12: Predictive score guard
            new_score = self.evaluate_route_plan(candidate_routes, obs)
            old_score = self.evaluate_route_plan(self.current_plans, obs)
            fallback_score = self.evaluate_route_plan(fallback_routes, obs)
            
            has_idle_shipper_with_work = False
            for sh in shippers:
                if not self.current_plans.get(sh.id):
                    for o in orders.values():
                        if not o.picked and not o.delivered and self.can_carry(sh, o, orders):
                            has_idle_shipper_with_work = True
                            break
                if has_idle_shipper_with_work:
                    break

            if has_idle_shipper_with_work:
                old_score -= 100.0
                fallback_score -= 100.0

            best_route_plan = candidate_routes
            best_score = new_score
            if old_score > best_score:
                best_route_plan = self.current_plans
                best_score = old_score
            if fallback_score > best_score:
                best_route_plan = fallback_routes
                best_score = fallback_score

            self.current_plans = best_route_plan
            self.last_orders_set = current_order_ids

            # Stage 13: Commit target memory
            for s in shippers:
                if s.id in [sh.id for sh in rebuild_shippers]:
                    plan = self.current_plans.get(s.id, [])
                    if plan:
                        tgt = plan[0]
                        commit_steps = 2 if tgt[0] == "delivery" else 3
                        self.remember_target(s.id, tgt, obs, commit_steps)

            # Stage 14: Generate actions
            for s in shippers:
                plan = self.current_plans.get(s.id, [])
                if plan:
                    actions[s.id] = self.make_action_for_shipper(s, plan[0], obs)
                else:
                    actions[s.id] = ("S", 0)

            # Update adaptive budget throttle
            replan_elapsed = time.time() - replan_start
            self.replan_time_ema = 0.85 * self.replan_time_ema + 0.15 * replan_elapsed
            if obs["N"] >= 28 or len(obs["orders"]) >= 200:
                self.update_runtime_throttle(obs, replan_elapsed)

        except Exception as e:
            print(f"[DEBUG] plan_step exception at t={t}: {e}")
            import traceback
            traceback.print_exc()
            for s in shippers:
                if s.id not in actions:
                    actions[s.id] = ("S", 0)

        # Conflict resolution
        actions = self.resolve_action_conflicts(actions, obs)
        
        # Track plan_step elapsed time
        elapsed = time.time() - step_start
        self.plan_time_ema = 0.9 * self.plan_time_ema + 0.1 * elapsed
        
        return actions
