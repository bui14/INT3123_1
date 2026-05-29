from __future__ import annotations

import time
import random
from collections import deque
from typing import Dict, List, Optional, Tuple, Set

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class GreedyBFS(Solver):
    """
    GreedyBFS v23 - Global Iterative Task Assignment (GITA).
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        if not hasattr(env, "cfg"):
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
        super().__init__(env)
        self._dist_cache: Dict[Tuple[Position, Position], int] = {}
        self._move_cache: Dict[Tuple[Position, Position], Move] = {}
        self._bottlenecks: Optional[Set[Position]] = None

    def _neighbors(self, pos: Position) -> List[Tuple[Move, Position]]:
        res = []
        for m in MOVES:
            nxt = valid_next_pos(pos, m, self.grid)
            if nxt != pos: res.append((m, nxt))
        return res

    def _bfs(self, start: Position, goal: Position) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid): return None
        q = deque([start])
        parent = {start: (None, "S")}
        while q:
            cur = q.popleft()
            if cur == goal: return parent
            for m, nxt in self._neighbors(cur):
                if nxt not in parent:
                    parent[nxt] = (cur, m)
                    q.append(nxt)
        return None

    def _dist(self, a: Position, b: Position) -> int:
        if a == b: return 0
        key = (a, b)
        if key in self._dist_cache: return self._dist_cache[key]
        p = self._bfs(a, b)
        d = INF
        if p and b in p:
            d, cur = 0, b
            while cur != a:
                prev, _ = p[cur]
                cur, d = prev, d + 1
        self._dist_cache[key] = d
        return d

    def _next_move(self, start: Position, goal: Position) -> Move:
        if start == goal: return "S"
        key = (start, goal)
        if key in self._move_cache: return self._move_cache[key]
        p = self._bfs(start, goal)
        m = "S"
        if p and goal in p:
            cur = goal
            while True:
                prev, mov = p[cur]
                if prev == start:
                    m = mov; break
                cur = prev
        self._move_cache[key] = m
        return m

    def _is_bottleneck(self, pos: Position) -> bool:
        if self._bottlenecks is None:
            self._bottlenecks = set()
            for r in range(len(self.grid)):
                for c in range(len(self.grid[0])):
                    p = (r, c)
                    if is_valid_cell(p, self.grid) and len(self._neighbors(p)) <= 2:
                        self._bottlenecks.add(p)
        return pos in self._bottlenecks

    def _now(self, obs: dict) -> int: return int(obs.get("t", 0))
    def _all_carried(self, shippers: List[Shipper]) -> Set[int]:
        res = set()
        for s in shippers: res.update(s.bag)
        return res

    def _T(self) -> int: return int(getattr(self.env, "T", 1))

    def _est_reward(self, o: Order, t_del: int) -> float:
        rb = {o.w <= 0.2: 4.0, o.w <= 3.0: 10.0, o.w <= 10.0: 15.0, o.w <= 30.0: 20.0}.get(True, 30.0)
        α = {1: 1.0, 2: 2.0, 3: 3.0}[o.p]
        β = {1: 0.1, 2: 0.3, 3: 0.5}[o.p]
        T = self._T()
        if t_del <= o.et:
            return α * rb * (1.0 + max(0.0, (o.et - t_del) / max(o.et, 1)))
        return β * rb * max(0.0, 1.0 - (t_del - o.et) / T)

    def _min_slack(self, s: Shipper, orders: Dict[int, Order], now: int) -> int:
        ms = INF
        for oid in s.bag:
            o = orders.get(oid)
            if o and not o.delivered:
                ms = min(ms, o.et - (now + self._dist(s.position, (o.ex, o.ey)) + 1))
        return ms

    def _best_mission(self, s: Shipper, orders: Dict[int, Order], now: int, res_pick: Set[int], carried: Set[int]) -> Tuple[float, List[Tuple]]:
        targets = []
        for oid in s.bag: targets.append(("D", oid, (orders[oid].ex, orders[oid].ey)))
        if len(s.bag) < s.K_max:
            for o in orders.values():
                if o.delivered or o.picked or o.id in carried or o.id in res_pick: continue
                if s.can_carry(o, orders): targets.append(("P", o.id, (o.sx, o.sy)))
        
        if not targets: return -INF, []
        targets.sort(key=lambda x: self._dist(s.position, x[2]))
        
        best_s, best_c = -INF, []
        is_large = (len(self.grid) >= 20)
        breadth1 = 48 if is_large else 32
        breadth2 = 16 if is_large else 10
        p_bonus = 25.0 if is_large else 12.0
        dist_offset = 1.5 if is_large else 0.5

        for t1 in targets[:breadth1]:
            d1 = self._dist(s.position, t1[2])
            if d1 >= INF: continue
            arr1 = now + d1 + 1
            o1 = orders[t1[1]]
            s1 = self._est_reward(o1, arr1) if t1[0] == "D" else self._est_reward(o1, arr1 + self._dist(t1[2], (o1.ex, o1.ey)) + 1) * 0.8
            
            max_f = 0.0
            new_bag = [oid for oid in s.bag if oid != t1[1]] if t1[0] == "D" else list(s.bag) + [t1[1]]
            t2s = []
            for oid in new_bag: t2s.append(("D", oid, (orders[oid].ex, orders[oid].ey)))
            if len(new_bag) < s.K_max:
                for o in orders.values():
                    if o.delivered or o.picked or o.id in carried or o.id in res_pick or o.id == t1[1]: continue
                    if s.can_carry(o, orders): t2s.append(("P", o.id, (o.sx, o.sy)))
            
            t2s.sort(key=lambda x: self._dist(t1[2], x[2]))
            for t2 in t2s[:breadth2]:
                d2 = self._dist(t1[2], t2[2])
                if d2 < INF:
                    arr2 = arr1 + d2 + 1
                    o2 = orders[t2[1]]
                    s2 = self._est_reward(o2, arr2) if t2[0] == "D" else self._est_reward(o2, arr2 + self._dist(t2[2], (o2.ex, o2.ey)) + 1) * 0.8
                    max_f = max(max_f, s2)

            score = (s1 + max_f * 0.6 + o1.p * p_bonus) / (d1 + dist_offset)
            if score > best_s: best_s, best_c = score, [t1]
                
        return best_s, best_c

    def _avoid_collision(self, s: Shipper, act: Action, reserved: Set[Position], goal: Optional[Position]) -> Action:
        move, op = act
        nxt = valid_next_pos(s.position, move, self.grid)
        if nxt not in reserved: return act

        alts = []
        for m in MOVES:
            anxt = valid_next_pos(s.position, m, self.grid)
            if anxt != s.position and anxt not in reserved:
                alts.append((self._dist(anxt, goal) if goal else 0, self._is_bottleneck(anxt), m))
        if alts:
            alts.sort()
            return (alts[0][2], 0)
        return ("S", 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders, shippers, now = obs["orders"], obs["shippers"], self._now(obs)
        actions, final_actions, res_pick, res_pos = {}, {}, set(), set()
        carried = self._all_carried(shippers)
        
        # 1. Immediate actions (uncontested)
        unassigned_s = []
        for s in shippers:
            # Deliver if at target
            if any(oid in s.bag and not orders[oid].delivered and s.position == (orders[oid].ex, orders[oid].ey) for oid in s.bag):
                final_actions[s.id] = ("S", 2); res_pos.add(s.position); continue
            # Pickup if at target – prioritized by env rules
            candidates = [o for o in orders.values() if not o.delivered and not o.picked and o.id not in carried and s.can_carry(o, orders) and s.position == (o.sx, o.sy)]
            if candidates:
                p_best = min(candidates, key=lambda o: (-o.p, o.et, o.id))
                final_actions[s.id] = ("S", 1); res_pos.add(s.position); carried.add(p_best.id); res_pick.add(p_best.id); continue
            unassigned_s.append(s)

        # 2. Iterative Global Task Assignment
        while unassigned_s:
            best_score, best_pair = -INF, None # (shipper, chain)
            for s in unassigned_s:
                score, chain = self._best_mission(s, orders, now, res_pick, carried)
                if score > best_score:
                    best_score, best_pair = score, (s, chain)
            
            if not best_pair or best_score <= -INF: break
            
            s, chain = best_pair
            dest = chain[0][2]
            actions[s.id] = (self._next_move(s.position, dest), 0, dest)
            if chain[0][0] == "P": res_pick.add(chain[0][1])
            unassigned_s.remove(s)

        # 3. Fallback for remaining
        for s in unassigned_s:
            fallback = next((o for o in orders.values() if not o.delivered and not o.picked and o.id not in carried and o.id not in res_pick and s.can_carry(o, orders)), None)
            if fallback:
                dest = (fallback.sx, fallback.sy)
                actions[s.id] = (self._next_move(s.position, dest), 0, dest)
                res_pick.add(fallback.id)
            else:
                actions[s.id] = ("S", 0, None)

        # 4. Collision Avoidance based on assignment order (urgency)
        sorted_s = sorted(shippers, key=lambda x: (self._min_slack(x, orders, now), -len(x.bag), x.id))
        for s in sorted_s:
            if s.id in final_actions: continue
            move, op, goal = actions[s.id]
            act = self._avoid_collision(s, (move, op), res_pos, goal)
            final_actions[s.id] = act
            res_pos.add(valid_next_pos(s.position, act[0], self.grid))
            
        return final_actions

    def run(self) -> dict:
        st = time.time(); obs = self.env.reset()
        while not obs.get("done", False):
            obs, _, done, _ = self.env.step(self._decide_actions(obs))
            if done: break
        return self.env.result(self.method_name, elapsed_sec=time.time() - st)