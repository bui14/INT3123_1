from __future__ import annotations

import random
import time
from collections import OrderedDict, deque
from itertools import permutations
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, valid_next_pos
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
INF = 10**8
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


def _estimate_average_shortest_path(grid: List[List[int]], bfs: "GridPaths", sample_size: int = 40) -> float:
    free_cells = [(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 0]
    if len(free_cells) < 2:
        return 1.0
    rng = random.Random(42)
    total_dist = 0
    valid_pairs = 0
    for _ in range(sample_size):
        u = rng.choice(free_cells)
        v = rng.choice(free_cells)
        if u != v:
            d = bfs.distance(u, v)
            if d < INF:
                total_dist += d
                valid_pairs += 1
    if valid_pairs == 0:
        return float(len(grid))
    return total_dist / valid_pairs


def _get_cbs_params(n: int, c: int, t: int, grid: List[List[int]], bfs: "GridPaths") -> Dict[str, Any]:
    asd = _estimate_average_shortest_path(grid, bfs)
    horizon = int(max(8, min(18, 5 + c + asd // 1.2)))
    route_travel_penalty = 0.05 + min(0.08, 0.02 * max(0, c - 2))
    route_late_penalty = 0.42 + min(0.35, 0.012 * asd)
    if asd > 16.5:
        scale = 16.5 / asd
        route_travel_penalty *= scale
        route_late_penalty *= scale
    return {
        "horizon": horizon,
        "route_travel_penalty": route_travel_penalty,
        "route_late_penalty": route_late_penalty,
    }


def evaluate_detour_net_reward(
    shipper: Shipper,
    passing: Order,
    current_delivery: Order,
    router: "GridPaths",
    now: int,
    T: int,
) -> float:
    primary_drop = (current_delivery.ex, current_delivery.ey)
    new_drop = (passing.ex, passing.ey)
    primary_dist = router.distance(shipper.position, primary_drop)
    onward = router.distance(primary_drop, new_drop)
    if primary_dist >= INF or onward >= INF:
        return -INF
    finish = now + primary_dist + onward
    if finish >= T:
        return -INF
    reward = delivery_reward(passing, finish, T)
    late = max(0, finish - passing.et)
    return reward + 28.0 * reward / (onward + 8.0) - 0.12 * onward - 0.28 * late


def is_pickup_safe(
    shipper: Shipper,
    order: Order,
    orders: Dict[int, Order],
    router: "GridPaths",
    now: int,
    T: int,
) -> bool:
    pickup = (order.sx, order.sy)
    d = router.distance(shipper.position, pickup)
    if d >= INF:
        return False
    finish_pickup = now + d
    for oid in shipper.bag:
        o = orders.get(oid)
        if o and not o.delivered:
            d_del = router.distance(pickup, (o.ex, o.ey))
            if d_del >= INF:
                return False
            if finish_pickup + d_del > o.et + T // 4:
                return False
    return True


def resolve_collisions_and_blocks(
    shippers: List[Shipper],
    actions: Dict[int, Action],
    grid: List[List[int]],
    orders: Dict[int, Order],
    pickup_at_fn: Callable,
    deliverable_fn: Callable,
    target_positions: Dict[int, Position],
    dist_fn: Callable,
    slacks: Dict[int, float] = {},
    allow_unblock: bool = False,
) -> Dict[int, Action]:
    by_id = {s.id: s for s in shippers}

    def next_pos(sid: int) -> Position:
        move, _ = actions[sid]
        return valid_next_pos(by_id[sid].position, move, grid)

    intended: Dict[int, Position] = {sid: next_pos(sid) for sid in actions}

    changed = True
    rounds = 0
    while changed and rounds < 5:
        changed = False
        rounds += 1
        cell_owners: Dict[Position, int] = {}
        for sid in sorted(actions, key=lambda i: (slacks.get(i, INF), i)):
            pos = intended[sid]
            if pos in cell_owners:
                s = by_id[sid]
                goal = target_positions.get(sid)
                blocked_cells = set(cell_owners.keys()) | {by_id[i].position for i in actions}
                alts = [
                    (dist_fn(anxt, goal) if goal else 0, m, anxt)
                    for m in MOVES
                    for anxt in [valid_next_pos(s.position, m, grid)]
                    if anxt != s.position and anxt not in blocked_cells
                ]
                if alts:
                    alts.sort()
                    actions[sid] = (alts[0][1], 0)
                    intended[sid] = alts[0][2]
                else:
                    actions[sid] = ("S", 0)
                    intended[sid] = s.position
                changed = True
            else:
                cell_owners[pos] = sid

        pos_list = list(intended.items())
        for i, (sid1, nxt1) in enumerate(pos_list):
            for sid2, nxt2 in pos_list[i + 1:]:
                s1, s2 = by_id[sid1], by_id[sid2]
                if nxt1 == s2.position and nxt2 == s1.position:
                    loser = sid2 if (slacks.get(sid1, INF), sid1) < (slacks.get(sid2, INF), sid2) else sid1
                    s_loser = by_id[loser]
                    goal_loser = target_positions.get(loser)
                    blocked = {nxt1, nxt2, s1.position, s2.position}
                    alts = [
                        (dist_fn(anxt, goal_loser) if goal_loser else 0, m, anxt)
                        for m in MOVES
                        for anxt in [valid_next_pos(s_loser.position, m, grid)]
                        if anxt not in blocked
                    ]
                    if alts:
                        alts.sort()
                        actions[loser] = (alts[0][1], 0)
                        intended[loser] = alts[0][2]
                    else:
                        actions[loser] = ("S", 0)
                        intended[loser] = s_loser.position
                    changed = True

    return actions


class GridPaths:
    """Shortest-path distance fields cached by destination (BFS from goal)."""

    def __init__(self, grid: List[List[int]], cache_limit: int):
        self.grid = grid
        self.cache_limit = cache_limit
        self.neighbor_map: Dict[Position, List[Tuple[Move, Position]]] = {}
        self.fields: "OrderedDict[Position, Dict[Position, int]]" = OrderedDict()
        for r, row in enumerate(grid):
            for c, value in enumerate(row):
                if value != 0:
                    continue
                pos = (r, c)
                neighbors: List[Tuple[Move, Position]] = []
                for move in MOVES:
                    nxt = valid_next_pos(pos, move, grid)
                    if nxt != pos:
                        neighbors.append((move, nxt))
                self.neighbor_map[pos] = neighbors

    def neighbors(self, pos: Position, include_wait: bool = False) -> Iterable[Tuple[Move, Position]]:
        if include_wait:
            yield "S", pos
        yield from self.neighbor_map.get(pos, [])

    def _field(self, goal: Position) -> Dict[Position, int]:
        cached = self.fields.get(goal)
        if cached is not None:
            self.fields.move_to_end(goal)
            return cached
        dist: Dict[Position, int] = {}
        if goal in self.neighbor_map:
            dist[goal] = 0
            q = deque([goal])
            while q:
                pos = q.popleft()
                nd = dist[pos] + 1
                for _, nxt in self.neighbor_map[pos]:
                    if nxt not in dist:
                        dist[nxt] = nd
                        q.append(nxt)
        self.fields[goal] = dist
        self.fields.move_to_end(goal)
        while len(self.fields) > self.cache_limit:
            self.fields.popitem(last=False)
        return dist

    def distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        return self._field(goal).get(start, INF)

    def move_between(self, start: Position, nxt: Position) -> Move:
        if start == nxt:
            return "S"
        for move, pos in self.neighbor_map.get(start, []):
            if pos == nxt:
                return move
        return "S"


class MAPDCBSSolver(Solver):
    """Online MAPD task assignment with cached BFS routes and CBS-lite reservations."""

    method_name = "MAPDCBSSolver"

    _W_ROUGH_REWARD = 80.0
    _W_EXACT_REWARD = 95.0
    _W_BACKLOG_REWARD = 16.0
    _PROACTIVE_THRESHOLD = 0.15

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
        self.grid: List[List[int]] = [[0]]
        self.router: Optional[GridPaths] = None
        self.N = 1
        self.C = 1
        self.T = 1
        self.horizon = 5
        self.pickup_tasks: Dict[int, int] = {}
        self.replanned_tasks: Dict[int, int] = {}
        self.delivery_commitments: Dict[int, int] = {}
        self.bundle_followups: Dict[int, int] = {}
        self.failed_assignment_retry: Dict[int, int] = {}
        self.run_deadline = 0.0
        self.repairs = 0
        self.repair_search_budget = 0

    def _configure(self, obs: dict) -> None:
        self.N = int(obs["N"])
        self.C = int(obs["C"])
        self.T = int(obs["T"])
        self.grid = obs["grid"]
        cache_limit = max(32, min(88, 24 + 2 * self.C))
        if self.C >= 15:
            cache_limit = max(48, min(180, 24 + 5 * self.C))
        self.router = GridPaths(self.grid, cache_limit)
        params = _get_cbs_params(self.N, self.C, self.T, self.grid, self.router)
        self.horizon = params["horizon"]
        self.pickup_tasks.clear()
        self.replanned_tasks.clear()
        self.delivery_commitments.clear()
        self.bundle_followups.clear()
        self.failed_assignment_retry.clear()

    def _mdist(self, a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _endgame_window(self) -> int:
        extra = 2 * max(0, self.N - 30) if self.C >= 15 else 0
        return max(20, min(320, self.T // 10 + extra))

    def _carried(self, shipper: Shipper, orders: Dict[int, Order]) -> List[Order]:
        return [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]

    def _carried_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(order.w for order in self._carried(shipper, orders))

    def _can_take_light(self, shipper: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
        return (
            not order.picked
            and not order.delivered
            and len(shipper.bag) < shipper.K_max
            and self._carried_weight(shipper, orders) + order.w <= shipper.W_max
        )

    def _pickup_at(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> Optional[Order]:
        candidates = [
            order
            for order in orders.values()
            if (order.sx, order.sy) == pos and self._can_take_light(shipper, order, orders)
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda order: (-order.p, order.et, order.id))

    def _deliverable(self, shipper: Shipper, orders: Dict[int, Order], pos: Position) -> bool:
        return any((order.ex, order.ey) == pos for order in self._carried(shipper, orders))

    def _visible_density(self, orders: Dict[int, Order]) -> Tuple[Dict[Position, int], Dict[Position, int]]:
        pickups: Dict[Position, int] = {}
        drops: Dict[Position, int] = {}
        for order in orders.values():
            if order.delivered:
                continue
            if not order.picked:
                pickup = (order.sx, order.sy)
                pickups[pickup] = pickups.get(pickup, 0) + 1
            drop = (order.ex, order.ey)
            drops[drop] = drops.get(drop, 0) + 1
        return pickups, drops

    def _cluster_value(self, cell: Position, counts: Dict[Position, int]) -> float:
        value = float(counts.get(cell, 0))
        r, c = cell
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2)):
            value += (0.45 if abs(dr) + abs(dc) == 1 else 0.20) * counts.get((r + dr, c + dc), 0)
        return value

    def _delivery_target(self, shipper: Shipper, orders: Dict[int, Order], now: int) -> Optional[Order]:
        carried = self._carried(shipper, orders)
        if not carried:
            self.delivery_commitments.pop(shipper.id, None)
            return None
        committed_oid = self.delivery_commitments.get(shipper.id)
        if committed_oid is not None:
            committed = next((order for order in carried if order.id == committed_oid), None)
            if committed is not None:
                return committed
            self.delivery_commitments.pop(shipper.id, None)
        best: Optional[Order] = None
        best_key: Optional[Tuple[float, int, int]] = None
        for order in carried:
            goal = (order.ex, order.ey)
            dist = self.router.distance(shipper.position, goal)
            if dist >= INF:
                continue
            finish = now + dist
            slack = order.et - finish
            same_drop = sum(1 for other in carried if (other.ex, other.ey) == goal)
            score = (
                delivery_reward(order, finish, self.T)
                + 12.0 * order.p
                + 5.0 * same_drop
                + max(0.0, 30.0 - slack) * (0.28 + 0.09 * order.p)
                - 0.10 * dist
            )
            key = (score, -order.et, -order.id)
            if best_key is None or key > best_key:
                best_key = key
                best = order
        return best

    def _passing_pickup_is_worthwhile(
        self,
        shipper: Shipper,
        current_delivery: Order,
        passing: Order,
        orders: Dict[int, Order],
        now: int,
    ) -> bool:
        if (passing.ex, passing.ey) == (current_delivery.ex, current_delivery.ey):
            return True
        score = evaluate_detour_net_reward(shipper, passing, current_delivery, self.router, now, self.T)
        return score >= 10.0

    def _can_add_orders(self, shipper: Shipper, additions: List[Order], orders: Dict[int, Order]) -> bool:
        if any(order.picked or order.delivered for order in additions):
            return False
        carried = self._carried(shipper, orders)
        return (
            len(carried) + len(additions) <= shipper.K_max
            and sum(order.w for order in carried) + sum(order.w for order in additions) <= shipper.W_max
        )

    def _best_delivery_plan(
        self,
        start: Position,
        now: int,
        cargo: List[Order],
    ) -> Optional[Tuple[float, Set[int], int]]:
        grouped: Dict[Position, List[Order]] = {}
        for order in cargo:
            grouped.setdefault((order.ex, order.ey), []).append(order)
        if not grouped:
            return (0.0, set(), now)
        drops = list(grouped)
        if len(drops) <= 5:
            sequences = permutations(drops)
        else:
            sequences = [
                tuple(
                    sorted(
                        drops,
                        key=lambda drop: (
                            min(order.et for order in grouped[drop]),
                            self.router.distance(start, drop),
                            drop,
                        ),
                    )
                )
            ]
        best: Optional[Tuple[float, Set[int], int]] = None
        best_key: Optional[Tuple[int, float, int]] = None
        for sequence in sequences:
            pos = start
            finish = now
            travel = 0
            reward = 0.0
            on_time: Set[int] = set()
            feasible = True
            for drop in sequence:
                dist = self.router.distance(pos, drop)
                if dist >= INF:
                    feasible = False
                    break
                travel += dist
                finish += dist
                for order in grouped[drop]:
                    reward += delivery_reward(order, finish, self.T)
                    if finish <= order.et:
                        on_time.add(order.id)
                pos = drop
            if not feasible:
                continue
            value = reward - 0.10 * travel
            key = (len(on_time), value, -finish)
            if best_key is None or key > best_key:
                best_key = key
                best = (value, on_time, finish)
        return best

    def _best_loaded_bundle_pickup(
        self,
        shipper: Shipper,
        delivery: Order,
        orders: Dict[int, Order],
        now: int,
        excluded: Set[int],
        deadline: float,
    ) -> Optional[Order]:
        followup_oid = self.bundle_followups.get(shipper.id)
        if followup_oid is None:
            return None
        carried = self._carried(shipper, orders)
        if len(carried) != 1 or len(carried) >= shipper.K_max:
            self.bundle_followups.pop(shipper.id, None)
            return None
        baseline = self._best_delivery_plan(shipper.position, now, carried)
        if baseline is None:
            return None
        baseline_value, baseline_on_time, _ = baseline
        committed_oid = self.pickup_tasks.get(shipper.id)
        rough: List[Tuple[float, int, Order]] = []
        for order in orders.values():
            if time.perf_counter() > deadline:
                break
            if (
                order.id != followup_oid
                or order.id in excluded
                or order.id in shipper.bag
                or not self._can_add_orders(shipper, [order], orders)
            ):
                continue
            pickup = (order.sx, order.sy)
            cheap_dist = self._mdist(shipper.position, pickup)
            cheap_trip = self._mdist(pickup, (order.ex, order.ey))
            cheap_finish = now + max(1, cheap_dist) + cheap_trip
            if cheap_finish >= self.T:
                continue
            rough_value = (
                delivery_reward(order, cheap_finish, self.T)
                + 8.0 * order.p
                - 0.18 * cheap_dist
                - 0.08 * cheap_trip
                + (18.0 if order.id == committed_oid else 0.0)
            )
            rough.append((rough_value, order.id, order))
        rough.sort(key=lambda item: (-item[0], item[1]))
        candidate_cap = max(5, min(12, 5 + self.C // 3))
        if committed_oid is not None:
            rough.sort(key=lambda item: (item[2].id != committed_oid, -item[0], item[1]))
        best: Optional[Order] = None
        best_key: Optional[Tuple[float, int, int]] = None
        for _, _, order in rough[:candidate_cap]:
            if time.perf_counter() > deadline:
                break
            pickup = (order.sx, order.sy)
            d_pickup = self.router.distance(shipper.position, pickup)
            if d_pickup >= INF:
                continue
            pickup_time = now + max(1, d_pickup)
            plan = self._best_delivery_plan(pickup, pickup_time, carried + [order])
            if plan is None:
                continue
            value, on_time, finish = plan
            if not baseline_on_time.issubset(on_time) or order.id not in on_time:
                continue
            gain = value - baseline_value - 0.10 * d_pickup
            if gain <= 0.0:
                continue
            if order.id == committed_oid:
                return order
            key = (gain + (8.0 if order.id == committed_oid else 0.0), order.p, -finish)
            if best_key is None or key > best_key:
                best_key = key
                best = order
        if best is None:
            self.bundle_followups.pop(shipper.id, None)
        return best

    def _pickup_before_committed_target(
        self,
        shipper: Shipper,
        primary: Order,
        orders: Dict[int, Order],
        now: int,
    ) -> Optional[Order]:
        auxiliary = self._pickup_at(shipper, orders, shipper.position)
        if auxiliary is None or auxiliary.id == primary.id:
            return None
        if auxiliary.p >= primary.p:
            return None
        if (auxiliary.sx, auxiliary.sy) == (primary.sx, primary.sy):
            return None
        if not self._can_add_orders(shipper, [auxiliary, primary], orders):
            return None
        direct_delivery = self.router.distance(shipper.position, (auxiliary.ex, auxiliary.ey))
        if direct_delivery >= INF or auxiliary.et - (now + 1 + direct_delivery) > 12:
            return None
        if self.router.distance((auxiliary.ex, auxiliary.ey), (primary.ex, primary.ey)) > 8:
            return None
        primary_pickup = (primary.sx, primary.sy)
        to_primary = self.router.distance(shipper.position, primary_pickup)
        if to_primary >= INF:
            return None
        pickup_time = now + 1 + max(1, to_primary)
        plan = self._best_delivery_plan(primary_pickup, pickup_time, [auxiliary, primary])
        if plan is None:
            return None
        _, on_time, _ = plan
        return auxiliary if auxiliary.id in on_time and primary.id in on_time else None

    def _rough_pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        now: int,
        pickup_counts: Dict[Position, int],
        drop_counts: Dict[Position, int],
    ) -> float:
        if not self._can_take_light(shipper, order, orders):
            return -INF
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        travel = self._mdist(shipper.position, pickup) + self._mdist(pickup, drop) + 1
        if now + travel >= self.T:
            return -INF
        finish = now + travel
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        density = self._cluster_value(pickup, pickup_counts) + 0.35 * self._cluster_value(drop, drop_counts)
        return (
            reward
            + self._W_ROUGH_REWARD * reward / (travel + 8.0)
            + 4.0 * order.p
            + 1.5 * density
            - 0.15 * travel
            - 0.80 * late
        )

    def _exact_pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        now: int,
        pickup_counts: Dict[Position, int],
        drop_counts: Dict[Position, int],
    ) -> float:
        if not self._can_take_light(shipper, order, orders):
            return -INF
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.router.distance(shipper.position, pickup)
        d2 = self.router.distance(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return -INF
        travel = d1 + d2 + 1
        finish = now + travel
        if finish >= self.T:
            return -INF
        reward = delivery_reward(order, finish, self.T)
        late = max(0, finish - order.et)
        slack = order.et - finish
        density = self._cluster_value(pickup, pickup_counts) + 0.35 * self._cluster_value(drop, drop_counts)
        persistence = 6.0 if (
            self.pickup_tasks.get(shipper.id) == order.id
            or self.replanned_tasks.get(shipper.id) == order.id
        ) else 0.0
        return (
            reward
            + self._W_EXACT_REWARD * reward / (travel + 8.0)
            + 5.0 * order.p
            + 1.7 * density
            + min(8.0, max(0.0, slack) * 0.08)
            + persistence
            - 0.17 * travel
            - 0.90 * late
        )

    def _backlog_pickup_score(
        self,
        shipper: Shipper,
        order: Order,
        orders: Dict[int, Order],
        now: int,
        pickup_counts: Dict[Position, int],
    ) -> float:
        if not self._can_take_light(shipper, order, orders):
            return -INF
        pickup = (order.sx, order.sy)
        drop = (order.ex, order.ey)
        d1 = self.router.distance(shipper.position, pickup)
        d2 = self.router.distance(pickup, drop)
        if d1 >= INF or d2 >= INF:
            return -INF
        travel = d1 + d2 + 1
        finish = now + travel
        if finish >= self.T:
            return -INF
        reward = delivery_reward(order, finish, self.T)
        density = self._cluster_value(pickup, pickup_counts)
        return reward + self._W_BACKLOG_REWARD * reward / (travel + 8.0) + 0.4 * density - 0.14 * travel

    def _valid_task(self, shipper: Shipper, orders: Dict[int, Order], used_orders: Set[int]) -> Optional[Order]:
        oid = self.pickup_tasks.get(shipper.id)
        order = orders.get(oid) if oid is not None else None
        if (
            order is None
            or oid in used_orders
            or order.picked
            or order.delivered
            or not self._can_take_light(shipper, order, orders)
        ):
            self.pickup_tasks.pop(shipper.id, None)
            return None
        return order

    def _target_priority(self, shipper: Shipper, kind: str, order: Order, now: int) -> float:
        goal = (order.ex, order.ey) if kind == "deliver" else (order.sx, order.sy)
        dist = self.router.distance(shipper.position, goal)
        if dist >= INF:
            return -INF
        if kind == "deliver":
            slack = order.et - (now + dist)
            return (
                1000.0
                + delivery_reward(order, now + dist, self.T)
                + 18.0 * order.p
                + max(0.0, 45.0 - slack) * (0.4 + 0.1 * order.p)
                - 0.08 * dist
            )
        slack = order.et - (now + dist)
        return 50.0 + 5.0 * order.p + max(0.0, 20.0 - slack) * 0.1 - 0.06 * dist

    # ------------------------------------------------------------------
    # Assignment helpers
    # ------------------------------------------------------------------

    def _partition_idle_shippers(
        self,
        idle: List[Shipper],
        orders: Dict[int, Order],
        now: int,
        used_orders: Set[int],
        used_cells: Dict[Position, int],
    ) -> List[Shipper]:
        reconsider_dist = max(8, min(14, self.N // 7 + 4))
        reconsider_period = max(4, min(10, self.N // 12 + 4))
        if self.N >= 30 and reconsider_period <= 8:
            reconsider_period = max(3, reconsider_period - 1)
        pending: List[Shipper] = []
        for shipper in sorted(idle, key=lambda s: s.id):
            order = self._valid_task(shipper, orders, used_orders)
            if order is None:
                pending.append(shipper)
                continue
            if (
                now % reconsider_period == 0
                and self.router.distance(shipper.position, (order.sx, order.sy)) > reconsider_dist
            ):
                self.replanned_tasks[shipper.id] = order.id
                self.pickup_tasks.pop(shipper.id, None)
                pending.append(shipper)
                continue
            pickup = (order.sx, order.sy)
            used_orders.add(order.id)
            used_cells[pickup] = used_cells.get(pickup, 0) + 1
        return pending

    def _collect_candidates(
        self,
        pending: List[Shipper],
        orders: Dict[int, Order],
        now: int,
        step_deadline: float,
        pickup_counts: Dict[Position, int],
        drop_counts: Dict[Position, int],
        used_orders: Set[int],
        exact_cap: int,
    ) -> List[Tuple[float, int, int, Order]]:
        candidates: List[Tuple[float, int, int, Order]] = []
        for shipper in pending:
            if time.perf_counter() > step_deadline:
                break
            rough: List[Tuple[float, int, int, Order]] = []
            for order in orders.values():
                if order.id in used_orders or order.picked or order.delivered:
                    continue
                score = self._rough_pickup_score(shipper, order, orders, now, pickup_counts, drop_counts)
                if score > 0.0:
                    rough.append((score, order.et, order.id, order))
            rough.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, _, _, order in rough[:exact_cap]:
                if time.perf_counter() > step_deadline:
                    break
                score = self._exact_pickup_score(shipper, order, orders, now, pickup_counts, drop_counts)
                if score > 0.0:
                    candidates.append((score, shipper.id, order.id, order))
        return candidates

    def _coordinate_assign(
        self,
        candidates: List[Tuple[float, int, int, Order]],
        used_orders: Set[int],
        used_cells: Dict[Position, int],
        pickup_counts: Dict[Position, int],
    ) -> None:
        scores_by_order: Dict[int, List[float]] = {}
        for score, _, oid, _ in candidates:
            scores_by_order.setdefault(oid, []).append(score)
        regrets: Dict[int, float] = {}
        for oid, values in scores_by_order.items():
            values.sort(reverse=True)
            regrets[oid] = values[0] - (values[1] if len(values) > 1 else 0.0)

        use_coordination = self.N >= 20
        regret_weight = 0.35 if use_coordination else 0.20
        cluster_assignment: Dict[Position, int] = {}

        for score, sid, oid, order in sorted(
            candidates,
            key=lambda item: (item[0] + regret_weight * regrets.get(item[2], 0.0), item[0], item[3].p, -item[3].et, -item[2]),
            reverse=True,
        ):
            if sid in self.pickup_tasks or oid in used_orders:
                continue
            pickup = (order.sx, order.sy)
            best_cluster = None
            if use_coordination:
                max_c = -1
                for cluster_cell, count in pickup_counts.items():
                    if self._mdist(pickup, cluster_cell) <= 3 and count > max_c:
                        max_c = count
                        best_cluster = cluster_cell
                if best_cluster and cluster_assignment.get(best_cluster, 0) >= max(1, pickup_counts.get(best_cluster, 1)):
                    continue
            cell_limit = max(1, min(3, pickup_counts.get(pickup, 1)))
            if used_cells.get(pickup, 0) >= cell_limit:
                continue
            self.pickup_tasks[sid] = oid
            used_orders.add(oid)
            used_cells[pickup] = used_cells.get(pickup, 0) + 1
            if use_coordination and best_cluster:
                cluster_assignment[best_cluster] = cluster_assignment.get(best_cluster, 0) + 1

    def _assign_backlog_orders(
        self,
        pending: List[Shipper],
        orders: Dict[int, Order],
        now: int,
        step_deadline: float,
        used_orders: Set[int],
        used_cells: Dict[Position, int],
        pickup_counts: Dict[Position, int],
        exact_cap: int,
    ) -> None:
        backlog: List[Tuple[float, int, int, Order]] = []
        backlog_cap = exact_cap if self.C >= 15 else max(3, exact_cap // 2)
        for shipper in pending:
            if now < self.T - self._endgame_window():
                continue
            if shipper.id in self.pickup_tasks or time.perf_counter() > step_deadline:
                continue
            rough2: List[Tuple[float, int, int, Order]] = []
            for order in orders.values():
                if order.id in used_orders or order.picked or order.delivered or not self._can_take_light(shipper, order, orders):
                    continue
                travel = (
                    self._mdist(shipper.position, (order.sx, order.sy))
                    + self._mdist((order.sx, order.sy), (order.ex, order.ey))
                    + 1
                )
                finish = now + travel
                if finish >= self.T:
                    continue
                value = delivery_reward(order, finish, self.T) - 0.10 * travel
                rough2.append((value, order.et, order.id, order))
            rough2.sort(key=lambda item: (-item[0], item[1], item[2]))
            for _, _, _, order in rough2[:backlog_cap]:
                if time.perf_counter() > step_deadline:
                    break
                score = self._backlog_pickup_score(shipper, order, orders, now, pickup_counts)
                if score > 0.0:
                    backlog.append((score, shipper.id, order.id, order))

        for score, sid, oid, order in sorted(backlog, key=lambda item: (item[0], item[3].p, -item[3].et), reverse=True):
            if sid in self.pickup_tasks or oid in used_orders:
                continue
            pickup = (order.sx, order.sy)
            cell_limit = max(1, min(3, pickup_counts.get(pickup, 1)))
            if used_cells.get(pickup, 0) >= cell_limit:
                continue
            self.pickup_tasks[sid] = oid
            used_orders.add(oid)
            used_cells[pickup] = used_cells.get(pickup, 0) + 1

    def _assign_pickups(
        self,
        idle: List[Shipper],
        orders: Dict[int, Order],
        now: int,
        step_deadline: float,
        excluded_orders: Optional[Set[int]] = None,
    ) -> None:
        if not idle:
            return
        pickup_counts, drop_counts = self._visible_density(orders)
        self.replanned_tasks.clear()
        used_orders: Set[int] = set(excluded_orders or ())
        used_cells: Dict[Position, int] = {}
        pending = self._partition_idle_shippers(idle, orders, now, used_orders, used_cells)
        exact_cap = max(6, min(14, 6 + len(pending) // 3))
        candidates = self._collect_candidates(
            pending, orders, now, step_deadline, pickup_counts, drop_counts, used_orders, exact_cap
        )
        self._coordinate_assign(candidates, used_orders, used_cells, pickup_counts)
        self._assign_backlog_orders(
            pending, orders, now, step_deadline, used_orders, used_cells, pickup_counts, exact_cap
        )
        for shipper in pending:
            if shipper.id not in self.pickup_tasks:
                self.failed_assignment_retry[shipper.id] = now + 2

    # ------------------------------------------------------------------
    # Path planning helpers
    # ------------------------------------------------------------------

    def _reserved_path(
        self,
        start: Position,
        goal: Position,
        vertex_res: Dict[int, Set[Position]],
        edge_res: Set[Tuple[int, Position, Position]],
    ) -> List[Position]:
        path = [start]
        pos = start
        needs_repair = False
        for t in range(1, self.horizon + 1):
            current_dist = self.router.distance(pos, goal)
            options: List[Tuple[int, int, Move, Position]] = []
            for move, nxt in self.router.neighbors(pos, include_wait=True):
                if nxt in vertex_res.get(t, set()) or (t, nxt, pos) in edge_res:
                    continue
                dist = self.router.distance(nxt, goal)
                wait_cost = 1 if move == "S" and pos != goal else 0
                options.append((dist, wait_cost, move, nxt))
            if options:
                options.sort(key=lambda item: (item[0], item[1], item[2]))
                _, _, _, nxt = options[0]
                if current_dist < INF and self.router.distance(nxt, goal) >= current_dist and pos != goal:
                    self.repairs += 1
                    needs_repair = True
            else:
                nxt = pos
                self.repairs += 1
                needs_repair = True
            path.append(nxt)
            pos = nxt

        if needs_repair and self.repair_search_budget > 0:
            self.repair_search_budget -= 1
            repaired = self._space_time_repair(start, goal, vertex_res, edge_res)
            if repaired is not None and self._path_key(repaired, goal) < self._path_key(path, goal):
                path = repaired

        previous = path[0]
        for t, nxt in enumerate(path[1:], 1):
            vertex_res.setdefault(t, set()).add(nxt)
            edge_res.add((t, previous, nxt))
            previous = nxt
        return path

    def _path_key(self, path: List[Position], goal: Position) -> Tuple[int, int]:
        final_dist = self.router.distance(path[-1], goal)
        stalls = sum(
            1 for previous, current in zip(path, path[1:]) if previous == current and current != goal
        )
        return final_dist, stalls

    def _space_time_repair(
        self,
        start: Position,
        goal: Position,
        vertex_res: Dict[int, Set[Position]],
        edge_res: Set[Tuple[int, Position, Position]],
    ) -> Optional[List[Position]]:
        layer: Dict[Position, Tuple[int, Tuple[Position, ...]]] = {start: (0, (start,))}
        for t in range(1, self.horizon + 1):
            next_layer: Dict[Position, Tuple[int, Tuple[Position, ...]]] = {}
            for pos, (stalls, path) in layer.items():
                for move, nxt in self.router.neighbors(pos, include_wait=True):
                    if nxt in vertex_res.get(t, set()) or (t, nxt, pos) in edge_res:
                        continue
                    candidate = (
                        stalls + (1 if move == "S" and pos != goal else 0),
                        path + (nxt,),
                    )
                    previous = next_layer.get(nxt)
                    if previous is None or candidate < previous:
                        next_layer[nxt] = candidate
            if not next_layer:
                return None
            layer = next_layer
        _, best = min(
            layer.items(),
            key=lambda item: (self.router.distance(item[0], goal), item[1][0], item[0]),
        )
        return list(best[1])

    def _plan_paths(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Tuple[str, Order, Position]],
        priorities: Dict[int, float],
        slacks: Dict[int, float],
        staying: Set[int],
    ) -> Dict[int, List[Position]]:
        vertex_res: Dict[int, Set[Position]] = {}
        edge_res: Set[Tuple[int, Position, Position]] = set()
        by_id = {shipper.id: shipper for shipper in shippers}
        for sid in staying:
            pos = by_id[sid].position
            for t in range(1, self.horizon + 1):
                vertex_res.setdefault(t, set()).add(pos)
                edge_res.add((t, pos, pos))
        paths: Dict[int, List[Position]] = {}
        self.repair_search_budget = max(2, min(12, len(targets) // 2 + 1))
        for sid in sorted(targets, key=lambda item: (-priorities.get(item, 0.0), slacks.get(item, INF), item)):
            shipper = by_id[sid]
            path = self._reserved_path(shipper.position, targets[sid][2], vertex_res, edge_res)
            paths[sid] = path
        return paths

    # ------------------------------------------------------------------
    # Decision pipeline
    # ------------------------------------------------------------------

    def _expire_stale_followups(self, orders: Dict[int, Order]) -> None:
        for sid, oid in list(self.bundle_followups.items()):
            followup = orders.get(oid)
            if followup is None or followup.picked or followup.delivered:
                self.bundle_followups.pop(sid, None)

    def _process_loaded_shippers(
        self,
        shippers: List[Shipper],
        orders: Dict[int, Order],
        now: int,
        reserved_pickups: Set[int],
        bundle_deadline: float,
    ) -> Tuple[
        Dict[int, Action],
        Dict[int, Tuple[str, Order, Position]],
        Dict[int, float],
        Set[int],
        List[Shipper],
    ]:
        actions: Dict[int, Action] = {}
        targets: Dict[int, Tuple[str, Order, Position]] = {}
        priorities: Dict[int, float] = {}
        staying: Set[int] = set()
        idle: List[Shipper] = []
        for shipper in shippers:
            if self._deliverable(shipper, orders, shipper.position):
                actions[shipper.id] = ("S", 2)
                staying.add(shipper.id)
                self.pickup_tasks.pop(shipper.id, None)
                self.delivery_commitments.pop(shipper.id, None)
                self.bundle_followups.pop(shipper.id, None)
                continue
            delivery = self._delivery_target(shipper, orders, now)
            if delivery is not None:
                bundle = self._best_loaded_bundle_pickup(
                    shipper, delivery, orders, now, reserved_pickups, bundle_deadline,
                )
                if bundle is not None:
                    goal = (bundle.sx, bundle.sy)
                    targets[shipper.id] = ("pickup", bundle, goal)
                    priorities[shipper.id] = self._target_priority(shipper, "deliver", delivery, now) + 1.0
                    self.pickup_tasks[shipper.id] = bundle.id
                    self.delivery_commitments[shipper.id] = delivery.id
                    reserved_pickups.add(bundle.id)
                    continue
                goal = (delivery.ex, delivery.ey)
                targets[shipper.id] = ("deliver", delivery, goal)
                priorities[shipper.id] = self._target_priority(shipper, "deliver", delivery, now)
                self.pickup_tasks.pop(shipper.id, None)
                continue
            idle.append(shipper)
        return actions, targets, priorities, staying, idle

    def _handle_idle_pickup_targets(
        self,
        idle: List[Shipper],
        orders: Dict[int, Order],
        now: int,
        targets: Dict[int, Tuple[str, Order, Position]],
        priorities: Dict[int, float],
        staying: Set[int],
        reserved_pickups: Set[int],
    ) -> Dict[int, Action]:
        actions: Dict[int, Action] = {}
        for shipper in idle:
            oid = self.pickup_tasks.get(shipper.id)
            order = orders.get(oid) if oid is not None else None
            if order is None or order.picked or not self._can_take_light(shipper, order, orders):
                self.pickup_tasks.pop(shipper.id, None)
                actions[shipper.id] = ("S", 0)
                staying.add(shipper.id)
                continue
            pickup = (order.sx, order.sy)
            auxiliary = self._pickup_before_committed_target(shipper, order, orders, now)
            if auxiliary is not None:
                for other_sid, other_oid in list(self.pickup_tasks.items()):
                    if other_sid != shipper.id and other_oid == auxiliary.id:
                        self.pickup_tasks.pop(other_sid, None)
                actions[shipper.id] = ("S", 1)
                staying.add(shipper.id)
                reserved_pickups.add(auxiliary.id)
                self.bundle_followups[shipper.id] = order.id
                continue
            if shipper.position == pickup:
                actions[shipper.id] = ("S", 1)
                staying.add(shipper.id)
                continue
            targets[shipper.id] = ("pickup", order, pickup)
            priorities[shipper.id] = self._target_priority(shipper, "pickup", order, now)
        return actions

    def _compute_slacks(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Tuple[str, Order, Position]],
        now: int,
    ) -> Dict[int, float]:
        slacks: Dict[int, float] = {}
        for shipper in shippers:
            sid = shipper.id
            if sid not in targets:
                slacks[sid] = INF
                continue
            kind, order, goal = targets[sid]
            dist = self.router.distance(shipper.position, goal)
            if kind == "deliver":
                slacks[sid] = order.et - (now + dist)
            else:
                slacks[sid] = order.et - (now + dist + self.router.distance(goal, (order.ex, order.ey)))
        return slacks

    def _add_proactive_targets(
        self,
        idle: List[Shipper],
        orders: Dict[int, Order],
        targets: Dict[int, Tuple[str, Order, Position]],
        priorities: Dict[int, float],
        slacks: Dict[int, float],
        staying: Set[int],
    ) -> None:
        pickup_counts, _ = self._visible_density(orders)
        if not pickup_counts:
            return
        for shipper in idle:
            if shipper.id in staying or shipper.id in targets:
                continue
            best_cell = None
            best_v = -INF
            for cell in pickup_counts:
                dist = self.router.distance(shipper.position, cell)
                if dist >= INF or dist < 2:
                    continue
                v = self._cluster_value(cell, pickup_counts) / (dist + 6.0)
                if v > best_v:
                    best_v = v
                    best_cell = cell
            if best_cell and best_v > self._PROACTIVE_THRESHOLD:
                targets[shipper.id] = ("proactive", None, best_cell)
                priorities[shipper.id] = -1.0
                slacks[shipper.id] = INF

    def _build_path_actions(
        self,
        shippers: List[Shipper],
        targets: Dict[int, Tuple[str, Order, Position]],
        paths: Dict[int, List[Position]],
        orders: Dict[int, Order],
        now: int,
    ) -> Tuple[Dict[int, Action], Dict[int, Position]]:
        by_id = {s.id: s for s in shippers}
        actions: Dict[int, Action] = {}
        target_positions: Dict[int, Position] = {}
        claimed_pickups = {order.id for _, order, _ in targets.values() if order is not None}
        for sid, (kind, order, goal) in targets.items():
            shipper = by_id[sid]
            path = paths.get(sid, [shipper.position, shipper.position])
            nxt = path[1] if len(path) > 1 else shipper.position
            move = self.router.move_between(shipper.position, nxt)
            op = 0
            if nxt == goal and order is not None:
                op = 1 if kind == "pickup" else (2 if kind == "deliver" else 0)
            elif kind == "deliver" and order is not None and len(shipper.bag) < shipper.K_max:
                passing = self._pickup_at(shipper, orders, nxt)
                if (
                    passing is not None
                    and passing.id not in claimed_pickups
                    and self._passing_pickup_is_worthwhile(shipper, order, passing, orders, now)
                ):
                    op = 1
                    claimed_pickups.add(passing.id)
                    self.delivery_commitments[sid] = order.id
            actions[sid] = (move, op)
            target_positions[sid] = goal
        return actions, target_positions

    def _step_budget(self, now: int, assignable_count: int, cheap_only: bool) -> float:
        if cheap_only:
            return 0.02
        if now >= self.T - self._endgame_window():
            return min(0.90, 0.12 + 0.032 * max(1, assignable_count))
        return min(0.42, 0.04 + 0.014 * max(1, assignable_count))

    def _decide(self, obs: dict, cheap_only: bool = False) -> Dict[int, Action]:
        now = int(obs["t"])
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = list(obs["shippers"])

        self._expire_stale_followups(orders)

        reserved_pickups: Set[int] = set()
        bundle_budget = 0.008 if cheap_only else min(0.08, 0.010 + 0.004 * max(1, self.C))
        bundle_deadline = time.perf_counter() + bundle_budget

        actions, targets, priorities, staying, idle = self._process_loaded_shippers(
            shippers, orders, now, reserved_pickups, bundle_deadline
        )

        assignable = [
            s for s in idle
            if now >= self.failed_assignment_retry.get(s.id, -1) or s.id in self.pickup_tasks
        ]
        self._assign_pickups(
            assignable, orders, now,
            time.perf_counter() + self._step_budget(now, len(assignable), cheap_only),
            reserved_pickups,
        )

        actions.update(self._handle_idle_pickup_targets(
            idle, orders, now, targets, priorities, staying, reserved_pickups
        ))

        slacks = self._compute_slacks(shippers, targets, now)
        self._add_proactive_targets(idle, orders, targets, priorities, slacks, staying)

        paths = self._plan_paths(shippers, targets, priorities, slacks, staying)
        path_actions, target_positions = self._build_path_actions(shippers, targets, paths, orders, now)
        actions.update(path_actions)

        return resolve_collisions_and_blocks(
            shippers,
            actions,
            self.grid,
            orders,
            self._pickup_at,
            self._deliverable,
            target_positions,
            self.router.distance,
            slacks=slacks,
            allow_unblock=False,
        )

    def run(self) -> dict:
        start = time.time()
        obs = self.env.reset()
        self._configure(obs)
        self.run_deadline = start + max(180.0, min(900.0, 0.25 * self.T + 8.0 * self.C))
        while not obs.get("done", False):
            actions = self._decide(obs, cheap_only=time.time() > self.run_deadline)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break
        return self.env.result(self.method_name, time.time() - start)
