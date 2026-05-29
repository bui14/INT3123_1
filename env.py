"""
env.py — Online MAPD Graph Environment
=========================================

Môi trường online cho bài toán Multi-Agent Package Delivery.
- Chỉ G (tổng số đơn) được cố định từ đầu.
- Đơn hàng được sinh/reveal theo thời gian theo quá trình Poisson.
- Solver chỉ nhìn thấy observation hiện tại.

Cấu trúc module:
  Constants          — hằng số toàn cục
  Data classes       — Order, Shipper (state only)
  Grid helpers       — is_valid_cell, next_pos, valid_next_pos, manhattan
  Reward helpers     — r_base, delivery_reward, move_cost
  Action helpers     — parse_action, parse_actions, is_delivery_op
  Simulation helpers — _apply_moves, _order_rate, _init_shippers, _start_positions
  Config I/O         — load_config, parse_grid
  DeliveryEnv        — stateful simulator
"""

from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
TIME_UNIT_PER_HOUR = 10
TIME_UNIT_PER_DAY  = 240

ALPHA = {1: 1.0, 2: 2.0, 3: 3.0}
BETA  = {1: 0.1, 2: 0.3, 3: 0.5}
GAMMA = 1.0

HOTSPOT_RADIUS = 3
HOTSPOT_PROB   = 0.7

DIRS = {"S": (0, 0), "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Order:
    """Đơn hàng g_i = <sx, sy, ex, ey, et, w, p>. Tọa độ 0-index."""
    id: int; sx: int; sy: int; ex: int; ey: int
    et: int; w: float; p: int; appear_t: int
    picked: bool = False; delivered: bool = False
    carrier: int = -1; deliver_t: int = -1


@dataclass
class Shipper:
    """Trạng thái shipper. Logic kiểm tra/di chuyển dùng module-level functions."""
    id: int; r: int; c: int; W_max: float; K_max: int
    bag: List[int] = field(default_factory=list)
    total_reward: float = 0.0; steps_moved: int = 0

    @property
    def position(self) -> Tuple[int, int]:
        return self.r, self.c

    def move_to(self, pos: Tuple[int, int], orders: Dict[int, Order]) -> float:
        """Di chuyển đến pos; trả chi phí (<=0). Bỏ qua nếu pos == vị trí hiện tại."""
        if pos == self.position:
            return 0.0
        w_carried = sum(orders[oid].w for oid in self.bag if oid in orders)
        cost = move_cost(w_carried, self.W_max)
        self.r, self.c = pos
        self.total_reward += cost
        self.steps_moved += 1
        return cost

    def can_carry(self, order: Order, orders: Dict[int, Order]) -> bool:
        """True nếu còn đủ slot và tải trọng để nhận thêm order."""
        if order.picked or order.delivered:
            return False
        w_carried = sum(orders[oid].w for oid in self.bag if oid in orders)
        return len(self.bag) < self.K_max and w_carried + order.w <= self.W_max

    def can_pickup(self, order: Order, orders: Dict[int, Order]) -> bool:
        """True nếu shipper đứng đúng điểm lấy và can_carry(order)."""
        return (order.sx, order.sy) == self.position and self.can_carry(order, orders)

    def pickup_best(self, orders: Dict[int, Order]) -> Optional[int]:
        """
        Nhặt đúng một đơn tốt nhất tại ô hiện tại.

        Thứ tự ưu tiên:
          1. ưu tiên cao hơn trước;
          2. deadline sớm hơn trước;
          3. id nhỏ hơn trước.
        """
        candidates = [o for o in orders.values() if self.can_pickup(o, orders)]
        if not candidates:
            return None

        order = min(candidates, key=lambda o: (-o.p, o.et, o.id))
        order.picked = True
        order.carrier = self.id
        self.bag.append(order.id)
        return order.id


    def can_deliver(self, order: Order) -> bool:
        """True nếu shipper đang mang order và đứng đúng điểm giao."""
        return order.id in self.bag and not order.delivered and (order.ex, order.ey) == self.position

    def deliver(self, order: Order, t: int, T: int) -> float:
        """Giao order tại bước t; trả phần thưởng (0 nếu can_deliver thất bại)."""
        if not self.can_deliver(order):
            return 0.0
        reward = delivery_reward(order, t, T)
        order.delivered = True; order.deliver_t = t; order.carrier = self.id
        self.bag.remove(order.id)
        self.total_reward += reward
        return reward


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def is_valid_cell(pos: Tuple[int, int], grid: List[List[int]]) -> bool:
    """True nếu pos trong bản đồ và không phải ô vật cản (grid[r][c] == 0)."""
    r, c = pos
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0


def next_pos(pos: Tuple[int, int], move: str) -> Tuple[int, int]:
    """Tọa độ kế tiếp theo hướng move, không kiểm tra hợp lệ."""
    dr, dc = DIRS.get(move, (0, 0))
    return pos[0] + dr, pos[1] + dc


def valid_next_pos(pos: Tuple[int, int], move: str, grid: List[List[int]]) -> Tuple[int, int]:
    """Tọa độ kế tiếp sau move; giữ nguyên pos nếu ô đích bị chặn hoặc ra ngoài."""
    nxt = next_pos(pos, move)
    return nxt if is_valid_cell(nxt, grid) else pos


def manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
    """Khoảng cách Manhattan giữa hai ô lưới."""
    return abs(r1 - r2) + abs(c1 - c2)


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def r_base(w: float) -> float:
    """Phần thưởng cơ bản theo khối lượng đơn hàng."""
    if w <= 0.2:  return 4.0
    if w <= 3.0:  return 10.0
    if w <= 10.0: return 15.0
    if w <= 30.0: return 20.0
    return 30.0


def delivery_reward(order: Order, t_delivery: int, T: int) -> float:
    """Tính phần thưởng giao hàng theo công thức đề bài (có/không có penalty trễ hạn)."""
    rb = r_base(order.w)
    if t_delivery <= order.et:
        bonus = max(0.0, (order.et - t_delivery) / max(order.et, 1))
        return ALPHA[order.p] * rb * (1.0 + bonus)
    factor = max(0.0, 1.0 - (t_delivery - order.et) / max(T, 1))
    return BETA[order.p] * rb * factor


def move_cost(w_carried: float, w_max: float) -> float:
    """Chi phí di chuyển một bước theo tải trọng hiện tại."""
    return -0.01 * (1.0 + GAMMA * w_carried / max(w_max, 1.0))


# ---------------------------------------------------------------------------
# Action helpers
# ---------------------------------------------------------------------------

def parse_action(action: Any) -> Tuple[str, Any]:
    """
    Chuẩn hóa một action thô thành (move, op).

    Định dạng đầu vào được chấp nhận:
      None                      -> ("S", 0)
      "L" / "U" / ...           -> (move, 0)
      ("R", 1)                  -> ("R", 1)
      ("D", (2, 3))     -> ("D", (2, 3))
    """
    if action is None:
        return "S", 0
    if isinstance(action, str):
        return (action, 0) if action in DIRS else ("S", 0)
    if isinstance(action, (tuple, list)) and action:
        move = action[0] if action[0] in DIRS else "S"
        op   = action[1] if len(action) >= 2 else 0
        return move, op
    return "S", 0


def parse_actions(actions: Any, n_shippers: int) -> Dict[int, Any]:
    """
    Chuẩn hóa tập actions (dict / list / None) thành Dict[shipper_id -> action_raw].

    Solver có thể truyền:
      - dict  {0: action0, 1: action1, ...}
      - list  [action0, action1, ...]  — ánh xạ theo thứ tự index
      - None  -> tất cả shipper đứng yên
    """
    if actions is None:
        return {}
    if isinstance(actions, dict):
        return actions
    if isinstance(actions, (list, tuple)):
        return {i: actions[i] for i in range(min(len(actions), n_shippers))}
    return {}


def is_delivery_op(op: Any) -> bool:
    """True nếu cargo_op yêu cầu giao hàng.
    Với env chuẩn, op=2 nghĩa là giao tất cả đơn trong bag có đích tại ô hiện tại sau khi di chuyển."""
    return op == 2


# ---------------------------------------------------------------------------
# Simulation helpers (private — chỉ dùng nội bộ trong file này)
# ---------------------------------------------------------------------------

def _apply_moves(
    shippers: List[Shipper],
    moves: Dict[int, str],
    grid: List[List[int]],
    orders: Dict[int, Order],
) -> float:
    """Di chuyển tất cả shipper theo moves, xử lý va chạm (id nhỏ được ưu tiên giữ ô).
    Trả về tổng chi phí di chuyển của bước này (<=0)."""
    reward = 0.0
    old_positions = {s.id: s.position for s in shippers}
    occupied = set(old_positions.values())
    desired  = {s.id: valid_next_pos(s.position, moves.get(s.id, "S"), grid) for s in shippers}

    for shipper in sorted(shippers, key=lambda s: s.id):
        old    = old_positions[shipper.id]
        target = desired[shipper.id]
        occupied.discard(old)
        if target in occupied:
            target = old
        occupied.add(target)
        reward += shipper.move_to(target, orders)
    return reward


def _order_rate(
    t: int,
    lambda0: float,
    surge_windows: List[Tuple[int, int]],
    surge_amplitude: float,
    G: int,
    T: int,
) -> float:
    """Tốc độ sinh đơn lambda(t): lambda0*(1+A) trong surge window, lambda0 ngoài."""
    lam = float(lambda0 if lambda0 is not None else G / max(T, 1))
    amp = float(surge_amplitude)
    in_surge = any(ts <= t <= te for ts, te in surge_windows)
    return lam * (1.0 + amp) if in_surge else lam


def _start_positions(N: int, C: int, free_cells: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Chọn C vị trí xuất phát trải đều: ưu tiên 5 neo góc/tâm, sau đó max-min distance."""
    selected: List[Tuple[int, int]] = []
    anchors = [(0, 0), (0, N-1), (N-1, 0), (N-1, N-1), (N//2, N//2)]

    for anchor in anchors:
        cell = min(free_cells, key=lambda x: manhattan(x[0], x[1], anchor[0], anchor[1]))
        if cell not in selected:
            selected.append(cell)
        if len(selected) == C:
            return selected

    while len(selected) < C:
        cell = max(free_cells, key=lambda x: min(manhattan(x[0], x[1], y[0], y[1]) for y in selected))
        if cell in selected:
            cell = next(c for c in free_cells if c not in selected)
        selected.append(cell)
    return selected


def _init_shippers(
    N: int,
    C: int,
    W_max: List[float],
    K_max: List[int],
    free_cells: List[Tuple[int, int]],
) -> List[Shipper]:
    """Khởi tạo list Shipper từ tham số riêng lẻ; vị trí từ _start_positions()."""
    positions = _start_positions(N, C, free_cells)
    return [
        Shipper(i, r, c, float(W_max[i]), int(K_max[i]))
        for i, (r, c) in enumerate(positions)
    ]


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_int_pairs(value: str, key: str) -> List[Tuple[int, int]]:
    nums = list(map(int, value.split())) if value.strip() else []
    if len(nums) % 2 != 0:
        raise ValueError(f"{key} phải là danh sách cặp số nguyên.")
    return [(nums[i], nums[i+1]) for i in range(0, len(nums), 2)]


def _normalize_shipper_list(cfg: dict, key: str, typ):
    values = cfg.get(key)
    if values is None:
        raise ValueError(f"Thiếu {key} trong config {cfg.get('name', '')}.")
    if len(values) == cfg["C"]:
        return values
    if len(values) == 1:
        return [typ(values[0])] * cfg["C"]
    raise ValueError(f"{key} phải có đúng C={cfg['C']} phần tử hoặc 1 phần tử để broadcast.")


def parse_grid(lines: List[str], idx: int, N: int) -> Tuple[List[List[int]], int]:
    """Đọc N dòng bản đồ từ lines[idx:idx+N]; trả (grid, idx_sau)."""
    grid: List[List[int]] = []
    for row_i in range(N):
        if idx + row_i >= len(lines):
            raise ValueError(f"Thiếu dòng bản đồ thứ {row_i+1}/{N}.")
        row = list(map(int, lines[idx + row_i].split()))
        if len(row) != N:
            raise ValueError(f"Dòng {idx+row_i+1}: cần {N} cột, thực tế {len(row)}.")
        grid.append(row)
    return grid, idx + N


def load_config(filepath: str) -> List[dict]:
    """Đọc file config theo cấu trúc [SEED] ... [CONFIG] ... [MAP] ... [END]; trả list các cfg dict."""
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.rstrip() for line in f]

    configs: List[dict] = []
    base_seed = 42
    i = 0

    if i < len(lines) and _strip_comment(lines[i]) == "[SEED]":
        i += 1
        line = _strip_comment(lines[i])
        key, val = [x.strip() for x in line.split("=", 1)]
        if key != "base_seed":
            raise ValueError("Block [SEED] chỉ hỗ trợ dòng 'base_seed = ...'.")
        base_seed = int(val)
        i += 1

    while i < len(lines):
        if _strip_comment(lines[i]) != "[CONFIG]":
            i += 1
            continue

        cfg: dict = {}
        i += 1
        while i < len(lines) and _strip_comment(lines[i]) != "[MAP]":
            line = _strip_comment(lines[i])
            if line and "=" in line:
                key, val = [x.strip() for x in line.split("=", 1)]
                if   key == "K_max":                        cfg[key] = list(map(int,   val.split()))
                elif key == "W_max":                        cfg[key] = list(map(float, val.split()))
                elif key in {"N", "C", "G", "T"}:          cfg[key] = int(val)
                elif key in {"surge_windows", "hotspots"}:  cfg[key] = _parse_int_pairs(val, key)
                elif key in {"surge_amplitude", "lambda0"}: cfg[key] = float(val)
                else:                                        cfg[key] = val
            i += 1

        for key in ["name", "N", "C", "G", "T", "K_max", "W_max"]:
            if key not in cfg:
                raise ValueError(f"Thiếu '{key}' trong một [CONFIG].")

        cfg["K_max"] = _normalize_shipper_list(cfg, "K_max", int)
        cfg["W_max"] = _normalize_shipper_list(cfg, "W_max", float)

        cfg["base_seed"] = base_seed

        if i >= len(lines) or _strip_comment(lines[i]) != "[MAP]":
            raise ValueError(f"Config '{cfg.get('name')}' thiếu [MAP].")
        i += 1

        cfg["grid"], i = parse_grid(lines, i, cfg["N"])

        if i < len(lines) and _strip_comment(lines[i]) == "[END]":
            i += 1

        configs.append(cfg)

    return configs


# ---------------------------------------------------------------------------
# Internal env utilities (không public vì phụ thuộc random state)
# ---------------------------------------------------------------------------

def _free_cells(grid: List[List[int]]) -> List[Tuple[int, int]]:
    return [(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 0]


def _stable_seed(name: str, base_seed: int = SEED) -> int:
    digest = hashlib.md5(f"{base_seed}:{name}".encode()).hexdigest()
    return int(digest[:8], 16)


def _resolve_generation_params(cfg: dict, base_seed: int = SEED) -> dict:
    """Điền tham số surge/hotspot ẩn nếu config Phase 1 không công bố."""
    params = dict(cfg)
    params.setdefault("lambda0", cfg["G"] / max(cfg["T"], 1))

    if params.get("surge_windows") and params.get("hotspots"):
        params.setdefault("surge_amplitude", 3.0)
        return params

    cells = _free_cells(cfg["grid"])
    if not cells:
        params.update({"surge_windows": [], "hotspots": [], "surge_amplitude": 0.0})
        return params

    rng        = random.Random(_stable_seed(str(cfg.get("name", "unknown")), base_seed))
    n_windows  = 1 if cfg["N"] <= 10 else 2
    n_hotspots = min(max(1, cfg["C"] // 2), 3)
    amp        = 2.0 if str(cfg.get("name", "")) == "C1" else (2.5 if cfg["N"] <= 12 else 3.0)
    low        = max(1, int(0.15 * cfg["T"]))
    high       = max(low + 1, int(0.75 * cfg["T"]))
    duration   = max(20, min(cfg["T"] // 5, TIME_UNIT_PER_DAY // 2))
    starts     = [rng.randint(low, max(low, high - duration)) for _ in range(n_windows)]

    params["surge_windows"] = sorted((s, min(cfg["T"]-1, s+duration)) for s in starts)
    params["hotspots"]      = rng.sample(cells, min(n_hotspots, len(cells)))
    params.setdefault("surge_amplitude", amp)
    return params


def _binomial_draw(n: int, p: float, rng: random.Random) -> int:
    if n <= 0 or p <= 0: return 0
    if p >= 1:           return n
    return sum(1 for _ in range(n) if rng.random() < p)


# ---------------------------------------------------------------------------
# DeliveryEnv
# ---------------------------------------------------------------------------

class DeliveryEnv:
    """
    Stateful online simulator — vòng lặp chính cho solver và grader.

    Public API:
        reset()                     -> obs
        step(actions)               -> (obs, reward, done, info)
        observe()                   -> obs
        info()                      -> dict
        result(method, elapsed_sec) -> dict

    Private (đột biến state nội bộ, không thể là pure function):
        __reveal_orders()    — sinh đơn mới vào self.orders
        __new_order_count()  — tính số đơn cần sinh tại bước hiện tại
        __sample_order()     — tạo một Order ngẫu nhiên, tăng next_order_id
        __deliver()          — giao hàng + cập nhật delivered/on_time/late
    """

    def __init__(self, cfg: dict, seed: int = SEED, rng: Optional[random.Random] = None):
        config_data = copy.deepcopy(cfg)
        self.config_name = config_data.get("name", "unknown")
        self.grid = copy.deepcopy(config_data["grid"])
        self.N, self.C, self.G, self.T = config_data["N"], config_data["C"], config_data["G"], config_data["T"]
        self.__rng_seed = _stable_seed(str(self.config_name), seed)
        self.__W_max = [float(x) for x in config_data["W_max"]]
        self.__K_max = [int(x) for x in config_data["K_max"]]

        resolved = _resolve_generation_params(copy.deepcopy(config_data), seed)
        self.__lambda0 = float(resolved["lambda0"])
        self.__surge_windows = resolved.get("surge_windows", [])
        self.__hotspots = resolved.get("hotspots", [])
        self.__surge_amplitude = float(resolved.get("surge_amplitude", 3.0))

        initial_rng = rng if rng is not None else random.Random(self.__rng_seed)
        self.__initial_rng_state = initial_rng.getstate()

        self.free_cells = _free_cells(self.grid)
        if not self.free_cells:
            raise ValueError("Bản đồ không có ô trống.")
        self.reset()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> dict:
        """Reset về trạng thái ban đầu; replay cùng episode (seed cố định)."""
        self.__rng = random.Random()
        self.__rng.setstate(self.__initial_rng_state)
        self.t = 0
        self.next_order_id   = 0
        self.generated_count = 0
        self.orders: Dict[int, Order] = {}
        self.new_orders_last_step: List[int] = []
        self.shippers = _init_shippers(self.N, self.C, self.__W_max, self.__K_max, self.free_cells)
        self.total_reward    = 0.0
        self.total_movecost  = 0.0
        self.delivered = self.on_time = self.late = 0
        self.__reveal_orders()
        return self.observe()

    def observe(self) -> dict:
        """Trả snapshot trạng thái hiện tại cho solver (deep-copy nhẹ)."""
        return {
            "t": self.t, "N": self.N, "C": self.C, "G": self.G, "T": self.T,
            "grid": self.grid,
            "orders": {
                oid: Order(o.id, o.sx, o.sy, o.ex, o.ey, o.et, o.w, o.p, o.appear_t,
                           o.picked, o.delivered, o.carrier, o.deliver_t)
                for oid, o in self.orders.items() if not o.delivered
            },
            "new_order_ids": list(self.new_orders_last_step),
            "shippers": [
                Shipper(s.id, s.r, s.c, s.W_max, s.K_max, list(s.bag), s.total_reward, s.steps_moved)
                for s in self.shippers
            ],
            "done": self.t >= self.T,
        }

    def step(self, actions: Any) -> Tuple[dict, float, bool, dict]:
        """
        Thực thi một bước mô phỏng.

        Thứ tự: Di chuyển -> Nhặt hàng -> Giao hàng -> Sinh đơn mới.
        Trả (obs, reward_buoc, done, info).
        """
        if self.t >= self.T:
            return self.observe(), 0.0, True, self.info()

        parsed = {sid: parse_action(a) for sid, a in parse_actions(actions, self.C).items()}

        move_reward = _apply_moves(
            self.shippers,
            {sid: move for sid, (move, _) in parsed.items()},
            self.grid,
            self.orders,
        )
        self.total_movecost += move_reward
        step_reward = move_reward

        for shipper in sorted(self.shippers, key=lambda s: s.id):
            _, op = parsed.get(shipper.id, ("S", 0))
            if op == 1:
                shipper.pickup_best(self.orders)
                continue
            if is_delivery_op(op):
                step_reward += self.__deliver_many(shipper)

        self.total_reward += step_reward
        self.t += 1
        if self.t < self.T:
            self.__reveal_orders()
        return self.observe(), step_reward, self.t >= self.T, self.info()

    def info(self) -> dict:
        """Thống kê tích lũy đến bước hiện tại."""
        return {
            "generated":      self.generated_count,
            "total_orders":   self.G,
            "delivered":      self.delivered,
            "on_time":        self.on_time,
            "late":           self.late,
            "missed":         self.G - self.delivered,
            "total_reward":   self.total_reward,
            "total_movecost": self.total_movecost,
            "net_reward":     self.total_reward,
        }

    def result(self, method: str, elapsed_sec: float = 0.0) -> dict:
        """Kết quả cuối episode cho grader/report."""
        return {
            "method":           method,
            "config_name":      self.config_name,
            "total_orders":     self.G,
            "orders_generated": self.generated_count,
            "delivered":        self.delivered,
            "on_time":          self.on_time,
            "late":             self.late,
            "missed":           self.G - self.delivered,
            "delivery_rate":    100.0 * self.delivered / max(self.G, 1),
            "on_time_rate":     100.0 * self.on_time / max(self.delivered, 1),
            "total_reward":     round(self.total_reward - self.total_movecost, 4),
            "total_movecost":   round(self.total_movecost, 4),
            "net_reward":       round(self.total_reward, 4),
            "elapsed_sec":      round(elapsed_sec, 4),
            "shipper_rewards":  [round(s.total_reward, 4) for s in self.shippers],
            "status":           "OK",
        }

    # ------------------------------------------------------------------
    # Private — đột biến state nội bộ env (rng, orders, counters)
    # ------------------------------------------------------------------

    def __reveal_orders(self) -> None:
        """Sinh đơn mới vào self.orders theo số lượng tính bởi __new_order_count."""
        self.new_orders_last_step = []
        for _ in range(self.__new_order_count()):
            o = self.__sample_order()
            self.orders[o.id] = o
            self.generated_count += 1
            self.new_orders_last_step.append(o.id)

    def __new_order_count(self) -> int:
        """Số đơn cần sinh tại bước t, đảm bảo tổng đúng G đơn khi kết thúc."""
        remaining = self.G - self.generated_count
        if remaining <= 0:       return 0
        if self.T - self.t <= 1: return remaining
        now = _order_rate(self.t, self.__lambda0, self.__surge_windows, self.__surge_amplitude, self.G, self.T)
        future = sum(
            _order_rate(tt, self.__lambda0, self.__surge_windows, self.__surge_amplitude, self.G, self.T)
            for tt in range(self.t, self.T)
        )
        return _binomial_draw(remaining, now / max(future, 1e-12), self.__rng)

    def __sample_order(self) -> Order:
        """Sinh một Order ngẫu nhiên (có thể tập trung vào hotspot nếu đang surge)."""
        src      = self.__rng.choice(self.free_cells)
        hotspots = self.__hotspots
        in_surge = any(ts <= self.t <= te for ts, te in self.__surge_windows)

        if in_surge and hotspots and self.__rng.random() < HOTSPOT_PROB:
            center = self.__rng.choice(hotspots)
            nearby = [c for c in self.free_cells if manhattan(c[0], c[1], center[0], center[1]) <= HOTSPOT_RADIUS]
            src = self.__rng.choice(nearby or self.free_cells)

        destinations = [c for c in self.free_cells if c != src]
        dst      = self.__rng.choice(destinations or self.free_cells)
        priority = self.__rng.choices([1, 2, 3], weights=[0.5, 0.3, 0.2])[0]
        weight   = self.__rng.choices([0.1, 1.0, 5.0, 15.0, 40.0], weights=[0.2, 0.4, 0.25, 0.1, 0.05])[0]
        deadline = min(self.t + self.__rng.randint(1, 6) * (4 - priority) * TIME_UNIT_PER_HOUR, self.T - 1)

        oid = self.next_order_id
        self.next_order_id += 1
        return Order(oid, src[0], src[1], dst[0], dst[1], deadline, weight, priority, self.t)

    def __deliver_many(self, shipper: Shipper) -> float:
        """Giao tất cả đơn hợp lệ tại vị trí hiện tại trong cùng timestep.

        cargo_op = 2 không chỉ định id đơn. Vì vậy env duyệt toàn bộ bag của
        shipper và để Shipper.deliver() kiểm tra điều kiện cuối cùng:
        đơn phải đang được shipper mang và destination phải đúng vị trí hiện tại.
        """
        total = 0.0
        for oid in list(shipper.bag):
            total += self.__deliver(shipper, oid)
        return total

    def __deliver(self, shipper: Shipper, oid: int) -> float:
        """Giao đơn oid bởi shipper; cập nhật self.delivered / on_time / late."""
        order = self.orders.get(oid)
        if order is None:
            return 0.0
        was_on_time = self.t <= order.et
        reward = shipper.deliver(order, self.t, self.T)
        if reward <= 0.0:
            return 0.0
        self.delivered += 1
        if was_on_time: self.on_time += 1
        else:           self.late += 1
        return reward
