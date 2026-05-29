from __future__ import annotations

from typing import Optional

from env import DeliveryEnv, Order


class Solver:
    """
    Base class cho solver.
    """

    def __init__(self, env: DeliveryEnv):
        if not isinstance(env, DeliveryEnv):
            raise TypeError("Solver chỉ hỗ trợ khởi tạo dạng Solver(env: DeliveryEnv).")

        self.env: DeliveryEnv = env
        self.cfg = env.public_cfg if hasattr(env, "public_cfg") else env.cfg
        self.grid = env.grid
        self.orders: list[Order] = []

    def run(self) -> dict:
        raise NotImplementedError


def default_result(method: str, cfg: dict, orders: Optional[list[Order]] = None) -> dict:
    """Kết quả mặc định cho các solver skeleton chưa cài đặt."""
    total_orders = int(cfg.get("G", len(orders) if orders is not None else 0))
    return {
        "method": method,
        "config_name": cfg.get("name", "unknown"),
        "total_orders": total_orders,
        "orders_generated": 0,
        "delivered": 0,
        "on_time": 0,
        "late": 0,
        "missed": total_orders,
        "delivery_rate": 0.0,
        "on_time_rate": 0.0,
        "total_reward": 0.0,
        "total_movecost": 0.0,
        "net_reward": 0.0,
        "elapsed_sec": 0.0,
        "shipper_rewards": [],
        "status": "TODO",
    }
