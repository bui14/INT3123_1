"""
Grader dùng DeliveryEnv dạng stateful simulator:
- Mỗi solver được chạy trên một env mới có cùng seed/config.
- Env không sinh trước toàn bộ đơn hàng; đơn chỉ được sinh/reveal tại thời điểm t.
- Chỉ G là tổng số đơn cố định từ đầu.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time
from typing import Any

from env import DeliveryEnv, SEED, load_config

MAX_TOTAL_SECONDS = 3600

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_SOLVER_DIR = os.path.join(SCRIPT_DIR, "solvers")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
if BASE_SOLVER_DIR not in sys.path:
    sys.path.insert(0, BASE_SOLVER_DIR)

SOLVER_SOURCES = [
    ("GreedyBFS", "greedy_bfs.py"),
    ("VRPOrToolsSolver", "vrp_ortools.py"),
    ("ACOSolver", "aco_solver.py"),
    ("MAPDCBSSolver", "mapd_cbs_solver.py"),
]


def load_solver_class(class_name: str, file_name: str):
    path = os.path.join(BASE_SOLVER_DIR, file_name)
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Không tìm thấy {file_name} trong thư mục solvers.")
    spec = importlib.util.spec_from_file_location(class_name, path)
    if spec is None or spec.loader is None:
        sys.exit(f"[ERROR] Không thể load module {file_name}.")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    solver_cls = getattr(mod, class_name, None)
    if solver_cls is None:
        sys.exit(f"[ERROR] Không tìm thấy lớp {class_name} trong {file_name}.")
    return solver_cls


def load_solver_classes():
    return [(name, load_solver_class(name, file_name)) for name, file_name in SOLVER_SOURCES]


def score_result(result: dict) -> float:
    return float(result.get("net_reward", 0.0))


def _error_result(method: str, cfg: dict, error: str) -> dict:
    total_orders = int(cfg.get("G", 0))
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
        "status": "ERROR",
        "error": error,
    }


def _stable_config_seed(config_name: str, base_seed: int) -> int:
    """Tạo seed riêng cho từng config, ổn định và không phụ thuộc thứ tự chạy solver."""
    digest = hashlib.md5(f"{base_seed}:{config_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _run_solver(solver_cls: Any, cfg: dict, seed: int) -> dict:
    # Mỗi solver nhận một bản sao cfg và một env mới.
    # Nhờ vậy state/active_orders/orders_generated của solver trước không thể leak sang solver sau.
    env_cfg = copy.deepcopy(cfg)
    env = DeliveryEnv(env_cfg, seed=seed)
    solver = solver_cls(env)
    return solver.run()


def main():
    parser = argparse.ArgumentParser(description="Online MAPD graph/RL grader")
    parser.add_argument("--config", required=True, help="Đường dẫn file test_config.txt")
    parser.add_argument("--out", default="results", help="Thư mục lưu kết quả")
    parser.add_argument("--method", default="all", help="Phương pháp chạy: 'all' để chạy tất cả, hoặc tên phương pháp cụ thể (GreedyBFS, VRPOrToolsSolver, ACOSolver, MAPDCBSSolver)")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Đang load solver modules ...")
    all_solver_classes = load_solver_classes()
    print("Load thành công.")
    
    # Chọn phương pháp
    if args.method == "all":
        solver_classes = all_solver_classes
        print("Chạy tất cả phương pháp:", ", ".join(name for name, _ in solver_classes))
    else:
        # Tìm phương pháp cụ thể
        solver_classes = [(name, cls) for name, cls in all_solver_classes if name == args.method]
        if not solver_classes:
            available = [name for name, _ in all_solver_classes]
            sys.exit(f"[ERROR] Phương pháp '{args.method}' không tồn tại. Các phương pháp có sẵn: {', '.join(available)}")
        print(f"Chạy phương pháp: {args.method}")
    
    print("Solver sẽ chạy:", ", ".join(name for name, _ in solver_classes), "\n")

    print(f"Đọc config: {args.config}")
    configs = load_config(args.config)
    print(f"Tìm thấy {len(configs)} config.\n")

    all_results = []
    results_by_config = []
    total_start = time.time()

    for cfg in configs:
        name = cfg.get("name", "unknown")
        remaining = MAX_TOTAL_SECONDS - (time.time() - total_start)
        if remaining <= 0:
            print(f"[TIMEOUT] Đã vượt quá {MAX_TOTAL_SECONDS // 60} phút. Dừng lại.")
            break

        print(f"[{name}] N={cfg['N']} C={cfg['C']} G={cfg['G']} T={cfg['T']}  (còn {remaining / 60:.1f} phút)")
        print("  Chế độ online: chỉ G được biết trước; từng đơn được sinh/reveal trong step t.")
        config_seed = _stable_config_seed(str(name), cfg['base_seed'])

        cfg_results = []
        for solver_name, solver_cls in solver_classes:
            solver_start = time.time()
            try:
                result = _run_solver(solver_cls, cfg, config_seed)
            except Exception as e:
                result = _error_result(solver_name, cfg, str(e))

            wall = time.time() - solver_start
            result["wall_sec"] = round(wall, 2)
            result.setdefault("config_name", name)
            result.setdefault("method", solver_name)
            result.setdefault("total_orders", cfg["G"])
            result.setdefault("orders_generated", cfg["G"])
            result.setdefault("delivered", 0)
            result.setdefault("on_time", 0)
            result.setdefault("late", 0)
            result.setdefault("missed", result["total_orders"] - result["delivered"])
            result.setdefault("delivery_rate", 0.0)
            result.setdefault("on_time_rate", 0.0)
            result.setdefault("net_reward", 0.0)
            result.setdefault("total_reward", 0.0)
            result.setdefault("total_movecost", 0.0)
            result.setdefault("shipper_rewards", [])

            print(f"  [{result['method']}] Net reward: {result['net_reward']:.2f}")
            print(
                f"    Giao/Tổng: {result['delivered']}/{result['total_orders']}  "
                f"đúng hạn={result['on_time']}  trễ={result['late']}  bỏ lỡ={result['missed']}  "
                f"generated={result.get('orders_generated', 0)}  t={wall:.2f}s"
            )

            cfg_results.append(result)
            all_results.append(result)

        print("")
        config_payload = {
            "config_name": name,
            "orders_total_fixed": cfg["G"],
            "online_generation": True,
            "results": cfg_results,
        }
        results_by_config.append(config_payload)
        with open(os.path.join(args.out, f"result_{name}.json"), "w", encoding="utf-8") as f:
            json.dump(config_payload, f, ensure_ascii=False, indent=2)

    total_elapsed = time.time() - total_start
    methods = sorted({r.get("method", "unknown") for r in all_results})
    total_score_by_method = {
        method: round(sum(score_result(r) for r in all_results if r.get("method") == method), 4)
        for method in methods
    }

    print("=" * 100)
    print(f"{'Config':<10} {'Method':<28} {'Net Reward':>12} {'%Giao':>8} {'%Đúng hạn':>10} {'t(s)':>7}")
    print("-" * 100)
    for r in all_results:
        print(
            f"{r['config_name']:<10} {r['method']:<28} {r['net_reward']:>12.2f} "
            f"{r['delivery_rate']:>7.1f}% {r['on_time_rate']:>9.1f}% {r.get('wall_sec', 0):>7.1f}"
        )
    print("=" * 100)
    print("TỔNG ĐIỂM THEO PHƯƠNG PHÁP:")
    for method, score in total_score_by_method.items():
        print(f"- {method}: {score:.2f}")
    print(f"Tổng thời gian chạy: {total_elapsed:.1f}s / {MAX_TOTAL_SECONDS}s")

    summary = {
        "config_file": args.config,
        "seed": cfg['base_seed'],
        "online_generation": True,
        "total_elapsed": round(total_elapsed, 2),
        "total_score_by_method": total_score_by_method,
        "results_by_config": results_by_config,
        "all_results": all_results,
    }
    summary_path = os.path.join(args.out, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    all_results_path = os.path.join(args.out, "all_results.json")
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nĐã lưu tổng kết vào {summary_path}")
    print(f"Đã lưu toàn bộ kết quả vào {all_results_path}")


if __name__ == "__main__":
    main()
