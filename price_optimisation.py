"""Whiteboard-style linear programme for coffee-capsule prices.

Decision variables are the Regular price R and the Premium price P. Each
household buys a fixed quantity of one product, so revenue is linear in the
two prices. Which product they buy is a straight line in the R-P plane, fitted
by OLS from the weekly data (a linear probability model). Product assignments
are separate linear programmes; the best feasible one is kept.

The earlier successive linear programme of quadratic OLS revenue has been
retired. This module is the only price model the app uses.

Class-board numbers are not copied from anywhere. Slopes, intercepts, and
quantities are estimated here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog

FORMULATION_SUMMARY = """
Decision variables are the Regular price R and the Premium price P, both non-negative.

Each household buys a fixed quantity of one product (the average units bought on weeks they chose that product). Revenue is therefore linear: quantity times the price of the product they buy.

The product is not a decision variable inside one programme. Each household is a straight line in the R-P plane, fitted by OLS (a linear probability model at the 0.5 contour). One side of the line is Premium, the other is Regular, and a household that sometimes buys nothing also has a stop-buying line. Each assignment of products to households is its own linear programme. The assignment with the highest feasible revenue is the shared menu.

`scipy.optimize.linprog` minimises, so the objective vector is the negated quantity vector. Upper price bounds are the highest Regular and Premium prices in the experiment, so a price cannot run off to infinity. Those bounds are data, not demand coefficients.
""".strip()

LINE_STYLE = {
    "Household 1": "solid",
    "Household 2": "dash",
    "Household 3": "longdashdot",
}
LINE_COLOR = {
    "Household 1": "#172033",
    "Household 2": "#1677ff",
    "Household 3": "#ff8a1f",
}
_TOL = 1e-7


def discover_households(frame: pd.DataFrame) -> list[str]:
    """Column prefixes that have both a Regular and a Premium quantity."""
    found = []
    for col in frame.columns:
        if not str(col).endswith("_Regular") or col == "P_Regular":
            continue
        prefix = str(col)[: -len("_Regular")]
        if prefix == "P":
            continue
        if f"{prefix}_Premium" in frame.columns:
            found.append(prefix)
    return found


def household_label(prefix: str) -> str:
    if prefix.startswith("HH") and prefix[2:].isdigit():
        return f"Household {int(prefix[2:])}"
    return prefix


def _fit_lpm(prices: np.ndarray, target: np.ndarray) -> dict:
    """OLS linear probability model. Returns intercept and slopes on R and P."""
    model = LinearRegression()
    model.fit(prices, target)
    fitted = model.predict(prices)
    resid = target - fitted
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / ss_tot) if ss_tot > 1e-15 else float("nan")
    return {
        "intercept": float(model.intercept_),
        "b_R": float(model.coef_[0]),
        "b_P": float(model.coef_[1]),
        "r_squared": r2,
    }


def _score(line: dict, regular: float, premium: float) -> float:
    return line["intercept"] + line["b_R"] * regular + line["b_P"] * premium


def line_equation(line: dict) -> str:
    """0.5 contour written as P = slope·R + intercept, or a vertical R line."""
    intercept, b_r, b_p = line["intercept"], line["b_R"], line["b_P"]
    if abs(b_p) < 1e-12:
        if abs(b_r) < 1e-12:
            return "score does not depend on price"
        level = (0.5 - intercept) / b_r
        return f"R = {level:.4f}"
    slope = -b_r / b_p
    icept = (0.5 - intercept) / b_p
    sign = "+" if icept >= 0 else "−"
    return f"P = {slope:.4f} R {sign} {abs(icept):.4f}"


def _halfspace(line: dict, side: str) -> tuple[np.ndarray, float]:
    """score >= 0.5 (`high`) or score <= 0.5 (`low`) as one row of A_ub x <= b."""
    intercept, b_r, b_p = line["intercept"], line["b_R"], line["b_P"]
    if side == "high":
        return np.array([-b_r, -b_p], dtype=float), intercept - 0.5
    if side == "low":
        return np.array([b_r, b_p], dtype=float), 0.5 - intercept
    raise ValueError(side)


def inequality_text(line: dict, side: str) -> str:
    """Human reading of the half-space, as a bound on P or on R."""
    intercept, b_r, b_p = line["intercept"], line["b_R"], line["b_P"]
    if abs(b_p) < 1e-12:
        level = (0.5 - intercept) / b_r if abs(b_r) > 1e-12 else float("nan")
        wants_low_r = (side == "high" and b_r < 0) or (side == "low" and b_r > 0)
        op = "<=" if wants_low_r else ">="
        return f"R {op} {level:.4f}"
    slope = -b_r / b_p
    icept = (0.5 - intercept) / b_p
    wants_below = (side == "high" and b_p < 0) or (side == "low" and b_p > 0)
    op = "<=" if wants_below else ">="
    sign = "+" if icept >= 0 else "−"
    return f"P {op} {slope:.4f} R {sign} {abs(icept):.4f}"


def load_problem(csv_path: str = "coffee_capsules_data.csv") -> dict:
    """Fit one line per household and the fixed quantities, plus price bounds."""
    frame = pd.read_csv(csv_path)
    regular = frame["P_Regular"].to_numpy(dtype=float)
    premium = frame["P_Premium"].to_numpy(dtype=float)
    prices = np.column_stack([regular, premium])
    r_sd = float(np.std(regular, ddof=1))
    p_sd = float(np.std(premium, ddof=1))
    bounds = {
        "R_lower": 0.0,
        "P_lower": 0.0,
        "R_upper": float(np.max(regular)),
        "P_upper": float(np.max(premium)),
        "R_sd": r_sd,
        "P_sd": p_sd,
    }
    households = []
    for prefix in discover_households(frame):
        label = household_label(prefix)
        reg_qty = frame[f"{prefix}_Regular"].to_numpy(dtype=float)
        prem_qty = frame[f"{prefix}_Premium"].to_numpy(dtype=float)
        bought_reg = reg_qty > 0
        bought_prem = prem_qty > 0
        bought = bought_reg | bought_prem
        d_regular = float(reg_qty[bought_reg].mean()) if bought_reg.any() else 0.0
        d_premium = float(prem_qty[bought_prem].mean()) if bought_prem.any() else 0.0
        lines = []
        if bought.any() and (~bought).any():
            stop = _fit_lpm(prices, bought.astype(float))
            stop["kind"] = "stop"
            stop["equation"] = line_equation(stop)
            lines.append(stop)
        buyers = bought
        if buyers.sum() >= 2 and bought_reg[buyers].any() and bought_prem[buyers].any():
            switch_target = bought_prem[buyers].astype(float)
            switch = _fit_lpm(prices[buyers], switch_target)
            switch["kind"] = "switch"
            switch["equation"] = line_equation(switch)
            lines.append(switch)
        predicted = [
            predict_product({"lines": lines, "only_product": "Premium" if not bought_reg.any() else None}, r, p)
            for r, p in zip(regular, premium)
        ]
        observed = np.where(bought_reg, "Regular", np.where(bought_prem, "Premium", "None"))
        accuracy = float(np.mean([pred == obs for pred, obs in zip(predicted, observed)])) if len(observed) else float("nan")
        households.append(
            {
                "label": label,
                "prefix": prefix,
                "lines": lines,
                "d_regular": d_regular,
                "d_premium": d_premium,
                "n_regular": int(bought_reg.sum()),
                "n_premium": int(bought_prem.sum()),
                "n_none": int((~bought).sum()),
                "only_product": None if bought_reg.any() else "Premium",
                "accuracy": accuracy,
            }
        )
    return {"households": households, "bounds": bounds, "frame": frame}


def product_choices(household: dict) -> list[str]:
    """Products this household can be assigned, given the lines the data support."""
    kinds = {line["kind"] for line in household["lines"]}
    choices = []
    if "switch" in kinds or household["d_regular"] > 0:
        choices.append("Regular")
    if household["d_premium"] > 0 or "switch" in kinds:
        choices.append("Premium")
    if "stop" in kinds:
        choices.append("None")
    if household["only_product"] == "Premium" and "Regular" in choices and "switch" not in kinds:
        choices = [c for c in choices if c != "Regular"]
    return choices


def _sides_for(household: dict, product: str) -> list[tuple[dict, str]]:
    """Behavioural half-spaces that keep `product` on the correct side of each line."""
    kinds = {line["kind"]: line for line in household["lines"]}
    sides = []
    if product == "None":
        if "stop" not in kinds:
            raise ValueError(f"{household['label']} has no stop-buying line")
        sides.append((kinds["stop"], "low"))
        return sides
    if "stop" in kinds:
        sides.append((kinds["stop"], "high"))
    if "switch" in kinds:
        sides.append((kinds["switch"], "high" if product == "Premium" else "low"))
    elif product == "Regular" and household["only_product"] == "Premium":
        raise ValueError(f"{household['label']} never buys Regular")
    return sides


def price_box_rows(bounds: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rows = np.array(
        [
            [-1.0, 0.0],
            [1.0, 0.0],
            [0.0, -1.0],
            [0.0, 1.0],
        ]
    )
    rhs = np.array(
        [
            -bounds["R_lower"],
            bounds["R_upper"],
            -bounds["P_lower"],
            bounds["P_upper"],
        ],
        dtype=float,
    )
    names = [
        f"R >= {bounds['R_lower']:.4f}",
        f"R <= {bounds['R_upper']:.4f}",
        f"P >= {bounds['P_lower']:.4f}",
        f"P <= {bounds['P_upper']:.4f}",
    ]
    return rows, rhs, names


def build_constraint_system(households: list[dict], assignment: dict[str, str], bounds: dict) -> dict:
    """Stack behavioural half-spaces and the price box into A_ub x <= b_ub."""
    rows = []
    rhs = []
    names = []
    for household in households:
        product = assignment[household["label"]]
        for line, side in _sides_for(household, product):
            coeff, limit = _halfspace(line, side)
            rows.append(coeff)
            rhs.append(limit)
            kind = "switches so that" if line["kind"] == "switch" else "keeps buying so that"
            if product == "None":
                kind = "stops buying so that"
            names.append(f"{household['label']} {kind} {inequality_text(line, side)}")
    box_rows, box_rhs, box_names = price_box_rows(bounds)
    if rows:
        matrix = np.vstack([np.vstack(rows), box_rows])
        limits = np.concatenate([np.array(rhs, dtype=float), box_rhs])
    else:
        matrix = box_rows
        limits = box_rhs
    return {"A_ub": matrix, "b_ub": limits, "names": names + box_names}


def objective_coefficients(households: list[dict], assignment: dict[str, str]) -> np.ndarray:
    """c such that revenue = c · [R, P]. None contributes nothing."""
    c_r = 0.0
    c_p = 0.0
    for household in households:
        product = assignment[household["label"]]
        if product == "Regular":
            c_r += household["d_regular"]
        elif product == "Premium":
            c_p += household["d_premium"]
    return np.array([c_r, c_p], dtype=float)


def _vertices_from_constraints(matrix: np.ndarray, limits: np.ndarray) -> list[np.ndarray]:
    points = []
    n = len(limits)
    for i in range(n):
        for j in range(i + 1, n):
            pair = np.vstack([matrix[i], matrix[j]])
            if abs(np.linalg.det(pair)) < 1e-10:
                continue
            try:
                point = np.linalg.solve(pair, np.array([limits[i], limits[j]], dtype=float))
            except np.linalg.LinAlgError:
                continue
            if np.all(matrix @ point <= limits + 1e-6):
                points.append(point)
    unique = []
    for point in points:
        if not any(np.linalg.norm(point - kept) < 1e-6 for kept in unique):
            unique.append(point)
    if not unique:
        return []
    centroid = np.mean(unique, axis=0)
    unique.sort(key=lambda pt: np.arctan2(pt[1] - centroid[1], pt[0] - centroid[0]))
    return unique


def _tight_constraints(matrix: np.ndarray, limits: np.ndarray, point: np.ndarray) -> list[int]:
    slack = limits - matrix @ point
    return [i for i, gap in enumerate(slack) if abs(gap) <= 1e-5]


def _is_vertex(matrix: np.ndarray, limits: np.ndarray, point: np.ndarray) -> bool:
    tight = _tight_constraints(matrix, limits, point)
    for i in tight:
        for j in tight:
            if j <= i:
                continue
            if abs(np.linalg.det(np.vstack([matrix[i], matrix[j]]))) > 1e-8:
                return True
    return False


def solve_assignment(households: list[dict], assignment: dict[str, str], bounds: dict) -> dict:
    """One linear programme: this product assignment, maximised with linprog."""
    system = build_constraint_system(households, assignment, bounds)
    coeff = objective_coefficients(households, assignment)
    # linprog minimises. Negating the quantity vector maximises revenue.
    result = linprog(
        c=-coeff,
        A_ub=system["A_ub"],
        b_ub=system["b_ub"],
        bounds=[(bounds["R_lower"], bounds["R_upper"]), (bounds["P_lower"], bounds["P_upper"])],
        method="highs",
    )
    label = ", ".join(f"{name} buys {product}" for name, product in assignment.items())
    solved = {
        "assignment": dict(assignment),
        "assignment_label": label,
        "c": coeff,
        "A_ub": system["A_ub"],
        "b_ub": system["b_ub"],
        "constraint_names": system["names"],
        "status": int(result.status),
        "status_name": result.message,
        "feasible": bool(result.success),
    }
    if not result.success:
        solved.update(
            {
                "R": float("nan"),
                "P": float("nan"),
                "revenue": float("-inf"),
                "vertex": False,
                "tight": [],
                "satisfied": False,
            }
        )
        return solved
    point = np.asarray(result.x, dtype=float)
    revenue = float(coeff @ point)
    tight_idx = _tight_constraints(system["A_ub"], system["b_ub"], point)
    solved.update(
        {
            "R": float(point[0]),
            "P": float(point[1]),
            "revenue": revenue,
            "vertex": _is_vertex(system["A_ub"], system["b_ub"], point),
            "tight": [system["names"][i] for i in tight_idx],
            "satisfied": bool(np.all(system["A_ub"] @ point <= system["b_ub"] + 1e-6)),
            "vertices": _vertices_from_constraints(system["A_ub"], system["b_ub"]),
            "flat_price": (
                "Premium price is not in this objective, so revenue is unchanged along that edge; the solver returns a vertex."
                if abs(coeff[1]) < 1e-12
                else "Regular price is not in this objective, so revenue is unchanged along that edge; the solver returns a vertex."
                if abs(coeff[0]) < 1e-12
                else ""
            ),
        }
    )
    return solved


def enumerate_assignments(households: list[dict]) -> list[dict[str, str]]:
    """Cartesian product of each household's feasible products."""
    menus: list[dict[str, str]] = [{}]
    for household in households:
        extended = []
        for product in product_choices(household):
            for menu in menus:
                nxt = dict(menu)
                nxt[household["label"]] = product
                extended.append(nxt)
        menus = extended
    return menus


def best_programme(households: list[dict], bounds: dict, assignments: list[dict[str, str]] | None = None) -> dict:
    """Solve every assignment and keep the highest feasible revenue."""
    if assignments is None:
        assignments = enumerate_assignments(households)
    solved = [solve_assignment(households, menu, bounds) for menu in assignments]
    feasible = [item for item in solved if item["feasible"] and np.isfinite(item["revenue"])]
    if not feasible:
        return {"best": None, "cases": solved}
    best = max(feasible, key=lambda item: item["revenue"])
    return {"best": best, "cases": solved}


def predict_product(household: dict, regular: float, premium: float) -> str:
    """Product implied by which side of the fitted lines the price sits on."""
    kinds = {line["kind"]: line for line in household["lines"]}
    if "stop" in kinds and _score(kinds["stop"], regular, premium) < 0.5 - 1e-9:
        return "None"
    if "switch" in kinds:
        return "Premium" if _score(kinds["switch"], regular, premium) >= 0.5 - 1e-9 else "Regular"
    return household.get("only_product") or "Premium"


def revenue_of(household: dict, regular: float, premium: float) -> dict:
    product = predict_product(household, regular, premium)
    if product == "Regular":
        revenue = household["d_regular"] * regular
    elif product == "Premium":
        revenue = household["d_premium"] * premium
    else:
        revenue = 0.0
    return {"product": product, "revenue": float(revenue)}


def _format_objective(coeff: np.ndarray) -> str:
    parts = []
    if abs(coeff[0]) > 1e-12:
        parts.append(f"{coeff[0]:.4f} R")
    if abs(coeff[1]) > 1e-12:
        parts.append(f"{coeff[1]:.4f} P")
    if not parts:
        return "max  0"
    return "max  " + " + ".join(parts)


def format_programme(solved: dict, households: list[dict] | None = None) -> str:
    """Objective and every constraint, with the fitted numbers filled in."""
    lines = [
        "Decision variables: R (Regular price), P (Premium price).",
        _format_objective(solved["c"]),
        "subject to",
    ]
    for name in solved["constraint_names"]:
        lines.append(f"  {name}")
    if households is not None:
        lines.append("assignment")
        for label, product in solved["assignment"].items():
            match = next(item for item in households if item["label"] == label)
            quantity = {"Regular": match["d_regular"], "Premium": match["d_premium"], "None": 0.0}[product]
            lines.append(f"  {label} buys {product} (d = {quantity:.4f})")
    if solved["feasible"]:
        lines.append(
            f"optimum  R = {solved['R']:.4f}, P = {solved['P']:.4f}, revenue = {solved['revenue']:.4f}"
        )
        lines.append("binding constraints")
        for name in solved["tight"]:
            lines.append(f"  {name}")
    else:
        lines.append("infeasible")
    return "\n".join(lines)


def household_equations(household: dict) -> list[str]:
    """One readable equation per fitted line, plus the fixed quantities."""
    text = []
    for line in household["lines"]:
        if line["kind"] == "switch":
            role = "switches between Regular and Premium on"
        else:
            role = "stops buying above"
        text.append(f"{role} {line['equation']}")
    text.append(
        f"d_regular = {household['d_regular']:.4f} (n = {household['n_regular']}), "
        f"d_premium = {household['d_premium']:.4f} (n = {household['n_premium']})"
    )
    return text


def optimise_prices_detailed(csv_path: str = "coffee_capsules_data.csv") -> dict:
    """Shared menu plus one programme per household, and every assignment tried."""
    problem = load_problem(csv_path)
    households = problem["households"]
    bounds = problem["bounds"]
    shared = best_programme(households, bounds)
    per_household = []
    for household in households:
        solo = best_programme([household], bounds)
        per_household.append({"household": household, "programme": solo})
    return {
        "households": households,
        "bounds": bounds,
        "frame": problem["frame"],
        "shared": shared,
        "per_household": per_household,
        "formulation": format_programme(shared["best"], households) if shared["best"] else "No feasible shared programme.",
    }


def _shock_households(households: list[dict], label: str, field: str, factor: float) -> list[dict]:
    shocked = []
    for household in households:
        clone = {
            **household,
            "lines": [dict(line) for line in household["lines"]],
        }
        if clone["label"] == label:
            for line in clone["lines"]:
                line[field] = line[field] * factor
                line["equation"] = line_equation(line)
        shocked.append(clone)
    return shocked


def structural_sensitivity(detail: dict, pct: float = 0.10) -> pd.DataFrame:
    """Re-solve after shocking each fitted line coefficient, and after moving prices."""
    rows = []
    base = detail["shared"]["best"]
    bounds = detail["bounds"]
    fields = (("intercept", "intercept"), ("b_R", "slope on R"), ("b_P", "slope on P"))
    for household in detail["households"]:
        for field, pretty in fields:
            for sign, factor in (("+", 1.0 + pct), ("−", 1.0 - pct)):
                shocked = _shock_households(detail["households"], household["label"], field, factor)
                solved = best_programme(shocked, bounds)
                best = solved["best"]
                revenue = best["revenue"] if best else float("nan")
                rows.append(
                    {
                        "Household": household["label"],
                        "kind": "line coefficient",
                        "parameter": pretty,
                        "shock": f"{sign}{pct:.0%}",
                        "R_opt": best["R"] if best else float("nan"),
                        "P_opt": best["P"] if best else float("nan"),
                        "max_revenue": revenue,
                        "delta_revenue": revenue - base["revenue"] if best else float("nan"),
                        "assignment": best["assignment_label"] if best else "infeasible",
                        "status": "optimal" if best else "infeasible",
                    }
                )
        own = next(item["programme"]["best"] for item in detail["per_household"] if item["household"]["label"] == household["label"])
        for price_name, index in (("R", 0), ("P", 1)):
            for sign, factor in (("+", 1.0 + pct), ("−", 1.0 - pct)):
                point = [own["R"], own["P"]]
                point[index] = point[index] * factor
                assessed = revenue_of(household, point[0], point[1])
                rows.append(
                    {
                        "Household": household["label"],
                        "kind": "price move",
                        "parameter": price_name,
                        "shock": f"{sign}{pct:.0%}",
                        "R_opt": point[0],
                        "P_opt": point[1],
                        "max_revenue": assessed["revenue"],
                        "delta_revenue": assessed["revenue"] - own["revenue"],
                        "assignment": assessed["product"],
                        "status": "evaluated",
                    }
                )
    return pd.DataFrame(rows)


def bound_sensitivity(detail: dict) -> pd.DataFrame:
    """Re-solve the shared programme with a looser and a tighter price box."""
    rows = []
    base_bounds = detail["bounds"]
    variants = [
        ("Observed maximum", 0.0),
        ("Observed maximum + 1 SD", 1.0),
        ("Observed maximum + 2 SD", 2.0),
    ]
    for name, extra in variants:
        bounds = dict(base_bounds)
        bounds["R_upper"] = base_bounds["R_upper"] + extra * base_bounds["R_sd"]
        bounds["P_upper"] = base_bounds["P_upper"] + extra * base_bounds["P_sd"]
        solved = best_programme(detail["households"], bounds)
        best = solved["best"]
        rows.append(
            {
                "Price box": name,
                "R_upper": bounds["R_upper"],
                "P_upper": bounds["P_upper"],
                "R_opt": best["R"] if best else float("nan"),
                "P_opt": best["P"] if best else float("nan"),
                "max_revenue": best["revenue"] if best else float("nan"),
                "assignment": best["assignment_label"] if best else "infeasible",
            }
        )
    return pd.DataFrame(rows)


def regret_table(detail: dict, scenario: tuple[float, float]) -> pd.DataFrame:
    """Revenue and regret of the shared optimum, each household optimum, and the sliders."""
    shared = detail["shared"]["best"]
    menus = [("Shared LP optimum", shared["R"], shared["P"])]
    for item in detail["per_household"]:
        best = item["programme"]["best"]
        menus.append((f"{item['household']['label']} LP optimum", best["R"], best["P"]))
    menus.append(("Scenario sliders", float(scenario[0]), float(scenario[1])))
    rows = []
    for item in detail["per_household"]:
        household = item["household"]
        own = item["programme"]["best"]["revenue"]
        for menu, regular, premium in menus:
            assessed = revenue_of(household, regular, premium)
            rows.append(
                {
                    "Household": household["label"],
                    "Menu": menu,
                    "R": regular,
                    "P": premium,
                    "product": assessed["product"],
                    "revenue": assessed["revenue"],
                    "regret": own - assessed["revenue"],
                }
            )
    return pd.DataFrame(rows)


def _add_household_lines(fig: go.Figure, households: list[dict], bounds: dict, span: tuple[float, float]) -> None:
    r_grid = np.linspace(span[0], span[1], 80)
    for household in households:
        for line in household["lines"]:
            if abs(line["b_P"]) < 1e-12:
                continue
            slope = -line["b_R"] / line["b_P"]
            icept = (0.5 - line["intercept"]) / line["b_P"]
            p_grid = slope * r_grid + icept
            fig.add_trace(
                go.Scatter(
                    x=r_grid,
                    y=p_grid,
                    mode="lines",
                    name=f"{household['label']}: {line['equation']}",
                    line=dict(
                        color=LINE_COLOR.get(household["label"], "#172033"),
                        dash=LINE_STYLE.get(household["label"], "solid"),
                        width=2.5,
                        shape="spline",
                    ),
                    hovertemplate="R=%{x:.2f}<br>P=%{y:.2f}<extra>" + household["label"] + "</extra>",
                )
            )


def _shade(fig: go.Figure, solved: dict) -> None:
    vertices = solved.get("vertices") or []
    if len(vertices) < 3:
        return
    xs = [pt[0] for pt in vertices] + [vertices[0][0]]
    ys = [pt[1] for pt in vertices] + [vertices[0][1]]
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            fill="toself",
            mode="lines",
            name="Feasible region",
            line=dict(color="rgba(22,119,255,0.35)", width=1),
            fillcolor="rgba(22,119,255,0.16)",
            hoverinfo="skip",
        )
    )


def _objective_level(fig: go.Figure, solved: dict, bounds: dict) -> None:
    coeff = solved["c"]
    if not solved["feasible"]:
        return
    revenue = solved["revenue"]
    r0, r1 = bounds["R_lower"], bounds["R_upper"]
    if abs(coeff[1]) > 1e-10:
        p0 = (revenue - coeff[0] * r0) / coeff[1]
        p1 = (revenue - coeff[0] * r1) / coeff[1]
        fig.add_trace(
            go.Scatter(
                x=[r0, r1],
                y=[p0, p1],
                mode="lines",
                name="Objective level",
                line=dict(color="#0f9f6e", width=2, dash="dot"),
                hovertemplate="Iso-revenue<extra></extra>",
            )
        )
    elif abs(coeff[0]) > 1e-10:
        fig.add_vline(x=solved["R"], line=dict(color="#0f9f6e", width=2, dash="dot"))


def _optimum_marker(fig: go.Figure, solved: dict, bounds: dict) -> None:
    if not solved["feasible"]:
        return
    fig.add_trace(
        go.Scatter(
            x=[solved["R"]],
            y=[solved["P"]],
            mode="markers",
            name="Optimum",
            marker=dict(size=14, color="#172033", symbol="star", line=dict(color="white", width=1)),
            hovertemplate=f"R=%{{x:.2f}}<br>P=%{{y:.2f}}<br>Revenue={solved['revenue']:.2f}<extra></extra>",
        )
    )
    # Keep the label inside the axes when the star sits on the upper-right corner.
    ax = -120 if solved["R"] > 0.55 * bounds["R_upper"] else 80
    ay = 55 if solved["P"] > 0.72 * bounds["P_upper"] else -50
    fig.add_annotation(
        x=solved["R"],
        y=solved["P"],
        text=f"Optimal: P={solved['P']:.2f}, R={solved['R']:.2f}, Revenue={solved['revenue']:.2f}",
        showarrow=True,
        arrowhead=2,
        ax=ax,
        ay=ay,
        bgcolor="white",
        bordercolor="#172033",
        borderwidth=1,
        font=dict(size=12, color="#172033"),
    )


def _axis_layout(fig: go.Figure, title: str, bounds: dict) -> None:
    pad_r = max(4.0, 0.08 * bounds["R_upper"])
    pad_p = max(6.0, 0.08 * bounds["P_upper"])
    fig.update_layout(
        title=title,
        xaxis_title="Regular price R",
        yaxis_title="Premium price P",
        xaxis=dict(range=[bounds["R_lower"] - pad_r, bounds["R_upper"] + pad_r], zeroline=False),
        yaxis=dict(range=[bounds["P_lower"] - pad_p, bounds["P_upper"] + pad_p], zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=48, r=24, t=80, b=48),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=560,
    )
    fig.add_shape(
        type="rect",
        x0=bounds["R_lower"],
        x1=bounds["R_upper"],
        y0=bounds["P_lower"],
        y1=bounds["P_upper"],
        line=dict(color="rgba(23,32,51,0.25)", width=1, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )


def shared_figure(detail: dict) -> go.Figure:
    """All three household lines, the shared feasible region, and one optimum."""
    bounds = detail["bounds"]
    solved = detail["shared"]["best"]
    fig = go.Figure()
    _shade(fig, solved)
    span = (bounds["R_lower"] - 2, bounds["R_upper"] + 2)
    _add_household_lines(fig, detail["households"], bounds, span)
    _objective_level(fig, solved, bounds)
    _optimum_marker(fig, solved, bounds)
    _axis_layout(fig, "Shared prices", bounds)
    return fig


def household_figure(detail: dict, label: str) -> go.Figure:
    """One household: its own line, its own feasible region, its own optimum."""
    bounds = detail["bounds"]
    item = next(entry for entry in detail["per_household"] if entry["household"]["label"] == label)
    solved = item["programme"]["best"]
    fig = go.Figure()
    _shade(fig, solved)
    span = (bounds["R_lower"] - 2, bounds["R_upper"] + 2)
    _add_household_lines(fig, [item["household"]], bounds, span)
    _objective_level(fig, solved, bounds)
    _optimum_marker(fig, solved, bounds)
    _axis_layout(fig, label, bounds)
    fig.update_layout(height=480)
    return fig


def structural_sensitivity_figure(table: pd.DataFrame) -> go.Figure:
    coef = table[table["kind"] == "line coefficient"]
    fig = go.Figure()
    for household, color in LINE_COLOR.items():
        part = coef[coef["Household"] == household]
        if part.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[f"{row.parameter} {row.shock}" for row in part.itertuples()],
                y=part["delta_revenue"],
                name=household,
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="group",
        title="Shared revenue change when a fitted line coefficient moves ±10%",
        yaxis_title="Change in shared revenue (€)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=48, r=16, t=60, b=80),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def price_move_figure(table: pd.DataFrame) -> go.Figure:
    moves = table[table["kind"] == "price move"]
    fig = go.Figure()
    for household, color in LINE_COLOR.items():
        part = moves[moves["Household"] == household]
        fig.add_trace(
            go.Bar(
                x=[f"{row.parameter} {row.shock}" for row in part.itertuples()],
                y=part["delta_revenue"],
                name=household,
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="group",
        title="Revenue change when that household's own optimum price moves ±10%",
        yaxis_title="Change versus own optimum (€)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=48, r=16, t=60, b=48),
        legend=dict(orientation="h", y=1.12),
    )
    return fig


def bound_sensitivity_figure(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=table["Price box"],
            y=table["max_revenue"],
            marker_color="#1677ff",
            text=[f"R={r:.1f}, P={p:.1f}" for r, p in zip(table["R_opt"], table["P_opt"])],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Shared revenue as the upper price box widens",
        yaxis_title="Shared revenue (€)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=420,
        margin=dict(l=48, r=16, t=60, b=48),
    )
    return fig


def regret_figure(table: pd.DataFrame) -> go.Figure:
    pivot = table.pivot(index="Household", columns="Menu", values="regret")
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.to_numpy(),
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale="Blues",
            colorbar=dict(title="Regret €"),
            text=np.round(pivot.to_numpy(), 2),
            texttemplate="%{text:.2f}",
            hovertemplate="%{y}<br>%{x}<br>regret €%{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Regret versus each household's own LP optimum",
        height=360,
        margin=dict(l=80, r=16, t=60, b=80),
        paper_bgcolor="white",
    )
    return fig


def sanity_check(detail: dict) -> list[str]:
    """Optimum on a vertex, constraints satisfied, revenue equals c·x."""
    problems = []
    shared = detail["shared"]["best"]
    if shared is None:
        return ["shared programme infeasible"]
    point = np.array([shared["R"], shared["P"]])
    if not shared["satisfied"]:
        problems.append("shared optimum violates A_ub x <= b_ub")
    if not shared["vertex"]:
        problems.append("shared optimum is not a vertex")
    if abs(float(shared["c"] @ point) - shared["revenue"]) > 1e-6:
        problems.append("shared revenue is not c·x")
    bounds = detail["bounds"]
    if not (bounds["R_lower"] - 1e-8 <= shared["R"] <= bounds["R_upper"] + 1e-8):
        problems.append("shared R outside bounds")
    if not (bounds["P_lower"] - 1e-8 <= shared["P"] <= bounds["P_upper"] + 1e-8):
        problems.append("shared P outside bounds")
    best_revenue = shared["revenue"]
    for case in detail["shared"]["cases"]:
        if case["feasible"] and case["revenue"] > best_revenue + 1e-6:
            problems.append(f"a better assignment was discarded: {case['assignment_label']}")
    for item in detail["per_household"]:
        solo = item["programme"]["best"]
        if solo is None or not solo["vertex"] or not solo["satisfied"]:
            problems.append(f"{item['household']['label']} optimum failed the vertex check")
    return problems


def _print_report(detail: dict) -> None:
    print("Per-household lines and fixed quantities")
    for household in detail["households"]:
        print(household["label"])
        for line in household_equations(household):
            print(" ", line)
        print(f"  in-sample line accuracy {household['accuracy']:.3f}")
    print()
    print("Shared linear programme")
    print(detail["formulation"])
    print()
    print("Assignments evaluated")
    for case in sorted(detail["shared"]["cases"], key=lambda item: item["revenue"], reverse=True):
        flag = "feasible" if case["feasible"] else "infeasible"
        revenue = f"{case['revenue']:.4f}" if case["feasible"] else "—"
        print(f"  {case['assignment_label']}: {flag}, revenue {revenue}")
    print()
    print("Per-household optima")
    for item in detail["per_household"]:
        best = item["programme"]["best"]
        print(
            f"  {item['household']['label']}: {best['assignment_label']}, "
            f"R={best['R']:.4f}, P={best['P']:.4f}, revenue={best['revenue']:.4f}, vertex={best['vertex']}"
        )
    problems = sanity_check(detail)
    print()
    print("Sanity:", "ok" if not problems else "; ".join(problems))


if __name__ == "__main__":
    _print_report(optimise_prices_detailed())
