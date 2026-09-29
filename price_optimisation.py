"""Per-household linear programmes for revenue-maximising capsule prices.

The question the owner asks is which Premium price ``P`` and Regular price
``R`` maximise profit. Capsule costs are not in the dataset, and quantities
are not choice variables: each household's quantities are the predictions of
its own OLS demand equations. The linear programme therefore maximises
*revenue* and the only decision variables are the two prices. That coincides
with profit maximisation only when marginal cost is zero or already sunk.

Revenue with linear demand is a quadratic function of ``(R, P)``. A linear
programme cannot take that quadratic as its objective directly, so each solve
is a successive linear programme: at a feasible reference price the quadratic
is replaced by its first-order Taylor expansion, ``scipy.optimize.linprog``
maximises that linear function (HiGHS minimises the negated gradient) inside
the demand-and-price polytope cut by a trust box around the reference, and the
step is accepted only when the true quadratic revenue does not fall. The trust
box is what stops a single linear objective from jumping to a corner of the
price region. One linear programme over the whole polytope would be optimal
at a vertex, and the revenue maximum for a household with a concave hill is
interior.

The programme is solved separately for every household. Households do not
share demand coefficients, so they do not share the half-planes on which
predicted demand hits zero (the household stops buying that capsule). An
optional common-price solve is reported alongside the separate optima for
comparison; it is not a substitute for the per-household programmes.

Constraints exist to keep the solution out of two unusable regions: prices
running to infinity, and prices at which the linear demand prediction is
negative. They are not a cost function.

Run ``python price_optimisation.py`` to fit the OLS equations on
``coffee_capsules_data.csv``, solve every household, and compare each LP
solution with the exact maximum of the quadratic on the same polygon.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "coffee_capsules_data.csv"

# Upper price guardrail = observed maximum + this many sample standard
# deviations. It is a bound on the decision variables, not a demand
# coefficient. One standard deviation sits just outside the experiment so a
# household whose revenue hill is near the edge of the sample (Household 1)
# is not truncated, while a household whose quadratic is unbounded in a price
# (Households 2 and 3) cannot send that price to infinity.
DEFAULT_UPPER_SD_MULTIPLIER = 1.0
UPPER_SD_MULTIPLIERS = (0.0, 1.0, 2.0)

TRUST_INIT_FRACTION = 0.25
TRUST_GROW = 1.5
TRUST_SHRINK = 0.5
TRUST_MIN = 1e-7
STATIONARY_STEP = 1e-6
MAX_ITER = 80
FEASIBILITY_TOL = 1e-7
# Directional derivative of revenue toward the best feasible vertex, in euros.
# Below this, the first-order linear model has nothing left to gain.
STATIONARITY_TOL = 1e-3

COEF_NAMES = ("a_R", "b_RR", "b_RP", "a_P", "b_PR", "b_PP")

FORMULATION_SUMMARY = """
The owner's question is the optimal Premium price P and Regular price R to maximise profit. Costs are not observed, so the linear programme maximises revenue by changing only those two prices. Predicted quantities come from household-specific OLS demand and are not decision variables. Revenue with linear demand is quadratic, so the solver uses a successive first-order Taylor linearisation (a trust-region linear programme at each step). scipy.optimize.linprog minimises, therefore the objective vector is the negated revenue gradient. Constraints are rebuilt for every household from that household's OLS coefficients: predicted Regular quantity >= 0, predicted Premium quantity >= 0, and price bounds R >= 0, P >= 0 plus a finite upper guardrail taken from the observed prices. Those constraints stop prices going to infinity or into a region where the household's predicted demand turns negative and it stops buying. Each household is its own programme because the coefficients, and therefore the demand boundaries, differ.
""".strip()


def resolve_data_path(path=None) -> Path:
    if path is None:
        return DEFAULT_DATA
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    alongside = ROOT / candidate
    if alongside.is_file():
        return alongside
    return candidate


def household_label(prefix: str) -> str:
    if prefix.startswith("HH") and prefix[2:].isdigit():
        return f"Household {int(prefix[2:])}"
    return prefix


def discover_households(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Return ``(label, regular_quantity_column, premium_quantity_column)``.

    ``P_Regular`` and ``P_Premium`` are the prices faced by every household,
    not a household's purchased quantities. They share the ``_Regular`` /
    ``_Premium`` suffix, so they are excluded by name.
    """
    pairs = []
    for column in frame.columns:
        if column == "P_Regular" or not str(column).endswith("_Regular"):
            continue
        prefix = column[: -len("_Regular")]
        premium = f"{prefix}_Premium"
        if premium == "P_Premium" or premium not in frame.columns:
            continue
        pairs.append((household_label(prefix), column, premium))
    if not pairs:
        raise ValueError(
            "Expected household quantity columns named {id}_Regular and {id}_Premium."
        )
    return pairs


def load_prices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    missing = [name for name in ("P_Regular", "P_Premium") if name not in frame.columns]
    if missing:
        raise ValueError(f"Price column(s) missing from the dataset: {missing}")
    regular = frame["P_Regular"].to_numpy(dtype=float)
    premium = frame["P_Premium"].to_numpy(dtype=float)
    return regular, premium


def fit_ols(quantity, regular_price, premium_price) -> dict:
    """OLS of one capsule's quantity on an intercept, R, and P.

    The slope on R and the slope on P are whichever price is in that
    equation. Mapping them into ``(a_R, b_RR, b_RP, a_P, b_PR, b_PP)`` is
    done by the caller so the cross-price names stay unambiguous.
    """
    y = np.asarray(quantity, dtype=float)
    regular = np.asarray(regular_price, dtype=float)
    premium = np.asarray(premium_price, dtype=float)
    design = np.column_stack([np.ones(len(y)), regular, premium])
    beta, _residuals, rank, _singular = np.linalg.lstsq(design, y, rcond=None)
    # An identically zero series has a zero coefficient vector. Forcing it
    # avoids a numerical dust row that would make "quantity >= 0" look like
    # a real price restriction.
    if np.allclose(y, 0.0):
        beta = np.zeros(3, dtype=float)
    fitted = design @ beta
    total = float(np.sum((y - y.mean()) ** 2))
    residual = float(np.sum((y - fitted) ** 2))
    r_squared = np.nan if total < 1e-12 else 1.0 - residual / total
    return {
        "a": float(beta[0]),
        "b_R": float(beta[1]),
        "b_P": float(beta[2]),
        "r_squared": float(r_squared) if np.isfinite(r_squared) else np.nan,
        "rank": int(rank),
        "n": int(len(y)),
        "always_zero": bool(np.allclose(y, 0.0)),
    }


def fit_household(quantity_regular, quantity_premium, regular_price, premium_price) -> dict:
    """Two independent OLS fits. Coefficients are entirely data-determined."""
    regular = fit_ols(quantity_regular, regular_price, premium_price)
    premium = fit_ols(quantity_premium, regular_price, premium_price)
    # Q_R = a_R + b_RR R + b_RP P
    # Q_P = a_P + b_PR R + b_PP P
    coef = (
        regular["a"],
        regular["b_R"],
        regular["b_P"],
        premium["a"],
        premium["b_R"],
        premium["b_P"],
    )
    return {
        "coef": coef,
        "r_squared_regular": regular["r_squared"],
        "r_squared_premium": premium["r_squared"],
        "rank_regular": regular["rank"],
        "rank_premium": premium["rank"],
        "n": regular["n"],
        "regular_always_zero": regular["always_zero"],
        "premium_always_zero": premium["always_zero"],
    }


def coef_from_mapping(mapping) -> tuple[float, float, float, float, float, float]:
    return tuple(float(mapping[name]) for name in COEF_NAMES)


def quantities(prices, coef) -> tuple[float, float]:
    regular_price, premium_price = np.asarray(prices, dtype=float)
    a_R, b_RR, b_RP, a_P, b_PR, b_PP = coef
    q_regular = a_R + b_RR * regular_price + b_RP * premium_price
    q_premium = a_P + b_PR * regular_price + b_PP * premium_price
    return float(q_regular), float(q_premium)


def revenue(prices, coef) -> float:
    """Quadratic revenue R * Q_R(R, P) + P * Q_P(R, P)."""
    regular_price, premium_price = np.asarray(prices, dtype=float)
    q_regular, q_premium = quantities(prices, coef)
    return float(regular_price * q_regular + premium_price * q_premium)


def gradient(prices, coef) -> np.ndarray:
    """Analytical gradient of quadratic revenue with respect to (R, P).

    dRev/dR = Q_R + R * dQ_R/dR + P * dQ_P/dR
    dRev/dP = Q_P + P * dQ_P/dP + R * dQ_R/dP
    """
    regular_price, premium_price = np.asarray(prices, dtype=float)
    _a_R, b_RR, b_RP, _a_P, b_PR, b_PP = coef
    q_regular, q_premium = quantities(prices, coef)
    d_regular = q_regular + regular_price * b_RR + premium_price * b_PR
    d_premium = q_premium + premium_price * b_PP + regular_price * b_RP
    return np.array([d_regular, d_premium], dtype=float)


def hessian(coef) -> np.ndarray:
    _a_R, b_RR, b_RP, _a_P, b_PR, b_PP = coef
    cross = b_RP + b_PR
    return np.array([[2.0 * b_RR, cross], [cross, 2.0 * b_PP]], dtype=float)


def revenue_is_concave(coef) -> bool:
    eigenvalues = np.linalg.eigvalsh(hessian(coef))
    return bool(np.all(eigenvalues <= 1e-8))


def linear_objective_vector(revenue_gradient) -> np.ndarray:
    """Objective vector for ``linprog``.

    ``linprog`` minimises ``c @ x``. Maximising the linearised revenue
    ``g @ x`` is the same programme with ``c = -g``.
    """
    return -np.asarray(revenue_gradient, dtype=float)


def price_bounds_from_sample(regular_price, premium_price, sd_multiplier: float) -> dict:
    """Finite price box used to stop an otherwise unbounded price.

    Lower bounds are 0, as required. Upper bounds are the observed maximum
    plus ``sd_multiplier`` sample standard deviations of that price. The
    multiplier is a guardrail choice; the location of the guardrail is
    calculated from the sample, not typed in as a demand coefficient.
    """
    regular = np.asarray(regular_price, dtype=float)
    premium = np.asarray(premium_price, dtype=float)
    regular_sd = float(np.std(regular, ddof=1)) if len(regular) > 1 else 0.0
    premium_sd = float(np.std(premium, ddof=1)) if len(premium) > 1 else 0.0
    return {
        "R_lower": 0.0,
        "R_upper": float(np.max(regular) + sd_multiplier * regular_sd),
        "P_lower": 0.0,
        "P_upper": float(np.max(premium) + sd_multiplier * premium_sd),
        "R_observed_min": float(np.min(regular)),
        "R_observed_max": float(np.max(regular)),
        "P_observed_min": float(np.min(premium)),
        "P_observed_max": float(np.max(premium)),
        "R_observed_mean": float(np.mean(regular)),
        "P_observed_mean": float(np.mean(premium)),
        "upper_sd_multiplier": float(sd_multiplier),
    }


def build_constraint_system(coef, price_bounds: dict) -> dict:
    """Build ``A_ub @ x <= b_ub`` and solver bounds for ``x = [R, P]``.

    Rows are assembled from this household's OLS coefficients and from the
    price box. Nothing here is a copied regression slope.

    Demand, written as an upper bound:
        Q_R >= 0  <=>  -b_RR R - b_RP P <= a_R
        Q_P >= 0  <=>  -b_PR R - b_PP P <= a_P
    Prices:
        -R <= -R_lower,  -P <= -P_lower,  R <= R_upper,  P <= P_upper.

    A numerically zero demand row (quantity predicted to be identically zero)
    is dropped when it is redundant, and reported as infeasible when it says
    ``0 <= negative``.
    """
    a_R, b_RR, b_RP, a_P, b_PR, b_PP = coef
    r_lower = float(price_bounds["R_lower"])
    r_upper = float(price_bounds["R_upper"])
    p_lower = float(price_bounds["P_lower"])
    p_upper = float(price_bounds["P_upper"])

    rows = [
        (np.array([-b_RR, -b_RP], dtype=float), a_R, "Q_regular>=0"),
        (np.array([-b_PR, -b_PP], dtype=float), a_P, "Q_premium>=0"),
        (np.array([-1.0, 0.0], dtype=float), -r_lower, "R>=R_lower"),
        (np.array([0.0, -1.0], dtype=float), -p_lower, "P>=P_lower"),
    ]
    if np.isfinite(r_upper):
        rows.append((np.array([1.0, 0.0], dtype=float), r_upper, "R<=R_upper"))
    if np.isfinite(p_upper):
        rows.append((np.array([0.0, 1.0], dtype=float), p_upper, "P<=P_upper"))

    kept_rows = []
    kept_rhs = []
    kept_names = []
    dropped = []
    structurally_infeasible = False
    for normal, rhs, name in rows:
        if float(np.linalg.norm(normal)) < 1e-12:
            if rhs < -1e-9:
                structurally_infeasible = True
                dropped.append(f"{name} (infeasible identity)")
            else:
                dropped.append(f"{name} (redundant identity)")
            continue
        kept_rows.append(normal)
        kept_rhs.append(rhs)
        kept_names.append(name)

    A_ub = np.vstack(kept_rows) if kept_rows else np.zeros((0, 2))
    b_ub = np.asarray(kept_rhs, dtype=float)
    solver_bounds = [
        (r_lower, None if not np.isfinite(r_upper) else r_upper),
        (p_lower, None if not np.isfinite(p_upper) else p_upper),
    ]
    return {
        "A_ub": A_ub,
        "b_ub": b_ub,
        "bounds": solver_bounds,
        "names": kept_names,
        "dropped": dropped,
        "infeasible": structurally_infeasible,
        "price_bounds": price_bounds,
    }


def is_feasible(prices, A_ub, b_ub, tol: float = FEASIBILITY_TOL) -> bool:
    if A_ub.size == 0:
        return True
    return bool(np.all(A_ub @ np.asarray(prices, dtype=float) <= b_ub + tol))


def _intersection(normal_i, rhs_i, normal_j, rhs_j):
    matrix = np.vstack([normal_i, normal_j])
    if abs(np.linalg.det(matrix)) < 1e-10:
        return None
    try:
        point = np.linalg.solve(matrix, np.array([rhs_i, rhs_j], dtype=float))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def dedupe_points(points, tol: float = 1e-6) -> list[np.ndarray]:
    unique = []
    for point in points:
        point = np.asarray(point, dtype=float)
        if not np.all(np.isfinite(point)):
            continue
        if any(np.linalg.norm(point - kept) <= tol for kept in unique):
            continue
        unique.append(point)
    return unique


def polygon_vertices(A_ub, b_ub) -> list[np.ndarray]:
    """Vertices of a 2-dimensional polyhedron, by pairwise binding constraints."""
    if len(b_ub) < 2:
        return []
    found = []
    for i in range(len(b_ub)):
        for j in range(i + 1, len(b_ub)):
            point = _intersection(A_ub[i], b_ub[i], A_ub[j], b_ub[j])
            if point is not None and is_feasible(point, A_ub, b_ub):
                found.append(point)
    return dedupe_points(found)


def _trust_bounds(prices, delta: float, solver_bounds):
    (r_lower, r_upper), (p_lower, p_upper) = solver_bounds
    r_upper = np.inf if r_upper is None else r_upper
    p_upper = np.inf if p_upper is None else p_upper
    r_lo = max(r_lower, float(prices[0]) - delta)
    r_hi = min(r_upper, float(prices[0]) + delta)
    p_lo = max(p_lower, float(prices[1]) - delta)
    p_hi = min(p_upper, float(prices[1]) + delta)
    if r_lo > r_hi:
        r_lo = r_hi = float(np.clip(prices[0], r_lower, r_upper))
    if p_lo > p_hi:
        p_lo = p_hi = float(np.clip(prices[1], p_lower, p_upper))
    return [(r_lo, r_hi), (p_lo, p_hi)]


def _box_span(solver_bounds) -> tuple[float, bool]:
    (r_lower, r_upper), (p_lower, p_upper) = solver_bounds
    finite = (
        r_upper is not None
        and p_upper is not None
        and np.isfinite(r_upper)
        and np.isfinite(p_upper)
    )
    span = 0.0
    if r_upper is not None and np.isfinite(r_upper):
        span = max(span, float(r_upper) - float(r_lower))
    if p_upper is not None and np.isfinite(p_upper):
        span = max(span, float(p_upper) - float(p_lower))
    if span <= 0:
        span = 100.0
    return span, finite


def successive_linear_programme(revenue_fn, gradient_fn, system: dict, starts) -> dict:
    """Maximise ``revenue_fn`` by a sequence of trust-region linear programmes.

    At reference ``x0`` the linear programme is

        minimise    c @ x
        subject to  A_ub @ x <= b_ub
                    x inside the trust box around x0 (passed as ``bounds``)

    with ``c = -gradient_fn(x0)``. The true revenue, not the linear
    surrogate, decides whether the step is kept. Repeating this is successive
    linear programming (Griffith and Stewart, 1961): a local linear model of
    a smooth nonlinear objective, re-centered until the trust box collapses.
    Several feasible starts are used because a non-concave quadratic can have
    its maximum on the boundary while the gradient still points at a poor vertex.
    """
    A_ub = system["A_ub"]
    b_ub = system["b_ub"]
    solver_bounds = system["bounds"]
    if system["infeasible"]:
        return _empty_solve("infeasible", "Demand constraints cannot be satisfied.")

    span, _finite_box = _box_span(solver_bounds)
    best = None

    for start in dedupe_points(starts, tol=1e-5):
        if not is_feasible(start, A_ub, b_ub):
            continue
        prices = np.asarray(start, dtype=float).copy()
        delta = TRUST_INIT_FRACTION * span
        iterations = 0
        last_status = None
        last_c = linear_objective_vector(gradient_fn(prices))
        for _ in range(MAX_ITER):
            iterations += 1
            last_c = linear_objective_vector(gradient_fn(prices))
            trust = _trust_bounds(prices, delta, solver_bounds)
            result = linprog(last_c, A_ub=A_ub, b_ub=b_ub, bounds=trust, method="highs")
            last_status = int(result.status)
            # A trust box is bounded, so status 3 here is a numerical failure
            # rather than an unbounded price. The structural unboundedness test
            # is ``is_revenue_unbounded``, applied before this loop.
            if result.status != 0:
                delta *= TRUST_SHRINK
                if delta < TRUST_MIN:
                    break
                continue
            proposal = np.asarray(result.x, dtype=float)
            step = float(np.linalg.norm(proposal - prices))
            proposed_revenue = float(revenue_fn(proposal))
            current_revenue = float(revenue_fn(prices))
            if proposed_revenue > current_revenue + 1e-9:
                prices = proposal
                full_trust_step = step >= 0.9 * delta * np.sqrt(2.0)
                delta = min(delta * TRUST_GROW, span) if full_trust_step else (
                    delta * TRUST_SHRINK if step < 0.3 * delta else delta
                )
            else:
                delta *= TRUST_SHRINK
            if step < STATIONARY_STEP and delta < 1e-3:
                break

        stationarity = _directional_improvement(gradient_fn, prices, system)
        record = {
            "prices": prices,
            "revenue": float(revenue_fn(prices)),
            "iterations": iterations,
            "last_c": last_c,
            "linprog_status": last_status if last_status is not None else stationarity["status"],
            "directional_improvement": stationarity["improvement"],
            "converged": bool(
                np.isfinite(stationarity["improvement"])
                and stationarity["improvement"] <= STATIONARITY_TOL
            ),
        }
        if best is None or record["revenue"] > best["revenue"] + 1e-8:
            best = record

    if best is None:
        return _empty_solve(
            "infeasible",
            "No price inside the bounds keeps predicted demand non-negative.",
        )

    status = "optimal" if best["converged"] else "iteration_limit"
    message = {
        "optimal": "Successive linear programme converged to a first-order point.",
        "iteration_limit": "Iteration limit reached before the trust region collapsed.",
    }[status]
    return {
        "status": status,
        "message": message,
        "prices": best["prices"],
        "revenue": best["revenue"],
        "iterations": best["iterations"],
        "objective_c": best["last_c"],
        "linprog_status": best["linprog_status"],
        "directional_improvement": best["directional_improvement"],
        "converged": best["converged"],
    }


def _directional_improvement(gradient_fn, prices, system) -> dict:
    """Best linearised gain ``g @ (z - x)`` over the full structural polytope.

    This is the LP the user-facing solve reduces to once the trust region has
    collapsed: same ``A_ub``, ``b_ub``, and price bounds, objective ``c = -g``.
    A non-positive value means no feasible direction improves the linear model.
    """
    gradient_at_prices = np.asarray(gradient_fn(prices), dtype=float)
    objective = linear_objective_vector(gradient_at_prices)
    result = linprog(
        objective,
        A_ub=system["A_ub"],
        b_ub=system["b_ub"],
        bounds=system["bounds"],
        method="highs",
    )
    if result.status == 3:
        return {"improvement": np.inf, "unbounded": True, "status": 3}
    if result.status != 0:
        return {"improvement": np.nan, "unbounded": False, "status": int(result.status)}
    improvement = float(gradient_at_prices @ (np.asarray(result.x, dtype=float) - prices))
    return {"improvement": improvement, "unbounded": False, "status": 0}


def _empty_solve(status: str, message: str) -> dict:
    return {
        "status": status,
        "message": message,
        "prices": np.array([np.nan, np.nan]),
        "revenue": np.nan,
        "iterations": 0,
        "objective_c": np.array([np.nan, np.nan]),
        "linprog_status": {"infeasible": 2, "unbounded": 3}.get(status, 4),
        "directional_improvement": np.nan,
        "converged": False,
    }


def recession_rays(system: dict) -> list[np.ndarray]:
    """Unit recession directions of ``A_ub x <= b_ub`` together with the price bounds.

    A direction ``d`` is a recession direction when ``x + t d`` stays feasible for
    every ``t >= 0``. In two dimensions those directions lie along the edges of
    the recession cone, which are orthogonal to one binding normal and satisfy
    the remaining homogeneous inequalities. A nonempty polytope has no nonzero
    recession direction, so quadratic revenue on the default price box is bounded.
    """
    (r_lower, r_upper), (p_lower, p_upper) = system["bounds"]
    normals = []
    if system["A_ub"].size:
        normals.extend(np.asarray(row, dtype=float) for row in system["A_ub"])
    if r_upper is not None:
        normals.append(np.array([1.0, 0.0]))
    if r_lower is not None:
        normals.append(np.array([-1.0, 0.0]))
    if p_upper is not None:
        normals.append(np.array([0.0, 1.0]))
    if p_lower is not None:
        normals.append(np.array([0.0, -1.0]))
    if not normals:
        return [
            np.array([1.0, 0.0]),
            np.array([-1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0.0, -1.0]),
        ]
    matrix = np.vstack(normals)
    rays = []
    for normal in normals:
        tangent = np.array([-normal[1], normal[0]], dtype=float)
        scale = float(np.linalg.norm(tangent))
        if scale < 1e-12:
            continue
        for sign in (1.0, -1.0):
            direction = sign * tangent / scale
            if np.all(matrix @ direction <= 1e-7):
                rays.append(direction)
    return dedupe_points(rays)


def is_revenue_unbounded(coef, system: dict) -> bool:
    """True when quadratic revenue tends to infinity along some feasible ray.

    Positive curvature along a recession direction is enough. Zero curvature
    is unbounded when the slope along that ray is positive somewhere feasible.
    Negative curvature dies out, so that ray does not make the programme unbounded.
    """
    if system["infeasible"]:
        return False
    rays = recession_rays(system)
    if not rays:
        return False
    curvature_matrix = hessian(coef)
    for direction in rays:
        curvature = 0.5 * float(direction @ curvature_matrix @ direction)
        if curvature > 1e-9:
            return True
        if curvature < -1e-9:
            continue
        slope_at_origin = float(gradient(np.zeros(2), coef) @ direction)
        slope_gradient = curvature_matrix @ direction
        if float(np.linalg.norm(slope_gradient)) <= 1e-8:
            if slope_at_origin > 1e-8:
                return True
            continue
        # Maximise the directional slope over the feasible set. An unbounded
        # slope, or a positive maximum slope, means revenue grows without a
        # finite upper bound along this ray.
        slope_programme = linprog(
            linear_objective_vector(slope_gradient),
            A_ub=system["A_ub"],
            b_ub=system["b_ub"],
            bounds=system["bounds"],
            method="highs",
        )
        if slope_programme.status == 3:
            return True
        if slope_programme.status == 0:
            best_slope = slope_at_origin + float(slope_gradient @ slope_programme.x)
            if best_slope > 1e-8:
                return True
    return False


def feasible_start(system: dict):
    """Any feasible point, from a Phase-I linear programme with a zero objective."""
    if system["infeasible"]:
        return None
    result = linprog(
        np.zeros(2),
        A_ub=system["A_ub"],
        b_ub=system["b_ub"],
        bounds=system["bounds"],
        method="highs",
    )
    if result.status != 0:
        return None
    return np.asarray(result.x, dtype=float)


def candidate_starts(system: dict, preferred) -> list[np.ndarray]:
    starts = []
    phase_one = feasible_start(system)
    if phase_one is not None:
        starts.append(phase_one)
    vertices = polygon_vertices(system["A_ub"], system["b_ub"])
    starts.extend(vertices)
    if vertices:
        centre = np.mean(np.vstack(vertices), axis=0)
        starts.append(centre)
    starts.extend(preferred)
    return [point for point in dedupe_points(starts) if is_feasible(point, system["A_ub"], system["b_ub"])]


def critical_point(coef):
    """Unconstrained stationary point of quadratic revenue, if the Hessian is invertible."""
    try:
        point = np.linalg.solve(hessian(coef), np.array([-coef[0], -coef[3]], dtype=float))
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(point)):
        return None
    return point


def max_quadratic_on_segment(start, end, coef):
    """Exact maximum of quadratic revenue on the segment from ``start`` to ``end``."""
    start = np.asarray(start, dtype=float)
    direction = np.asarray(end, dtype=float) - start
    slope = float(gradient(start, coef) @ direction)
    curvature = 0.5 * float(direction @ hessian(coef) @ direction)
    candidates = [0.0, 1.0]
    if curvature < -1e-12:
        peak = -slope / (2.0 * curvature)
        if 0.0 <= peak <= 1.0:
            candidates.append(float(peak))
    best_t = max(candidates, key=lambda t: revenue(start + t * direction, coef))
    point = start + best_t * direction
    return point, float(revenue(point, coef))


def exact_quadratic_optimum(coef, system: dict):
    """Global maximum of quadratic revenue on the LP feasible polygon.

    Used only as a check on the successive linear programme. With two prices
    the maximum is either the interior critical point, when revenue is concave
    there and the point is feasible, or the best point on an edge. Each edge
    is a univariate quadratic, maximised in closed form.
    """
    if system["infeasible"]:
        return None
    vertices = polygon_vertices(system["A_ub"], system["b_ub"])
    if not vertices:
        return None
    candidates = list(vertices)
    A_ub, b_ub = system["A_ub"], system["b_ub"]
    for index in range(len(b_ub)):
        tight = [vertex for vertex in vertices if abs(float(A_ub[index] @ vertex) - b_ub[index]) <= 1e-5]
        if len(tight) < 2:
            continue
        tangent = np.array([-A_ub[index, 1], A_ub[index, 0]], dtype=float)
        ordered = sorted(tight, key=lambda vertex: float(vertex @ tangent))
        edge_point, _edge_revenue = max_quadratic_on_segment(ordered[0], ordered[-1], coef)
        if is_feasible(edge_point, A_ub, b_ub, tol=1e-6):
            candidates.append(edge_point)
    interior = critical_point(coef)
    if interior is not None and revenue_is_concave(coef) and is_feasible(interior, A_ub, b_ub, tol=1e-6):
        candidates.append(interior)
    best = max(candidates, key=lambda point: revenue(point, coef))
    return {"R": float(best[0]), "P": float(best[1]), "revenue": float(revenue(best, coef))}


def solve_revenue_lp(coef, price_bounds: dict, preferred_starts=None) -> dict:
    """Solve one household (or any single linear-demand system) by successive LPs."""
    system = build_constraint_system(coef, price_bounds)
    if system["infeasible"] or is_revenue_unbounded(coef, system):
        status = "infeasible" if system["infeasible"] else "unbounded"
        message = (
            "Demand constraints cannot be satisfied."
            if status == "infeasible"
            else "Revenue increases without bound along a feasible price ray. A finite upper price guardrail is required."
        )
        solved = _empty_solve(status, message)
        solved["system"] = system
        solved["coef"] = coef
        solved["quadratic"] = None
        solved["revenue_gap"] = np.nan
        return solved
    preferred = [] if preferred_starts is None else list(preferred_starts)
    starts = candidate_starts(system, preferred)
    solved = successive_linear_programme(
        revenue_fn=lambda prices: revenue(prices, coef),
        gradient_fn=lambda prices: gradient(prices, coef),
        system=system,
        starts=starts,
    )
    solved["system"] = system
    solved["coef"] = coef
    solved["quadratic"] = exact_quadratic_optimum(coef, system) if solved["status"] != "infeasible" else None
    if solved["quadratic"] is not None and np.isfinite(solved["revenue"]):
        gap = float(solved["revenue"] - solved["quadratic"]["revenue"])
        solved["revenue_gap"] = gap
        if solved["status"] == "iteration_limit" and abs(gap) <= 1e-3:
            solved["status"] = "optimal"
            solved["converged"] = True
            solved["message"] = (
                "Successive linear programme matches the exact quadratic maximum on this polygon."
            )
    else:
        solved["revenue_gap"] = np.nan
    return solved


def solve_pooled_revenue_lp(coefs, price_bounds: dict, preferred_starts=None) -> dict:
    """One price pair maximising the sum of household revenues.

    Demand constraints are stacked, one pair per household, because every
    household must still be on the non-negative side of its own demand
    boundary. The objective gradient is the sum of the household gradients,
    since total revenue is the sum of revenues. This is a comparison
    aggregate. The owner asked for a separate programme per household, and
    that remains the primary result.
    """
    summed = tuple(float(sum(coef[index] for coef in coefs)) for index in range(6))
    # Stack each household's demand rows. The price box is added once, from the
    # summed system, so the same R <= R_upper row is not copied three times.
    pieces = []
    rhs_values = []
    names = []
    dropped = []
    infeasible = False
    for household_index, coef in enumerate(coefs):
        part = build_constraint_system(coef, price_bounds)
        infeasible = infeasible or part["infeasible"]
        for row, rhs, name in zip(part["A_ub"], part["b_ub"], part["names"]):
            if name.startswith("Q_"):
                pieces.append(row)
                rhs_values.append(rhs)
                names.append(f"household_{household_index + 1}:{name}")
        dropped.extend(part["dropped"])
    price_only = build_constraint_system(summed, price_bounds)
    for row, rhs, name in zip(price_only["A_ub"], price_only["b_ub"], price_only["names"]):
        if not name.startswith("Q_"):
            pieces.append(row)
            rhs_values.append(rhs)
            names.append(name)
    system = {
        "A_ub": np.vstack(pieces) if pieces else np.zeros((0, 2)),
        "b_ub": np.asarray(rhs_values, dtype=float),
        "bounds": price_only["bounds"],
        "names": names,
        "dropped": dropped,
        "infeasible": infeasible or price_only["infeasible"],
        "price_bounds": price_bounds,
    }
    if system["infeasible"] or is_revenue_unbounded(summed, system):
        status = "infeasible" if system["infeasible"] else "unbounded"
        message = (
            "Demand constraints cannot be satisfied for every household at the same prices."
            if status == "infeasible"
            else "Total revenue increases without bound along a feasible price ray. A finite upper price guardrail is required."
        )
        solved = _empty_solve(status, message)
        solved["system"] = system
        solved["coef"] = summed
        solved["quadratic"] = None
        solved["revenue_gap"] = np.nan
        return solved
    preferred = [] if preferred_starts is None else list(preferred_starts)
    starts = candidate_starts(system, preferred)

    def total_revenue(prices):
        return float(sum(revenue(prices, coef) for coef in coefs))

    def total_gradient(prices):
        total = np.zeros(2)
        for coef in coefs:
            total = total + gradient(prices, coef)
        return total

    solved = successive_linear_programme(total_revenue, total_gradient, system, starts)
    solved["system"] = system
    solved["coef"] = summed
    # Exact check uses the summed quadratic on the stacked polygon.
    solved["quadratic"] = exact_quadratic_optimum(summed, system) if solved["status"] != "infeasible" else None
    if solved["quadratic"] is not None and np.isfinite(solved["revenue"]):
        solved["revenue_gap"] = float(solved["revenue"] - solved["quadratic"]["revenue"])
    else:
        solved["revenue_gap"] = np.nan
    return solved


def _binding_flags(prices, coef, price_bounds) -> dict:
    if not np.all(np.isfinite(prices)):
        return {
            "R_bound_active": "",
            "P_bound_active": "",
            "Q_regular_binding": False,
            "Q_premium_binding": False,
        }
    regular_price, premium_price = prices
    q_regular, q_premium = quantities(prices, coef)
    r_bound = ""
    p_bound = ""
    if regular_price <= price_bounds["R_lower"] + 1e-4:
        r_bound = "lower"
    elif regular_price >= price_bounds["R_upper"] - 1e-4:
        r_bound = "upper"
    if premium_price <= price_bounds["P_lower"] + 1e-4:
        p_bound = "lower"
    elif premium_price >= price_bounds["P_upper"] - 1e-4:
        p_bound = "upper"
    return {
        "R_bound_active": r_bound,
        "P_bound_active": p_bound,
        "Q_regular_binding": bool(abs(q_regular) <= 1e-4),
        "Q_premium_binding": bool(abs(q_premium) <= 1e-4),
    }


def _notes(model: dict, solved: dict, price_bounds: dict) -> str:
    if solved["status"] == "infeasible":
        return solved["message"]
    if solved["status"] == "unbounded":
        return solved["message"]
    prices = solved["prices"]
    notes = []
    if not revenue_is_concave(model["coef"]):
        notes.append(
            "Quadratic revenue is not concave for this household, so the successive LP is multi-started and checked against the exact edge maximum."
        )
    flags = _binding_flags(prices, model["coef"], price_bounds)
    if flags["R_bound_active"] == "upper":
        notes.append("Regular price is on the upper guardrail, which is there to stop the price going to infinity.")
    if flags["P_bound_active"] == "upper":
        notes.append("Premium price is on the upper guardrail.")
    if flags["R_bound_active"] == "lower" or flags["P_bound_active"] == "lower":
        notes.append("A price is on its lower bound of zero.")
    if flags["Q_regular_binding"]:
        notes.append("Predicted Regular demand is zero, so the household is at the point of stopping Regular purchases.")
    if flags["Q_premium_binding"]:
        notes.append("Predicted Premium demand is zero, so the household is at the point of stopping Premium purchases.")
    outside = (
        prices[0] > price_bounds["R_observed_max"] + 1e-6
        or prices[0] < price_bounds["R_observed_min"] - 1e-6
        or prices[1] > price_bounds["P_observed_max"] + 1e-6
        or prices[1] < price_bounds["P_observed_min"] - 1e-6
    )
    if outside:
        notes.append(
            "The optimum lies outside the observed experimental price range, so it extrapolates the linear OLS fit."
        )
    if model["regular_always_zero"]:
        notes.append(
            "Regular quantity is identically zero in the sample, so the Regular OLS coefficients are zero. Regular revenue is zero; Regular price only matters through the cross effect on Premium demand."
        )
    if model["premium_always_zero"]:
        notes.append("Premium quantity is identically zero in the sample.")
    if solved["status"] == "iteration_limit":
        notes.append(solved["message"])
    return " ".join(notes)


def _result_row(label: str, model: dict, solved: dict, price_bounds: dict, mean_prices) -> dict:
    a_R, b_RR, b_RP, a_P, b_PR, b_PP = model["coef"]
    prices = np.asarray(solved["prices"], dtype=float)
    q_regular, q_premium = (np.nan, np.nan)
    revenue_regular = np.nan
    revenue_premium = np.nan
    if np.all(np.isfinite(prices)):
        q_regular, q_premium = quantities(prices, model["coef"])
        revenue_regular = float(prices[0] * q_regular)
        revenue_premium = float(prices[1] * q_premium)
    baseline_q_regular, baseline_q_premium = quantities(mean_prices, model["coef"])
    flags = _binding_flags(prices, model["coef"], price_bounds)
    quadratic = solved.get("quadratic") or {}
    objective_c = np.asarray(solved["objective_c"], dtype=float)
    return {
        "Household": label,
        "R_opt": float(prices[0]),
        "P_opt": float(prices[1]),
        "max_revenue": float(solved["revenue"]) if np.isfinite(solved["revenue"]) else np.nan,
        "revenue_regular": revenue_regular,
        "revenue_premium": revenue_premium,
        "Q_regular": q_regular,
        "Q_premium": q_premium,
        "status": solved["status"],
        "converged": bool(solved["converged"]),
        "iterations": int(solved["iterations"]),
        "linprog_status": int(solved["linprog_status"]) if solved["linprog_status"] is not None else np.nan,
        "objective_c_R": float(objective_c[0]) if np.all(np.isfinite(objective_c)) else np.nan,
        "objective_c_P": float(objective_c[1]) if np.all(np.isfinite(objective_c)) else np.nan,
        "directional_improvement": float(solved["directional_improvement"])
        if np.isfinite(solved["directional_improvement"])
        else np.nan,
        "revenue_concave": revenue_is_concave(model["coef"]),
        "quadratic_R": quadratic.get("R", np.nan),
        "quadratic_P": quadratic.get("P", np.nan),
        "quadratic_revenue": quadratic.get("revenue", np.nan),
        "revenue_gap": solved.get("revenue_gap", np.nan),
        "baseline_revenue": float(revenue(mean_prices, model["coef"])),
        "baseline_Q_regular": baseline_q_regular,
        "baseline_Q_premium": baseline_q_premium,
        "mean_price_feasible": is_feasible(
            mean_prices, solved["system"]["A_ub"], solved["system"]["b_ub"]
        )
        if solved.get("system") is not None
        else False,
        "a_R": a_R,
        "b_RR": b_RR,
        "b_RP": b_RP,
        "a_P": a_P,
        "b_PR": b_PR,
        "b_PP": b_PP,
        "r_squared_regular": model["r_squared_regular"],
        "r_squared_premium": model["r_squared_premium"],
        "ols_rank_regular": model["rank_regular"],
        "ols_rank_premium": model["rank_premium"],
        "n_obs": model["n"],
        "regular_always_zero": model["regular_always_zero"],
        "premium_always_zero": model["premium_always_zero"],
        **price_bounds,
        **flags,
        "constraint_names": ";".join(solved["system"]["names"]) if solved.get("system") else "",
        "notes": _notes(model, solved, price_bounds),
        "solver": "scipy.optimize.linprog/highs",
        "linearisation": "successive_first_order_taylor_trust_region",
    }


def optimise_prices_detailed(
    csv_path=None,
    sd_multipliers=UPPER_SD_MULTIPLIERS,
    default_multiplier: float = DEFAULT_UPPER_SD_MULTIPLIER,
) -> dict:
    """Fit per-household OLS demand and solve a revenue LP for every household.

    Returns DataFrames that downstream sensitivity checks and plots can reuse
    without refitting: coefficients, bounds, optimal prices, demands, true
    quadratic revenue, solver status, and the exact quadratic comparison.
    """
    frame = pd.read_csv(resolve_data_path(csv_path))
    regular_price, premium_price = load_prices(frame)
    mean_prices = np.array([float(np.mean(regular_price)), float(np.mean(premium_price))])
    models = []
    for label, regular_column, premium_column in discover_households(frame):
        fitted = fit_household(
            frame[regular_column].to_numpy(dtype=float),
            frame[premium_column].to_numpy(dtype=float),
            regular_price,
            premium_price,
        )
        fitted["label"] = label
        models.append(fitted)

    sensitivity_rows = []
    common_rows = []
    for multiplier in sd_multipliers:
        bounds = price_bounds_from_sample(regular_price, premium_price, multiplier)
        preferred = [mean_prices]
        for model in models:
            solved = solve_revenue_lp(model["coef"], bounds, preferred_starts=preferred)
            sensitivity_rows.append(_result_row(model["label"], model, solved, bounds, mean_prices))
        pooled = solve_pooled_revenue_lp([model["coef"] for model in models], bounds, preferred_starts=preferred)
        # The pooled objective uses summed coefficients; R^2 is not defined for that sum.
        pooled_model = {
            "coef": pooled["coef"],
            "r_squared_regular": np.nan,
            "r_squared_premium": np.nan,
            "rank_regular": np.nan,
            "rank_premium": np.nan,
            "n": models[0]["n"],
            "regular_always_zero": False,
            "premium_always_zero": False,
        }
        common_rows.append(
            _result_row("All households (one price menu)", pooled_model, pooled, bounds, mean_prices)
        )

    sensitivity = pd.DataFrame(sensitivity_rows)
    common = pd.DataFrame(common_rows)
    results = sensitivity.loc[
        np.isclose(sensitivity["upper_sd_multiplier"], default_multiplier)
    ].reset_index(drop=True)
    common_default = common.loc[np.isclose(common["upper_sd_multiplier"], default_multiplier)].reset_index(drop=True)
    finite_revenue = results["max_revenue"].to_numpy(dtype=float)
    total = float(np.nansum(finite_revenue))
    return {
        "results": results,
        "sensitivity": sensitivity,
        "common_price": common_default,
        "common_price_sensitivity": common,
        "total_separate_revenue": total,
        "mean_prices": mean_prices,
        "formulation": FORMULATION_SUMMARY,
        "n_households": int(len(results)),
    }


def assess_price_scenario(result_row, regular_price: float, premium_price: float) -> dict:
    """OLS demand and revenue at a caller-chosen price, using that household's row."""
    coef = coef_from_mapping(result_row)
    q_regular, q_premium = quantities((regular_price, premium_price), coef)
    inside_box = (
        result_row["R_lower"] - 1e-8 <= regular_price <= result_row["R_upper"] + 1e-8
        and result_row["P_lower"] - 1e-8 <= premium_price <= result_row["P_upper"] + 1e-8
    )
    demand_ok = q_regular >= -1e-8 and q_premium >= -1e-8
    algebraic = float(regular_price * q_regular + premium_price * q_premium)
    return {
        "Q_regular": q_regular,
        "Q_premium": q_premium,
        "revenue": algebraic if demand_ok else np.nan,
        "algebraic_revenue": algebraic,
        "feasible": bool(inside_box and demand_ok),
        "inside_price_box": bool(inside_box),
        "demand_nonnegative": bool(demand_ok),
    }


def _zero_demand_segment(intercept, slope_regular, slope_premium, price_bounds, samples: int = 200):
    r_lower, r_upper = price_bounds["R_lower"], price_bounds["R_upper"]
    p_lower, p_upper = price_bounds["P_lower"], price_bounds["P_upper"]
    if abs(slope_premium) >= 1e-10:
        regular_grid = np.linspace(r_lower, r_upper, samples)
        premium_grid = -(intercept + slope_regular * regular_grid) / slope_premium
        mask = (premium_grid >= p_lower) & (premium_grid <= p_upper)
        return regular_grid[mask], premium_grid[mask]
    if abs(slope_regular) < 1e-10:
        return np.array([]), np.array([])
    regular_root = -intercept / slope_regular
    if r_lower <= regular_root <= r_upper:
        return np.array([regular_root, regular_root]), np.array([p_lower, p_upper])
    return np.array([]), np.array([])


def feasible_region_figure(result_row, observed_prices, scenario_regular=None, scenario_premium=None):
    """Price-plane figure: demand boundaries, feasible polygon, LP optimum.

    ``observed_prices`` needs columns ``P_Regular`` and ``P_Premium``. The
    logistic decision map in the dashboard is a different model; this figure
    is the OLS feasible set the linear programme actually uses.
    """
    coef = coef_from_mapping(result_row)
    price_bounds = {
        "R_lower": float(result_row["R_lower"]),
        "R_upper": float(result_row["R_upper"]),
        "P_lower": float(result_row["P_lower"]),
        "P_upper": float(result_row["P_upper"]),
    }
    system = build_constraint_system(coef, {**price_bounds, **{
        key: float(result_row[key])
        for key in (
            "R_observed_min",
            "R_observed_max",
            "P_observed_min",
            "P_observed_max",
            "R_observed_mean",
            "P_observed_mean",
            "upper_sd_multiplier",
        )
        if key in result_row.index
    }})
    # ``build_constraint_system`` only reads the four price limits plus the
    # keys it stores through. Supply the observed keys when present; the
    # builder itself only needs the four limits, so fill anything missing.
    for key, value in price_bounds.items():
        system["price_bounds"].setdefault(key, value)

    r_lower, r_upper = price_bounds["R_lower"], price_bounds["R_upper"]
    p_lower, p_upper = price_bounds["P_lower"], price_bounds["P_upper"]
    regular_axis = np.linspace(r_lower, r_upper, 90)
    premium_axis = np.linspace(p_lower, p_upper, 90)
    regular_grid, premium_grid = np.meshgrid(regular_axis, premium_axis)
    q_regular = coef[0] + coef[1] * regular_grid + coef[2] * premium_grid
    q_premium = coef[3] + coef[4] * regular_grid + coef[5] * premium_grid
    revenue_grid = regular_grid * q_regular + premium_grid * q_premium
    revenue_grid = np.where((q_regular >= -1e-8) & (q_premium >= -1e-8), revenue_grid, np.nan)

    figure = go.Figure()
    if np.isfinite(revenue_grid).any():
        figure.add_trace(
            go.Contour(
                x=regular_axis,
                y=premium_axis,
                z=revenue_grid,
                colorscale="YlOrRd",
                contours={"coloring": "fill", "showlines": False},
                colorbar={"title": "Revenue (€)", "thickness": 12},
                hovertemplate="Regular €%{x:.1f}<br>Premium €%{y:.1f}<br>Revenue €%{z:.1f}<extra></extra>",
                name="Revenue",
            )
        )
    vertices = polygon_vertices(system["A_ub"], system["b_ub"])
    if len(vertices) >= 3:
        centre = np.mean(np.vstack(vertices), axis=0)
        ordered = sorted(vertices, key=lambda point: np.arctan2(point[1] - centre[1], point[0] - centre[0]))
        polygon_r = [point[0] for point in ordered] + [ordered[0][0]]
        polygon_p = [point[1] for point in ordered] + [ordered[0][1]]
        figure.add_trace(
            go.Scatter(
                x=polygon_r,
                y=polygon_p,
                mode="lines",
                name="Feasible boundary",
                line={"color": "#172033", "width": 2},
                hoverinfo="skip",
            )
        )
    regular_line_r, regular_line_p = _zero_demand_segment(coef[0], coef[1], coef[2], price_bounds)
    premium_line_r, premium_line_p = _zero_demand_segment(coef[3], coef[4], coef[5], price_bounds)
    if len(regular_line_r):
        figure.add_trace(
            go.Scatter(
                x=regular_line_r,
                y=regular_line_p,
                mode="lines",
                name="Regular demand = 0",
                line={"color": "#1677ff", "width": 2, "dash": "dash"},
            )
        )
    if len(premium_line_r):
        figure.add_trace(
            go.Scatter(
                x=premium_line_r,
                y=premium_line_p,
                mode="lines",
                name="Premium demand = 0",
                line={"color": "#ff8a1f", "width": 2, "dash": "dash"},
            )
        )
    observed = observed_prices[["P_Regular", "P_Premium"]].drop_duplicates()
    figure.add_trace(
        go.Scatter(
            x=observed["P_Regular"],
            y=observed["P_Premium"],
            mode="markers",
            name="Observed weeks",
            marker={"size": 9, "color": "#667085", "line": {"color": "white", "width": 1}},
            hovertemplate="Observed<br>Regular €%{x:.0f}<br>Premium €%{y:.0f}<extra></extra>",
        )
    )
    if scenario_regular is not None and scenario_premium is not None:
        figure.add_trace(
            go.Scatter(
                x=[scenario_regular],
                y=[scenario_premium],
                mode="markers",
                name="Scenario",
                marker={"symbol": "diamond", "size": 14, "color": "#1677ff", "line": {"color": "white", "width": 1}},
                hovertemplate="Scenario<br>Regular €%{x:.1f}<br>Premium €%{y:.1f}<extra></extra>",
            )
        )
    if np.isfinite(result_row["R_opt"]) and np.isfinite(result_row["P_opt"]):
        figure.add_trace(
            go.Scatter(
                x=[result_row["R_opt"]],
                y=[result_row["P_opt"]],
                mode="markers",
                name="LP optimum",
                marker={"symbol": "star", "size": 18, "color": "#172033", "line": {"color": "white", "width": 1}},
                hovertemplate=(
                    "LP optimum<br>Regular €%{x:.2f}<br>Premium €%{y:.2f}<br>"
                    f"Revenue €{result_row['max_revenue']:.2f}<extra></extra>"
                ),
            )
        )
    pad_r = 0.06 * (r_upper - r_lower)
    pad_p = 0.06 * (p_upper - p_lower)
    household = result_row["Household"] if "Household" in result_row.index else "Household"
    figure.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 36, "b": 10},
        title={"text": f"{household}: OLS feasible prices and revenue", "font": {"size": 14}},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, Arial, sans-serif", "color": "#172033"},
        xaxis={
            "title": "Regular price R (€)",
            "range": [r_lower - pad_r, r_upper + pad_r],
            "gridcolor": "#e8edf3",
            "zeroline": False,
        },
        yaxis={
            "title": "Premium price P (€)",
            "range": [p_lower - pad_p, p_upper + pad_p],
            "gridcolor": "#e8edf3",
            "zeroline": False,
        },
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hoverlabel={"bgcolor": "white"},
    )
    return figure


def sensitivity_figure(sensitivity: pd.DataFrame):
    """Optimal Regular price against the upper-guardrail rule, by household.

    Households whose quadratic revenue is not concave ride the Regular-price
    cap. This figure is the check that makes that visible.
    """
    figure = go.Figure()
    for household, group in sensitivity.groupby("Household", sort=False):
        ordered = group.sort_values("upper_sd_multiplier")
        figure.add_trace(
            go.Scatter(
                x=ordered["R_upper"],
                y=ordered["R_opt"],
                mode="lines+markers",
                name=str(household),
                hovertemplate=(
                    f"{household}<br>Regular cap €%{{x:.1f}}<br>Optimal Regular €%{{y:.1f}}"
                    "<br>Premium €%{customdata[0]:.1f}<br>Revenue €%{customdata[1]:.1f}<extra></extra>"
                ),
                customdata=np.column_stack([ordered["P_opt"], ordered["max_revenue"]]),
            )
        )
    figure.add_trace(
        go.Scatter(
            x=sensitivity["R_upper"],
            y=sensitivity["R_upper"],
            mode="lines",
            name="Regular price cap",
            line={"color": "#98a2b3", "dash": "dot"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 36, "b": 10},
        title={"text": "How the Regular-price guardrail moves the LP solution", "font": {"size": 14}},
        paper_bgcolor="white",
        plot_bgcolor="white",
        font={"family": "Inter, Arial, sans-serif", "color": "#172033"},
        xaxis={"title": "Upper bound on Regular price (€)", "gridcolor": "#e8edf3", "zeroline": False},
        yaxis={"title": "Optimal Regular price (€)", "gridcolor": "#e8edf3", "zeroline": False},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hoverlabel={"bgcolor": "white"},
    )
    return figure


def _finite_difference_gradient(coef, prices, step: float = 1e-5) -> np.ndarray:
    prices = np.asarray(prices, dtype=float)
    numerical = np.zeros(2)
    for index in range(2):
        forward = prices.copy()
        backward = prices.copy()
        forward[index] += step
        backward[index] -= step
        numerical[index] = (revenue(forward, coef) - revenue(backward, coef)) / (2 * step)
    return numerical


def run_sanity_checks(detail: dict) -> list[str]:
    """Plausibility checks against the repo data and the exact quadratic maximum."""
    failures = []
    results = detail["results"]
    if results.empty:
        return ["No household results were produced."]
    labels = list(results["Household"])
    if labels != ["Household 1", "Household 2", "Household 3"]:
        failures.append(
            f"Expected Household 1, Household 2 and Household 3 from the quantity columns, got {labels}."
        )
    for _, row in results.iterrows():
        label = row["Household"]
        if row["status"] not in {"optimal", "infeasible", "unbounded", "iteration_limit"}:
            failures.append(f"{label}: unexpected status {row['status']}.")
            continue
        if row["status"] != "optimal":
            failures.append(f"{label}: solver status is {row['status']}, not optimal.")
            continue
        if not (row["R_lower"] - 1e-5 <= row["R_opt"] <= row["R_upper"] + 1e-5):
            failures.append(f"{label}: Regular price {row['R_opt']} is outside bounds.")
        if not (row["P_lower"] - 1e-5 <= row["P_opt"] <= row["P_upper"] + 1e-5):
            failures.append(f"{label}: Premium price {row['P_opt']} is outside bounds.")
        if row["Q_regular"] < -1e-5 or row["Q_premium"] < -1e-5:
            failures.append(
                f"{label}: negative demand at the optimum (Q_R={row['Q_regular']}, Q_P={row['Q_premium']})."
            )
        coef = coef_from_mapping(row)
        # The constraint matrix has to move when the coefficients move.
        system = build_constraint_system(coef, row)
        slack = system["b_ub"] - system["A_ub"] @ np.array([row["R_opt"], row["P_opt"]])
        if np.any(slack < -1e-5):
            failures.append(f"{label}: optimum violates A_ub x <= b_ub.")
        objective = linear_objective_vector(gradient(np.array([row["R_opt"], row["P_opt"]]), coef))
        if objective.shape != (2,):
            failures.append(f"{label}: objective vector is not length 2.")
        if not np.allclose(objective, -gradient(np.array([row["R_opt"], row["P_opt"]]), coef)):
            failures.append(f"{label}: objective vector is not the negated gradient.")
        numerical = _finite_difference_gradient(coef, np.array([row["R_observed_mean"], row["P_observed_mean"]]))
        analytic = gradient(np.array([row["R_observed_mean"], row["P_observed_mean"]]), coef)
        if np.max(np.abs(numerical - analytic)) > 1e-3:
            failures.append(f"{label}: revenue gradient does not match a finite difference.")
        if np.isfinite(row["revenue_gap"]) and abs(row["revenue_gap"]) > 1e-2:
            failures.append(
                f"{label}: LP revenue differs from the exact quadratic maximum by {row['revenue_gap']:.4f}."
            )
        if row["revenue_concave"] and abs(row["directional_improvement"]) > 1e-2:
            failures.append(f"{label}: concave revenue is not stationary at the reported optimum.")
    # Matrices differ across households because the OLS slopes differ.
    if len(results) >= 2:
        first = build_constraint_system(coef_from_mapping(results.iloc[0]), results.iloc[0])
        second = build_constraint_system(coef_from_mapping(results.iloc[1]), results.iloc[1])
        if np.allclose(first["A_ub"], second["A_ub"]) and np.allclose(first["b_ub"], second["b_ub"]):
            failures.append("Household constraint systems are identical; they should follow each OLS fit.")
    common = detail["common_price"]
    if len(common) != 1 or common.iloc[0]["status"] != "optimal":
        failures.append("Common-price aggregate did not solve to optimality.")
    else:
        common_row = common.iloc[0]
        for _, row in results.iterrows():
            predicted = quantities((common_row["R_opt"], common_row["P_opt"]), coef_from_mapping(row))
            if predicted[0] < -1e-5 or predicted[1] < -1e-5:
                failures.append(f"Common price makes {row['Household']} demand negative.")
        if np.isfinite(common_row["revenue_gap"]) and abs(common_row["revenue_gap"]) > 1e-2:
            failures.append(
                f"Common-price LP differs from its quadratic maximum by {common_row['revenue_gap']:.4f}."
            )
    return failures


def run_solver_edge_cases() -> list[str]:
    """Infeasible and unbounded programmes should be reported, not raised."""
    failures = []
    bounds = {
        "R_lower": 0.0,
        "R_upper": 100.0,
        "P_lower": 0.0,
        "P_upper": 100.0,
        "R_observed_min": 0.0,
        "R_observed_max": 100.0,
        "P_observed_min": 0.0,
        "P_observed_max": 100.0,
        "R_observed_mean": 50.0,
        "P_observed_mean": 50.0,
        "upper_sd_multiplier": 0.0,
    }
    # Q_R = -1 for every price, so the demand row is 0 <= -1 after the slopes vanish.
    infeasible = solve_revenue_lp((-1.0, 0.0, 0.0, 1.0, 0.0, -0.01), bounds)
    if infeasible["status"] != "infeasible":
        failures.append(f"Expected infeasible status, got {infeasible['status']}.")
    open_bounds = dict(bounds)
    open_bounds["R_upper"] = np.inf
    open_bounds["P_upper"] = np.inf
    # Q_R = 1 + 0.2 R grows with Regular price, so revenue R*Q_R is unbounded above
    # once the Regular price has no finite cap. Premium demand still cuts off P.
    unbounded = solve_revenue_lp((1.0, 0.2, 0.0, 1.0, 0.0, -0.01), open_bounds, preferred_starts=[np.array([10.0, 10.0])])
    if unbounded["status"] != "unbounded":
        failures.append(f"Expected unbounded status, got {unbounded['status']}.")
    # The same increasing Regular demand is bounded once the price guardrail is finite.
    capped = solve_revenue_lp((1.0, 0.2, 0.0, 1.0, 0.0, -0.01), bounds, preferred_starts=[np.array([10.0, 10.0])])
    if capped["status"] != "optimal":
        failures.append(f"Expected the finite guardrail to make the programme optimal, got {capped['status']}.")
    elif abs(capped["prices"][0] - bounds["R_upper"]) > 1e-2:
        failures.append(
            f"Expected Regular price on the upper guardrail, got {capped['prices'][0]:.4f}."
        )
    return failures


def main() -> bool:
    detail = optimise_prices_detailed()
    results = detail["results"]
    columns = [
        "Household",
        "R_opt",
        "P_opt",
        "max_revenue",
        "Q_regular",
        "Q_premium",
        "status",
        "revenue_concave",
        "quadratic_revenue",
        "revenue_gap",
        "R_bound_active",
        "P_bound_active",
    ]
    print(FORMULATION_SUMMARY)
    print()
    print(results[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print()
    print(
        f"Sum of household-specific optima (a separate price menu for each household): "
        f"€{detail['total_separate_revenue']:.4f}"
    )
    common = detail["common_price"].iloc[0]
    print(
        f"Single price menu for every household: R={common['R_opt']:.4f}, P={common['P_opt']:.4f}, "
        f"revenue=€{common['max_revenue']:.4f}, status={common['status']}"
    )
    print()
    for _, row in results.iterrows():
        print(f"{row['Household']}: {row['notes']}")
        print(
            f"  OLS Q_R = {row['a_R']:.4f} + ({row['b_RR']:.4f}) R + ({row['b_RP']:.4f}) P"
            f"    [R²={row['r_squared_regular']:.3f}]"
        )
        premium_r2 = "n/a" if not np.isfinite(row["r_squared_premium"]) else f"{row['r_squared_premium']:.3f}"
        regular_r2_note = premium_r2
        print(
            f"  OLS Q_P = {row['a_P']:.4f} + ({row['b_PR']:.4f}) R + ({row['b_PP']:.4f}) P"
            f"    [R²={regular_r2_note}]"
        )
    print()
    failures = run_sanity_checks(detail) + run_solver_edge_cases()
    if failures:
        print("SANITY CHECKS FAILED")
        for failure in failures:
            print(f"- {failure}")
        return False
    print("SANITY CHECKS PASSED")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
