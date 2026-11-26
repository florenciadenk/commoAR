from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

COMMODITIES = {
    "ZS=F": "Soja",
    "CL=F": "Petroleo",
    "ZC=F": "Maiz",
    "GC=F": "Oro",
    "ZW=F": "Trigo",
}
ORDER = ["Soja", "Petroleo", "Maiz", "Oro", "Trigo"]

EXPORTS_USD = pd.Series(
    {"Soja": 9082, "Petroleo": 8116, "Maiz": 4241, "Oro": 3504, "Trigo": 2540},
    dtype=float,
)
WEIGHTS = EXPORTS_USD / EXPORTS_USD.sum()


def download_prices(period: str = "5y") -> pd.DataFrame:
    """Download and align the five international futures reference series."""
    raw = yf.download(
        list(COMMODITIES),
        period=period,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance no devolvió datos. Intentá nuevamente más tarde.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise RuntimeError("No se encontró la columna Close en la descarga.")
        prices = raw["Close"].copy()
    else:
        # Defensive fallback in case the provider changes its output format.
        if "Close" not in raw.columns:
            raise RuntimeError("No se encontró la columna Close en la descarga.")
        prices = raw[["Close"]].copy()
        if len(COMMODITIES) == 1:
            prices.columns = list(COMMODITIES)

    prices = prices.rename(columns=COMMODITIES)
    missing = [name for name in ORDER if name not in prices.columns]
    if missing:
        raise RuntimeError(f"Faltan series en la descarga: {', '.join(missing)}")

    prices = prices[ORDER].apply(pd.to_numeric, errors="coerce")
    missing_counts = prices.isna().sum()
    prices = prices.dropna(how="any")

    if prices.empty or len(prices) < 30:
        raise RuntimeError("No hay suficientes observaciones comunes para el análisis.")
    if (prices <= 0).any().any():
        raise RuntimeError("Se encontraron precios no positivos; no se pueden calcular retornos logarítmicos.")

    prices.attrs["missing_counts"] = missing_counts.to_dict()
    return prices


def build_analysis(prices: pd.DataFrame) -> dict:
    """Build the CommoAR base-100 index, returns, risk statistics and correlations."""
    base100 = prices.div(prices.iloc[0]).mul(100)
    base100["Indice_CommoAR"] = sum(base100[name] * WEIGHTS[name] for name in ORDER)

    returns = np.log(prices / prices.shift(1)).dropna()
    index_returns = np.log(
        base100["Indice_CommoAR"] / base100["Indice_CommoAR"].shift(1)
    ).dropna()

    summary = pd.DataFrame({
        "Variacion_acumulada_%": ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100,
        "Retorno_log_anualizado_%": returns.mean() * 252 * 100,
        "Volatilidad_anualizada_%": returns.std() * np.sqrt(252) * 100,
    })
    summary.loc["Indice_CommoAR"] = [
        (base100["Indice_CommoAR"].iloc[-1] / 100 - 1) * 100,
        index_returns.mean() * 252 * 100,
        index_returns.std() * np.sqrt(252) * 100,
    ]
    summary = summary.round(2)
    correlations = returns.corr().round(2)

    return {
        "prices": prices,
        "base100": base100,
        "returns": returns,
        "index_returns": index_returns,
        "summary": summary,
        "correlations": correlations,
        "start_date": prices.index.min(),
        "end_date": prices.index.max(),
        "observations": len(prices),
        "missing_counts": prices.attrs.get("missing_counts", {}),
    }


def monte_carlo(index_returns: pd.Series, simulations: int = 5000,
                days: int = 252, seed: int = 42) -> dict:
    """Simulate one-year index paths using historical daily log-return moments."""
    rng = np.random.default_rng(seed)
    daily_mean = float(index_returns.mean())
    daily_vol = float(index_returns.std())
    simulated_returns = rng.normal(daily_mean, daily_vol, size=(days, simulations))
    paths = EXPORTS_USD.sum() * np.exp(np.cumsum(simulated_returns, axis=0))
    finals = paths[-1, :]
    base = float(EXPORTS_USD.sum())
    p5, median, p95 = np.percentile(finals, [5, 50, 95])
    var_usd = base - float(p5)
    return {
        "paths": paths,
        "finals": finals,
        "base": base,
        "median": float(median),
        "p5": float(p5),
        "p95": float(p95),
        "var_usd": var_usd,
        "var_pct": var_usd / base * 100,
        "simulations": simulations,
        "days": days,
        "daily_mean": daily_mean,
        "daily_vol": daily_vol,
    }


def individual_shock(shocks_pct: dict) -> dict:
    shocks = pd.Series(shocks_pct, dtype=float).reindex(ORDER)
    impacts = EXPORTS_USD * shocks / 100
    total_impact = float(impacts.sum())
    base = float(EXPORTS_USD.sum())
    return {
        "type": "individual",
        "shocks_pct": shocks.to_dict(),
        "impacts": impacts,
        "total_impact": total_impact,
        "variation_pct": total_impact / base * 100,
        "base": base,
        "new_total": base + total_impact,
        "assumption": "Cantidades exportadas constantes; variación proporcional del valor ante cambios de precios.",
    }


def systemic_shock(trigger: str, shock_pct: float, include_contagion: bool,
                   returns: pd.DataFrame, mc: dict) -> dict:
    betas = returns.cov()[trigger] / returns[trigger].var()
    if include_contagion:
        shocks = betas * shock_pct
    else:
        shocks = pd.Series(0.0, index=ORDER)
        shocks[trigger] = shock_pct
    shocks = shocks.reindex(ORDER)

    impacts = EXPORTS_USD * shocks / 100
    direct = float(impacts[trigger])
    contagion = float(impacts.drop(trigger).sum())
    total = float(impacts.sum())
    base = float(EXPORTS_USD.sum())
    new_total = base + total

    agro = float(impacts[["Soja", "Maiz", "Trigo"]].sum())
    energy_mining = float(impacts[["Petroleo", "Oro"]].sum())

    finals = mc["finals"]
    if total < 0:
        mc_comparison = float((finals <= new_total).mean() * 100)
        mc_text = f"El {mc_comparison:.1f}% de los escenarios simulados termina en un valor igual o inferior."
    elif total > 0:
        mc_comparison = float((finals >= new_total).mean() * 100)
        mc_text = f"El {mc_comparison:.1f}% de los escenarios simulados termina en un valor igual o superior."
    else:
        mc_comparison = None
        mc_text = "No se aplicó un shock neto a la canasta."

    return {
        "type": "systemic",
        "trigger": trigger,
        "initial_shock_pct": float(shock_pct),
        "include_contagion": bool(include_contagion),
        "betas": betas,
        "shocks_pct": shocks,
        "impacts": impacts,
        "direct_impact": direct,
        "contagion_impact": contagion,
        "total_impact": total,
        "variation_pct": total / base * 100,
        "agro_impact": agro,
        "energy_mining_impact": energy_mining,
        "base": base,
        "new_total": new_total,
        "mc_comparison_pct": mc_comparison,
        "mc_text": mc_text,
        "assumption": "Cantidades constantes. El contagio se estima mediante betas históricas y no implica causalidad.",
    }
