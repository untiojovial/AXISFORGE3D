"""Summarize all frozen cohorts without rerunning model fits."""

import csv
import gzip
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/axisforge-bo"
RESULTS = DOC / "results"


def paired_interval(diff):
    values = np.asarray(diff)
    draws = np.random.default_rng(218).choice(values, (2000, len(values))).mean(axis=1)
    return f"{values.mean():.4g} [{np.quantile(draws, 0.025):.4g}, {np.quantile(draws, 0.975):.4g}]"


def main():
    with gzip.open(RESULTS / "scalar.jsonl.gz", "rt") as f:
        scalar = [json.loads(line) for line in f]
    advanced = [
        json.loads(line)
        for line in (RESULTS / "advanced_confirmatory.jsonl").read_text().splitlines()
    ]
    scaling = json.loads((RESULTS / "scaling.json").read_text())
    context = json.loads(
        (ROOT / "experiments/contextual_transfer_results/summary.json").read_text()
    )["results"]
    local = json.loads((RESULTS / "local_sphere_confirmatory.json").read_text())[
        "results"
    ]
    with (RESULTS / "paired.csv").open() as f:
        paired = list(csv.DictReader(f))
    doc = [
        "# Resultados de benchmarks · CPU",
        "Ocho funciones escalares, cuatro métodos y 30 semillas emparejadas produjeron 960 trayectorias de 16 evaluaciones. Los cuatro puntos iniciales coinciden dentro de cada pareja. Las dos implementaciones de EI recibieron las mismas alternativas por paso. Las diferencias de regret son nativo menos comparador; un valor negativo favorece a EI. Los corchetes indican intervalos bootstrap percentiles del 95 % para la **media de la diferencia emparejada**, sin corrección por comparaciones múltiples.",
        "| Función | Regret final medio EI | Diferencia frente a aleatorio [IC] | Diferencia frente a Sobol [IC] | RMSE predictivo | NLL gaussiana media | Cobertura marginal 95 % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in dict.fromkeys(r["problem"] for r in scalar):
        rows = [r for r in scalar if r["problem"] == name and r["method"] == "native"]

        def diff(method):
            p = next(
                p for p in paired if p["problem"] == name and p["comparator"] == method
            )
            return "%.3g [%.3g, %.3g]" % tuple(
                float(p[k])
                for k in ("paired_mean_difference", "bootstrap_low", "bootstrap_high")
            )

        avg = lambda k: np.mean([r[k] for r in rows])
        doc.append(
            f"| {name} | {avg('regret'):.4g} | {diff('random')} | {diff('sobol')} | {avg('rmse'):.4g} | {avg('nll'):.4g} | {avg('coverage95'):.1%} |"
        )
    doc.extend(
        [
            "",
            "Los recorridos de EI nativo y de scikit-learn coinciden para estas 240 combinaciones de semilla y función. Ambas implementaciones usan el mismo modelo especificado; el acuerdo comprueba las operaciones numéricas y la selección de candidatos. En Rastrigin los dos intervalos incluyen cero. La calibración varía por función. Una buena selección de candidatos no asegura por sí sola buenos intervalos predictivos.",
            "",
            "Cada línea del fichero crudo conserva coordenadas, valores, curva best-so-far, regret acumulado, primer cruce del umbral, tiempo y pico de memoria Python. Las columnas de predicción se calculan en 32 puntos adicionales de diagnóstico que el optimizador nunca observa. [Trayectorias escalares](results/scalar.jsonl.gz) · [Diferencias emparejadas](results/paired.csv) · [Protocolo](BENCHMARK_PROTOCOL.md)",
            "",
            "## Escenarios avanzados",
            "",
            "Se conservó la cohorte inicial de cinco semillas. Una segunda cohorte de 30 semillas nuevas repite exactamente los métodos y presupuestos, con 330 registros repartidos entre las siguientes comparaciones. Para restricciones se informa la tasa media de evaluaciones realmente infractoras, incluyendo la inicialización. Se comparan recomendaciones al mismo coste en multifidelidad, aunque el número de evaluaciones difiera.",
            "",
            "| Escenario y magnitud | Método frente a comparador | Media método | Media comparador | Diferencia emparejada [IC 95 %] |",
            "|---|---|---:|---:|---:|",
        ]
    )

    def compare(task, method, baseline, label, get):
        def values(m):
            return np.array(
                [
                    get(r)
                    for r in sorted(advanced, key=lambda z: z["seed"])
                    if r["task"] == task and r["method"] == m
                ]
            )

        a, b = values(method), values(baseline)
        doc.append(
            f"| {label} | {method} / {baseline} | {a.mean():.4g} | {b.mean():.4g} | {paired_interval(a - b)} |"
        )

    for name in ("zdt1", "dtlz2"):
        for baseline in ("random", "sobol"):
            compare(
                name,
                "qlog_ehvi",
                baseline,
                name.upper() + " · hipervolumen ↑",
                lambda r: r["hypervolume"][-1],
            )
    compare(
        "noisy_constraint",
        "feasible_ei",
        "random",
        "Restricción · tasa de violaciones ↓",
        lambda r: r["violation_rate"],
    )
    compare(
        "two_fidelities",
        "kg_per_cost",
        "random",
        "Fidelidad alta · regret ↓",
        lambda r: r["target_simple_regret"],
    )
    risks = [r for r in advanced if r["task"] == "input_risk"]
    a = np.array([r["robust_test_cvar"] for r in risks])
    b = np.array([r["nominal_test_cvar"] for r in risks])
    doc.append(
        f"| Perturbaciones · CVaR superior 90 % ↓ | decisión robusta / nominal | {a.mean():.4g} | {b.mean():.4g} | {paired_interval(a - b)} |"
    )
    a = np.array([r["local_bo_best"] for r in local])
    b = np.array([r["random_best"] for r in local])
    doc.append(
        f"| Sphere 10D · regret ↓ | BO local / aleatorio | {a.mean():.4g} | {b.mean():.4g} | {paired_interval(a - b)} |"
    )
    doc.extend(
        [
            "",
            "Las dos series multiobjetivo usan lotes de dos; el área depende de la referencia declarada por problema. En el problema restringido se usa un incumbente observado con ruido y un modelo independiente de factibilidad. Los resultados no constituyen una garantía de seguridad. La KG discreta ajusta un GP conjunto sobre coordenada y fidelidad; el GP específico multifidelidad de BoTorch se comprueba en la suite por separado. CVaR se evalúa con perturbaciones nuevas y un archivo de decisiones espaciadas. La variante de BO local se compara con el mismo presupuesto de 25 evaluaciones. [Cohorte avanzada inicial](results/advanced.jsonl) · [Confirmación](results/advanced_confirmatory.jsonl) · [BO local](results/local_sphere_confirmatory.json)",
            "",
            "## Transferencia entre tareas",
            "",
            "Treinta semillas, dos familias de historias y tres métodos generaron 180 trayectorias. Cada historia contiene cuatro observaciones previas y el presupuesto de la tarea nueva es de ocho evaluaciones. Para el método con historia frente al arranque en frío, la diferencia media de regret terminal fue:",
            "",
        ]
    )
    for group, methods in context.items():
        r = methods["context"]
        lo, hi = r["bootstrap_95_percentile_ci"]
        doc.append(
            f"- {group}: {r['paired_mean_regret_delta_vs_cold']:.4g} [{lo:.4g}, {hi:.4g}]."
        )
    doc.extend(
        [
            "",
            "En este ensayo la transferencia adversaria empeoró el regret medio; en historias relacionadas el intervalo incluye cero. La variante permutada cambia etiquetas por observación dentro de una tarea fuente. La desviación entre semillas, el error predictivo y la incertidumbre posterior se calculan a niveles distintos. [Datos, tiempos y ablación](../../experiments/contextual_transfer_results/raw.csv) · [Resumen](../../experiments/contextual_transfer_results/summary.json)",
            "",
            "## Escalado del GP exacto",
            "",
            "Medianas de tres procesos independientes, Linux x86_64, f64, seis dimensiones y un hilo BLAS. RSS es el máximo del proceso completo en MiB; incluye intérprete y bibliotecas. El tiempo de predicción corresponde a marginales, y excluye el inicio del proceso.",
            "",
            "| Observaciones | Consultas | Ajuste (ms) | Predicción (ms) | RSS pico (MiB) |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    grouped = defaultdict(list)
    for r in scaling:
        grouped[r["observations"], r["queries"]].append(r)
    for (n, m), rs in sorted(grouped.items()):
        med = lambda k: np.median([r[k] for r in rs])
        doc.append(
            f"| {n} | {m} | {med('fit_ms'):.2f} | {med('predict_ms'):.2f} | {med('peak_rss_mib'):.1f} |"
        )
    doc.extend(
        [
            "",
            "La factorización exacta crece cúbicamente en el número de observaciones. Estas medidas son específicas de esta máquina; las variantes variacionales y el entrenamiento en GPU requieren estudios de escalado propios. [Mediciones por proceso](results/scaling.json)",
            "",
            "Para reproducir la cohorte confirmatoria pasa `--confirmatory` a `tools/bo/advanced_benchmark.py` y a `tools/benchmarks/local_sphere.py`. Para reconstruir este informe desde los registros conservados ejecuta `python tools/bo/summarize.py`.",
        ]
    )
    # Blank lines delimit paragraphs and tables; keep consecutive table rows adjacent.
    doc.insert(1, "")
    doc.insert(3, "")
    (DOC / "BENCHMARK_REPORT.md").write_text("\n".join(doc) + "\n")


if __name__ == "__main__":
    main()
