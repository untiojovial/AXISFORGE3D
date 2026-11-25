# Backend comparison and design cross-review

Status: preliminary evidence for architecture review, not a dependency decision or performance benchmark. Accessed project-owned PyPI JSON metadata and descriptions on 2026-11-25; links below resolve to the queried metadata. Release dates are the upload time of the first file listed for the latest release, not the project's last commit, a guarantee of maintenance, or a compatibility test. Repository code, individual algorithms, transitive licenses, and GPU execution have not been audited. Apache license classifiers for NumPyro and Trieste do not alone specify a version: verify license files before adoption.

| Project / role | Latest PyPI version; first listed file UTC | License metadata | Maintenance signal |
| --- | --- | --- | --- |
| [BoTorch](https://pypi.org/pypi/botorch/json), BO on PyTorch | 0.18.1; 2026-06-08 | MIT | Recent release; project description calls it beta and actively developed |
| [GPyTorch](https://pypi.org/pypi/gpytorch/json), GP models | 1.15.2; 2026-02-28 | MIT | Recent release, ongoing support not independently measured |
| [PyTorch](https://pypi.org/pypi/torch/json), tensor/autograd | 2.15.0; 2026-10-28 | Composite SPDX expression including Apache-2.0, BSD, MIT, others | Recent release; inspect distribution notices |
| [JAX](https://pypi.org/pypi/jax/json), array/autodiff | 0.12.1; 2026-11-18 | Apache-2.0 | Recent release; hardware support depends on platform |
| [GPJax](https://pypi.org/pypi/gpjax/json), GP models | 1.1.0; 2026-11-10 | MIT | Recent release |
| [NumPyro](https://pypi.org/pypi/numpyro/json), probabilistic programming | 0.22.0; 2026-09-18 | Apache classifier; exact file to verify | Recent release; description says active development |
| [Trieste](https://pypi.org/pypi/trieste/json), BO on TensorFlow | 4.6.1; 2026-10-02 | Apache classifier; exact file to verify | Recent release |
| [GPflow](https://pypi.org/pypi/gpflow/json), GP on TensorFlow | 2.11.2; 2026-10-14 | Apache License 2.0 | Recent release |
| [BayesOpt](https://pypi.org/pypi/bayesopt/json), Martínez-Cantín's BO | 0.3; 2017-06-30 | AGPL>=3.0 | PyPI release is old; this does not prove repository abandonment. Investigate repository releases before deciding. AGPL obligations require review. |
| [Ax](https://pypi.org/pypi/ax-platform/json), experiment platform | 1.3.1; 2026-06-09 | MIT | Recent release; its BO uses BoTorch |

## Capabilities and fit

`D` = described in the primary project description; `B` = possible through the underlying computational framework but the specific algorithm/API has not been verified; `?` = not established. These are not implementation or performance claims. GPU means supported by the project or underlying framework subject to installed platform and drivers, not tested here. Every stack can connect to Rust through a newly built bridge/process; none has been demonstrated to integrate directly with AxisForge3D.

| Stack | CPU / GPU; differentiation | Kernels / custom probabilistic models | Acquisition / context / multiobjective / multifidelity | Embeddings / meta-learning | AxisForge3D fit |
| --- | --- | --- | --- | --- | --- |
| PyTorch + GPyTorch + BoTorch | CPU/GPU D; autodiff D | Modular GP/kernel and models D; neural integration D | Acquisitions D; context, multiobjective, multifidelity specific APIs ? | Deep kernels D; high-dimensional input support and implicit meta-learning for this project B, performance ? | Python↔Rust bridge required; candidate main stack pending prototype and pinned dependency test |
| JAX + GPJax + NumPyro | CPU/GPU D, device caveats; autodiff D | GPJax kernel guide/deep-kernel examples D; NumPyro probabilistic models D | Specific acquisition, contextual, multiobjective and multifidelity APIs ? | Deep-kernel example D; high-dimensional/meta-learning B, exact implementation ? | Separate bridge and JAX runtime; optional research backend only if evidence justifies it |
| TensorFlow + GPflow + Trieste | CPU/GPU and autodiff D | Composable kernels/likelihoods D; Trieste custom model D | Custom acquisitions, multiobjective and multifidelity D; contextual-specific API ? | High-dimensional embeddings/meta-learning B; exact methods ? | Separate TensorFlow runtime and bridge; compare before adding dependency cost |
| BayesOpt | CPU, hardware/autodiff ? | Specific custom kernels and probabilistic models ? | BO D; exact acquisition, context, multiobjective and multifidelity APIs ? | High-dimensional/meta-learning ? | Native-language route may reduce bridge cost but current interface/ABI and AGPL consequences are unverified |
| Ax + BoTorch stack | Inherits underlying stack where applicable | Inherits BoTorch where exposed; customization path ? | BO D; detailed acquisition, contextual, multiobjective and multifidelity interfaces ? | Underlying stack B; Ax-specific meta-learning ? | Additional orchestration layer may overlap own experiment infrastructure; assess only against concrete needs |

The source descriptions do not establish the full algorithm coverage, numerical accuracy, hardware performance, memory use or maintainability of any stack. In particular, support for embeddings does not establish scalability in 384 dimensions, useful transfer, ADKF-IFT, or contextual BO. Tensor autodiff alone does not establish correct implicit hypergradients. Benchmark candidates under matched dtype, initialization, optimization and evaluation budgets; include installation, CPU/GPU availability, compilation/warmup, wall time, peak memory, reproducibility, prediction and gradient agreement. Test the exact compatible versions together and report failures, not just fast runs. Alternatives should not be ruled out based on descriptions alone.

## Cross-review of research designs

The BO proposal's explicit objective direction, known-at-decision-time context and distinction among surrogate, acquisition and evaluator should be preserved in public contracts. Require explicit vector/batch shapes, dtype and device, observation noise convention and posterior latent versus observation variance; prevent accidental conversion of model uncertainty into empirical run variability. The design's proposed constrained acquisition must expose its independence assumption and feasibility semantics, not silently implement EI times marginal feasibility for correlated quantities.

The meta-learning note's support/query/task splits and time-safe context imply that context must carry provenance and availability time, task identity or an anonymous partition key, and embedding version/dimensions. Make these distinct from optimized parameters and from future outcomes; validate before an ask, not after fitting. Specify whether an encoder is frozen or trainable, and how model, optimizer and backend RNG states are checkpointed. The current notes do not establish the thesis's precise dimensions or original source algorithms; interface choices must not be presented as verified reproduction. Actual plugin contracts, acquisition catalog, scientific comparisons and numerical tests remain pending implementation and independent audit.

## Decision after prototyping

The CPU reference uses NumPy and SciPy for a small explicit fixed-parameter model: its posterior can be independently compared to scikit-learn and its state can be reconstructed from plain data. BoTorch/GPyTorch/PyTorch supply fitted GP, joint noisy acquisition, information gain, multiobjective and sparse/multitask implementations. Optional BoTorch SAAS inference brings in its upstream JAX/NumPyro dependency. Avoid a parallel GPJax/Trieste/Ax orchestration stack until a concrete algorithm or comparative measurement warrants it. The installed compatible CPU combination and versions differ from headline latest releases; see constraints-linux-cpu-py312.txt. Actual scope and tests are in README.md and BENCHMARK_REPORT.md. CUDA throughput or numerical parity could not be measured on this CPU instance.
