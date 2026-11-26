# DKL, ADKF-IFT and few-shot adaptation: closed for this article (section 10)

## The question

Could a stronger learned ordering, such as deep kernel learning, ADKF-IFT or adaptation from a few support
observations, save meaningful planning time once the native implementation is used?

## The bound

The bound is measured on the final integrated benchmark (`results/ordering_ceiling.json`, produced by
`tools/research/phase7_extension/ordering_ceiling.py`). A perfect oracle examines exactly one candidate. Against
the native Joint order it saves the extra rejected examinations, each costed at the object's own measured
rejected-candidate time. The bound ignores the oracle's own inference cost, so any real model gains less.

| Population | Joint's extra candidates (mean) | Rejected-candidate cost (median) | Oracle saving per object (mean) | Share of total time (mean) | Share of total time (worst object) |
|---|---|---|---|---|---|
| Development (60 parts) | 0.93 | 1.37 ms | 1.43 ms | 0.35 % | 16 % |
| EGAD, already evaluated (60) | 0.52 | 2.77 ms | 1.14 ms | 0.21 % | 11 % |

Robustness cannot be improved by adaptation without new physical outcomes on the target object. Few-shot
adaptation with 1–20 support grasps would mean executing grasps on that object, which costs orders of magnitude
more than the 0.3–0.5 s planning time it would try to shorten. The Phase VII ceiling (about 1.3 candidates, or
5 ms, against a 0.57 s fixed cost) is confirmed, and is tighter now that the duplicated generation is gone.

## Decision

No DKL, ADKF-IFT or few-shot experiment is added to the AxisForge3D article.

The modules stay in the AxisForgeBO library: `axisforge_bo`, the deep-kernel and meta-learning code and their
tests are not removed. Those methods may be studied in separate work where evaluations are expensive. Such work
is not part of this article, and no biomedical application is mixed into it.
