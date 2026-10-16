## What / why

<!-- One paragraph: the problem, and why this approach. Link the issue: Closes #… -->

## Type of change

- [ ] Code only (no effect on model outputs)
- [ ] **Model change** (weights, architecture, training config, preprocessing)
- [ ] **Operating point change** (thresholds only)
- [ ] **Dataset change** (new data, labels, splits)
- [ ] CI / tooling / docs

## How it was tested

<!-- Commands you ran and their result. For model / threshold / data changes, link the
`integration` run whose job summary shows the validation report. -->

- [ ] `make test` passes locally (lint + types + tests)
- [ ] New behaviour has a test that fails without this change

## Regulated-change checklist (model, threshold or dataset changes only)

- [ ] Dataset version and fingerprint recorded (`model_card.json`)
- [ ] Operating points tuned on **val**, confirmed on **test** (never tuned on test)
- [ ] Release gate passed (sensitivity lower CI bound, specificity, non-inferiority vs production)
- [ ] Subgroup table reviewed (body part, site, view, hardware) — no hidden failure
- [ ] ONNX parity: 0 decision flips at the shipped thresholds
- [ ] Risk impact noted (ISO 14971): does this change a hazard or a risk control?

## Reviewer notes

<!-- Where to start reading, what you are unsure about. Keep PRs < 400 changed lines;
split refactors from behaviour changes. -->
