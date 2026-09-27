# DetectionBench Roadmap

v1 shipped 2026-08-12: 6 datasets (LISA, ExDark, Brackish, GWHD, SeaDronesSee,
VisDrone-DET), model cards and weights public on Hugging Face, a launch
Reddit post, and a GWHD per-country domain-shift follow-up. This roadmap
starts fresh from that baseline instead of extending the pre-launch one --
the old phase numbering (correctness -> close-the-loop -> scale -> depth) is
kept because it's still the right ordering logic, but the contents are
rewritten against what's actually true today, not what was true on
2026-08-01.

**Ordering logic, unchanged from before:** ship correctness debt before new
scope, and don't let a promise made in public (a model-card claim, a Reddit
reply) sit undelivered longer than necessary -- the Reddit thread on the
launch post already surfaced two of these (see Phase 2).

---

## Phase 0 -- What v1 Actually Shipped

Recorded so this roadmap starts from an honest baseline, same as before.

- **6 datasets, 56 models trained and card-generated inside this repo:**
  LISA (8), ExDark (18), Brackish (8, YOLO-only), GWHD (9), SeaDronesSee
  (10), VisDrone-DET (3). The Reddit post's "26" for VisDrone counts the
  pre-existing YOLO zoo from the separate
  [VisDrone-dataset-python-toolkit](https://github.com/dronefreak/VisDrone-dataset-python-toolkit)
  repo, not models trained through DetectionBench -- only the 3 new RF-DETR
  cards here are actually this framework's output. Don't let that number
  imply more DetectionBench-side VisDrone coverage than exists.
- **Two-tier leaderboard**, built and verified across 3+ datasets: every HF
  model card shows the full, unfiltered per-dataset leaderboard
  (`generate_model_cards.py`); the GitHub README shows a curated,
  shortlist-only, cross-dataset table generated separately
  (`generate_readme_leaderboard.py`), both sorted by mAP@50.
  `publish_model_cards.sh` batch-publishes model card repos to HF via raw
  git (init/LFS-track/commit -s/push), single-model test mode included.
- **GWHD per-country domain-shift analysis** (`domain_breakdown.py`,
  `gwhd-domain-shift-analysis.md`): stratified eval across all 9 GWHD
  models by contributing country, reusing the same eval engines that
  produce the published aggregate numbers. Shipped in response to a real
  reviewer question, not planned in advance -- first proof that
  aggregate-hiding-domain-shift is a real, current problem in this project's
  own published numbers, not just a hypothetical one.
- **ONNX export built** (`export_onnx.py`, `configs/export_onnx.yaml`,
  `detectionbench-export-onnx`): one Hydra schema for both Ultralytics and
  RF-DETR checkpoints. Built, but not yet wired into any model card or
  verified against original PyTorch outputs for numerical drift -- see
  Phase 3.
- **Public methodology scrutiny, already happening.** The launch Reddit
  thread raised two concrete, valid gaps: (1) FLOPs/params were used as a
  cross-architecture speed proxy with no latency evidence behind it, and
  (2) precision is compared across models at each model's own independently
  chosen best-F1 confidence threshold, not a fixed or recall-matched one.
  Both replies committed to concrete follow-ups -- see Phase 2, item 1 and 2.
  Treat this as evidence the "trusted" positioning in `future.md` is
  actually being tested in public now, not just aspirational.

Carried over, still true, still not done (see Phase 1):

- Golden regression tests per evaluation pipeline -- still the single
  highest-leverage open item on this whole roadmap, per `future.md`'s
  Correctness Regression Tests section. Nothing since v1 shipped has built
  this.
- General Dataset Statistics module (class imbalance, objects/image,
  object-size distribution, computed from canonical COCO annotations) --
  still not built. `domain_breakdown.py` is real, useful, and shipped, but
  it's GWHD-specific bespoke analysis tooling with its own per-image
  metadata file, not the general per-dataset module `future.md` describes.
  Don't count it as the same deliverable.

---

## Phase 1 -- Correctness Debt (unfinished from before v1, now higher stakes)

*Same items as the old Phase 1, carried forward because they were never
done, not because they're new. Higher priority now than pre-launch: the
project is public, has outside readers checking its methodology, and a
silently-wrong number now damages real trust instead of just a personal
standard.*

1. **Golden regression tests, one per evaluation pipeline.** Pin known-good
   checkpoint + split + tight-tolerance metric fixtures for both the
   Ultralytics and RF-DETR eval paths (e.g. LISA's already-verified
   `yolo11x`/`rfdetr-nano` numbers). This is the only thing that would have
   caught the RF-DETR BGR/RGB bug automatically. Still not built.
2. **Dataset Statistics module, v1.** Class imbalance, objects/image,
   object-size distribution, computed directly from canonical COCO
   annotations. Scope v1 to the two most mature datasets with the most
   model cards riding on hand-written "Known Limitations" prose (LISA,
   ExDark) rather than all six at once.
3. **UAVDT full benchmark on the labelled-only test split.** The UAVDT adapter
   used to keep 50 tracking-only `S*` sequences (37,084 of 53,676 test
   frames) that have no detection labels, so every correct detection in them
   scored as a false positive -- re-evaluating `yolov8n`/`yolo26s` on only the
   16,592 labelled test frames roughly doubled their mAP (6.4 -> 12.3 and
   7.9 -> 14.9 mAP@50-95). The adapter now drops unlabelled sequences, but
   the existing COCO/YOLO copies must be regenerated. Still to do: (a)
   regenerate the splits and `docs/datasets/uavdt/` stats, (b) train with
   `configs/uavdt_{yolo,rfdetr}.yaml`, (c) re-evaluate every model
   (CPU is feasible while the GPU is busy: ~0.02-0.3 s/img YOLO, ~0.1-0.2
   s/img RF-DETR nano/small on the 20-thread box; every 3rd frame is a
   reasonable subsample, since frames are near-duplicates), (d) regenerate
   the 9 published `dronefreak/uavdt-*` cards and the UAVDT zoo, whose
   current numbers are roughly half of what the models achieve on labelled
   data.

---

## Phase 2 -- Deliver on Public Commitments

*Everything here was promised to a real audience this week and isn't done
yet. Treat these as the actual top of the queue -- ahead of new datasets,
ahead of new models -- because leaving them open longer erodes exactly the
credibility the launch post was trying to build.*

1. **Full PR-curve sweep dumps, both eval paths.** Right now
   `evaluate_rfdetr.py`'s `pick_best_f1_operating_point` and Ultralytics'
   internal `p_curve`/`r_curve` both compute a full precision/recall sweep
   across confidence thresholds and then discard everything except the
   single best-F1 point -- `metrics.json` never sees the rest. Persist the
   full sweep (threshold, precision, recall triples) for every future eval
   run, both families.
2. **Recall-matched precision comparison**, built on top of item 1, per the
   reddit reply commitment: precision at a few fixed recall levels (e.g.
   0.5, 0.7) across all models on at least ExDark, where the single-point
   data already suggested RF-DETR might be dominating the curve rather than
   trading precision for recall -- worth confirming for real instead of
   eyeballing single points.
3. **Wire real latency into the repo, not just promise it.** `benchmark.py`
   already measures GPU/CPU latency, FPS, VRAM -- and every model card's
   Reproducibility section already claims this is being measured -- but it
   has never actually been run or published for any of the 56 shipped
   models. Separately, real DRIVE AGX Orin + TensorRT FP16 benchmark data
   now exists (external, ad hoc, not yet in-repo) and a follow-up post was
   promised comparing RF-DETR vs. YOLO on identical hardware. Two related
   but distinct gaps to close:
   - Run `detectionbench-benchmark` for real on at least the shortlist
     models for one or two datasets, publish the numbers, stop the model
     cards making a claim the repo doesn't back up yet.
   - Write up the AGX Orin TensorRT results as their own post, and decide
     whether TensorRT/ONNX benchmarking belongs as a first-class
     `detectionbench-benchmark` mode going forward (currently raw PyTorch
     forward passes only, no TRT/ONNX Runtime path) rather than staying an
     ad hoc, out-of-repo process -- see Phase 3's ONNX item, same
     underlying gap.
4. **GWHD growth-stage stratified eval**, the explicitly-announced next
   step from the domain-shift post. Blocked on one small data-cleaning
   step first: normalize `"Post-flowering"` vs. `"Post-Flowering"` in the
   source metadata so growth stages don't fragment into near-duplicate
   groups before stratifying.

---

## Phase 3 -- Close Remaining v1 Completeness Gaps

*Not new scope -- finishing the shortlist commitment v1 already made.*

1. **LISA backfill.** Still only 8 of the 9 shortlist models: needs
   `yolo11n`/`yolov8s` (YOLO) and `rfdetr-small`/`rfdetr-medium` (RF-DETR)
   trained, evaluated, and card-regenerated to match every other dataset.
2. **Brackish RF-DETR.** `run_trainings.sh` is already staged for
   `rfdetr-nano`/`rfdetr-small`/`rfdetr-medium` on Brackish -- run it,
   evaluate, regenerate cards. Brackish is currently the only shortlist
   dataset with zero RF-DETR coverage.
3. **ONNX export: wire into model cards + verify numerical parity.** Export
   is built (Phase 0) but two things are missing before it's trustworthy
   enough to publish: (a) an actual numerical-drift check against the
   source PyTorch checkpoint (opset/simplify/half-precision are all
   plausible silent-wrongness vectors per `future.md`'s framing of exactly
   this risk), and (b) a model-card section surfacing export status, same
   as the benchmarking gap in Phase 2 item 3 -- don't repeat the pattern of
   claiming something the repo doesn't yet show.

---

## Phase 4 -- Depth, Once 1-3 Are Solid

*Opportunistic, same spirit as before -- pick these up once the foundation
isn't in question, not on a tight schedule.*

- **Confidence intervals** (multi-seed mean ± std), flagship comparisons
  only, not a blanket default -- still gated on Phase 1's regression tests
  landing first, per `future.md`'s explicit ordering ("confidence intervals
  on numbers you're not yet sure are correct is polishing the wrong thing
  first").
- **A real leaderboard page**, built from Phase 2's now-actually-published
  hardware-benchmarking data instead of staying a `future.md` aspiration.
- **VisDrone-DET YOLO backfill through DetectionBench itself**, if there's
  ever a reason to stop pointing at the separate toolkit repo for YOLO
  results -- not committed, VisDrone-DET's RF-DETR-only-here policy (Phase
  0) is deliberate for now.
- **A second model architecture family** (e.g. DINO) -- only after Phase
  1's regression-test infrastructure exists, so a new integration can't
  silently ship a correctness bug the way the RF-DETR BGR/RGB bug did.

---

## Parked / Long-Term Ideas (not committed)

Unchanged from before, still explicitly out of scope:

- Grounding DINO / open-vocabulary support -- needs a real design change to
  `DatasetSpec`'s fixed-`num_classes` assumption first, not just an adapter.
- Papers with Code metadata export.
- OBB (oriented bounding box) support -- relevant for the aerial/overhead
  datasets (SeaDronesSee, GWHD, VisDrone-DET), bigger lift than it looks
  (different label format, separate Ultralytics task head, no RF-DETR OBB
  support to fall back on).
- DocLayNet: stays a supported adapter, no further card or training work
  planned -- verified third-party DocLayNet model cards already exist on
  HF, duplicate effort for no benefit.
