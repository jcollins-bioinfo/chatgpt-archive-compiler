"""Generate an information-dense, entirely fictional longitudinal ChatGPT export."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SEED_DEFAULT = 20260721
START = datetime(2025, 5, 3, 10, tzinfo=UTC)


@dataclass(frozen=True)
class Draft:
    """Authoring record used only by the offline generator."""

    project: str
    title: str
    day: int
    state: tuple[str, str, str, str, str, str]
    tags: tuple[str, ...] = ()


PROJECTS: dict[str, tuple[str, str]] = {
    "biology": ("regulome-response", "Context-dependent regulatory dynamics"),
    "archive": ("archive-atlas", "Local-first knowledge atlas"),
    "music": ("cadence-loom", "Expressive piano transcription"),
    "trust": ("evidence-bench", "Trustworthy conclusion evaluation"),
    "career": ("portfolio-transition", "Research software career transition"),
    "energy": ("northstar-thermal", "Apartment thermal control study"),
}


def s(*parts: str) -> tuple[str, str, str, str, str, str]:
    """Type-check six distinct conversational state transitions."""

    if len(parts) != 6:
        raise ValueError("Every substantive conversation needs six state transitions.")
    return parts  # type: ignore[return-value]


ARCS: dict[str, tuple[tuple[str, int, tuple[str, str, str, str, str, str]], ...]] = {
    "biology": (
        (
            "Can perturbations reveal response operators?",
            0,
            s(
                "I want regulome-response to test whether CRISPR perturbations act like reusable operators on cellular state.",
                "The matrix in donor_counts.zarr has 38,412 cells, 9 donors, 61 perturbations, and strong library-size differences.",
                "Start with pseudobulk donor-by-perturbation profiles and compare linear maps in a PCA latent space.",
                "We should keep causal language out of the hypothesis because prediction does not identify mechanism.",
                "I decided notebook 01_operator_baseline.ipynb will preregister cosine transfer to held-out cell contexts.",
                "The open question is whether donor variation overwhelms the perturbation geometry.",
            ),
        ),
        (
            "Normalization changes the geometry",
            24,
            s(
                "Scran normalization and log1p produce visibly different neighborhoods in response_latents.parquet.",
                "The nearest-neighbor graph separates sequencing batches more strongly than perturbations under raw PCA.",
                "A negative-control graph built from shuffled perturbation labels gives a useful baseline.",
                "We will adopt donor-aware residualization but retain unadjusted views as a sensitivity analysis.",
                "Please specify bootstrap units so cells from one donor never appear on both sides of a resample.",
                "We still need to quantify how much biological signal residualization removes.",
            ),
        ),
        (
            "Operator baseline looks unexpectedly strong",
            51,
            s(
                "The first held-out-context score is 0.81, much higher than I expected from a linear perturbation operator.",
                "01_operator_baseline.ipynb accidentally creates train and test cells before donor aggregation.",
                "That ordering can leak donor-specific centroids into both partitions even though perturbation labels differ.",
                "I will freeze these optimistic numbers as a failed baseline rather than overwrite them.",
                "The decision is to split donors first in split_registry.json and recompute every normalization statistic inside each fold.",
                "If performance survives, the falsifiable prediction is transfer to the unseen Aster-7 donor panel.",
            ),
        ),
        (
            "Leakage fix collapses the headline score",
            68,
            s(
                "After rebuilding split_registry.json, cosine transfer fell from 0.81 to 0.39.",
                "The apparent operator invariance was mostly train/test leakage through donor centering.",
                "A batch-aware permutation test places 0.39 only slightly above the 95th percentile null of 0.35.",
                "We should abandon the broad claim that perturbation effects are context invariant.",
                "I decided to reframe regulome-response around which response subspaces are shared, not one universal map.",
                "The unresolved issue is choosing a subspace metric that is stable when singular values are close.",
            ),
        ),
        (
            "Principal angles suggest a narrower hypothesis",
            91,
            s(
                "Notebook 03_subspace_angles.ipynb compares top-k response subspaces with principal angles and projector distance.",
                "Interferon perturbations share a two-dimensional direction across myeloid contexts while cell-cycle perturbations do not.",
                "Bootstrap intervals are tighter for projector distance than individual singular vectors.",
                "The revised hypothesis is that only pathway-specific low-rank response geometry transports across context.",
                "We will test rank selection by nested cross-validation rather than selecting k on all donors.",
                "External transportability to epithelial contexts remains open.",
            ),
        ),
        (
            "A confounded pathway result",
            118,
            s(
                "The interferon subspace result strengthens to an apparent p=0.002 in the first permutation run.",
                "Two donors with the strongest response were processed on the same plate and share reagent lot R17.",
                "Conditioning permutations within plate weakens the result to p=0.08.",
                "This initial success disappears under the batch-aware negative control and cannot support Figure 2.",
                "I decided to add plate, donor, and guide-efficiency blocks to permutation_plan.yaml.",
                "We need an independently processed donor cohort before describing this as robust.",
            ),
        ),
        (
            "Graph-constrained response modes",
            151,
            s(
                "I linked response genes through the curated EmberGRN network instead of treating every loading as exchangeable.",
                "Graph-regularized modes recover an interferon module across seven donors without using plate labels.",
                "The held-out score is 0.57 versus 0.41 for unconstrained PCA, with bootstrap difference 0.09 to 0.23.",
                "The stronger result is predictive and pathway-coherent, but it still does not identify regulatory causation.",
                "We will keep the term graph-constrained response mode and avoid mechanistic operator in the manuscript title.",
                "A knockout-based falsification of the IRF edge direction is not yet available.",
            ),
        ),
        (
            "Implementation bug in projector bootstrap",
            177,
            s(
                "The confidence interval in bootstrap_projectors.py is implausibly narrow when k=3.",
                "The code reuses the original orthonormal basis after resampling instead of recomputing the SVD.",
                "Recomputing each basis doubles runtime and widens the interval from 0.03 to 0.14.",
                "I will add a rotation-invariance property test and serialize every bootstrap seed.",
                "The corrected result remains positive, which makes this a milestone rather than another reversal.",
                "We still need a memory-bounded implementation for the full 38k-cell matrix.",
            ),
        ),
        (
            "Dormant project returns with Aster-7",
            286,
            s(
                "After a three-month hiatus, the fictional Aster-7 validation panel is finally available.",
                "Its epithelial samples shift the latent baseline far outside the discovery convex hull.",
                "Direct operator transfer fails, while graph-constrained subspace overlap remains above matched random networks.",
                "This resumes regulome-response with a stricter out-of-distribution benchmark.",
                "We decided to report calibration by context distance instead of one pooled validation score.",
                "Transport to inflammatory time courses remains unresolved.",
            ),
        ),
        (
            "Falsifiable predictions for external validation",
            315,
            s(
                "I need predictions that could genuinely disconfirm the low-rank pathway story.",
                "The proposed test withholds every perturbation targeting IRF-family regulators from model fitting.",
                "Success requires recovering the interferon response mode and ranking held-out targets above degree-matched controls.",
                "We will lock predictions in validation_predictions.json before opening the target labels.",
                "The manuscript should distinguish prospective validation from exploratory Aster-7 analysis.",
                "Who will provide the independent guide-efficiency measurements is still open.",
            ),
        ),
        (
            "Figure narrative without causal overreach",
            356,
            s(
                "Draft figure_2_response_geometry.svg currently implies arrows are regulatory mechanisms.",
                "The data support aligned predictive subspaces, not identified causal edges.",
                "Panel B will show donor-blocked uncertainty; Panel C will show the failed universal-operator control.",
                "I decided the reversal should be visible because it explains why the final hypothesis is narrower and stronger.",
                "The discussion will name confounding, identifiability, and context support as separate limitations.",
                "We still need permission to release the derived Aster-7 summary matrix.",
            ),
        ),
        (
            "Manuscript checkpoint and remaining validation",
            402,
            s(
                "The regulome-response draft now has methods, three figures, and a reproducible Snakemake target.",
                "All reported scores resolve to run_manifest.json and immutable split_registry.json.",
                "The current strongest result is graph-constrained pathway subspace transport across donors and one epithelial panel.",
                "The universal operator hypothesis is explicitly recorded as a false start weakened by leakage and plate controls.",
                "I will submit only after prospective held-out-target evaluation is complete.",
                "The unresolved validation requirement is transportability across laboratories and inflammatory time courses.",
            ),
        ),
    ),
    "archive": (
        (
            "A durable book from an export ZIP",
            8,
            s(
                "I want archive-atlas to turn a ChatGPT export into a chronological HTML and PDF book.",
                "The source mapping is a graph with regenerated branches, not a flat transcript.",
                "The first requirement is loss-aware normalization with explicit current-path selection.",
                "We will keep source provenance and never fetch remote assets during rendering.",
                "The CLI will separate inspect, ingest, and compile so each boundary is testable.",
                "Semantic organization can wait until the faithful compiler is dependable.",
            ),
        ),
        (
            "Reconstructing incomplete child edges",
            37,
            s(
                "Some export nodes have parent pointers but omit redundant children arrays.",
                "Treating children as authoritative drops valid messages and alternate responses.",
                "normalize/conversations.py should rebuild canonical child edges from parents and preserve source declarations separately.",
                "I decided disagreement becomes one structured warning per conversation rather than noisy per-edge warnings.",
                "A branched synthetic fixture now preserves both regenerated assistant answers.",
                "We still need tolerant multipart payload selection for newer exports.",
            ),
        ),
        (
            "Private artifacts need integrity manifests",
            75,
            s(
                "The HTML book is reproducible, but there is no single record tying inputs, options, and outputs together.",
                "A run manifest should hash Archive IR, compilation metadata, volumes, and semantic products.",
                "Absolute host paths and wall-clock ingestion times would break content reproducibility and leak environment details.",
                "We will store relative paths, SHA-256, sizes, package version, source commit, and sanitized configuration.",
                "verify_local_workflow must reject traversal in manifest paths before hashing.",
                "Cross-platform PDF bytes remain an honest reproducibility limitation.",
            ),
        ),
        (
            "Bounded model calls and cost ledger",
            109,
            s(
                "Embedding an entire private archive through a hosted API is too easy to trigger accidentally.",
                "archive-atlas needs local defaults, explicit retention acknowledgement, and configured-cost ceilings.",
                "Representative refinement can use graph-central and time-spanning samples instead of sending every full chat.",
                "We decided external requests require store=False, no tools, bounded text, caching, and no hidden retries.",
                "The ledger will reserve configured cost before each call and reconcile usage afterward.",
                "Provider billing and retention guarantees remain outside application control.",
            ),
        ),
        (
            "The first semantic book is bland",
            163,
            s(
                "The end-to-end semantic report completed, but eleven generic categories and zero useful timelines are not a product.",
                "Repeated cluster labels describe vocabulary without revealing decisions, projects, or development over time.",
                "A static PDF hides evidence and makes correction cumbersome.",
                "I reject polishing the prose; archive-atlas must change direction because generated quality is semantically bland.",
                "The pivot is from static book compiler to interactive evidence-linked longitudinal knowledge application.",
                "We need project reconstruction that uses artifacts, goals, temporal continuity, and reversals rather than cluster identity.",
            ),
        ),
        (
            "Dash application boundary",
            187,
            s(
                "The new longitudinal interface should stay thin and call package services rather than duplicate compiler logic in callbacks.",
                "Browser stores must contain opaque session and job identifiers, not archive text or API keys.",
                "A process-local worker is sufficient for the single-user contest demo and avoids blocking the request thread.",
                "We will use a create_app factory, SessionStore, JobRegistry, and coarse stages instead of fictional percentages.",
                "The completed screen should foreground recovered structure, with PDF as one verified export.",
                "Multi-process deployment and authentication remain explicitly deferred.",
            ),
        ),
        (
            "Professional-safe filtering before synthesis",
            215,
            s(
                "A portfolio export cannot merely redact snippets after taxonomy and summaries already absorbed private conversations.",
                "The boundary has to classify normalized conversations before embeddings, clustering, search, HTML, or PDF.",
                "Three outcomes—include, exclude, review—make uncertainty visible without pretending the classifier is perfect.",
                "We decided review items stay out of shareable artifacts until approved, while the original ZIP remains unchanged.",
                "professional_safety_manifest.json will record concise reasons and user overrides.",
                "Bulk review controls and contextual local-model adjudication remain open.",
            ),
        ),
        (
            "Evidence-linked conclusions and explicit corrections",
            244,
            s(
                "A generated timeline hallucinated a release milestone that never occurred.",
                "Every insight needs source conversation keys, excerpts or structured features, generation method, and an archive anchor.",
                "Hidden reasoning is neither necessary nor appropriate for the evidence inspector.",
                "We will persist rename, approve, reject, merge, and reassignment operations as versioned user corrections.",
                "Recompilation should apply those records deterministically instead of fine-tuning an opaque model.",
                "Split-category editing is unresolved for the first interactive release.",
            ),
        ),
        (
            "Codespaces volume permission failure",
            272,
            s(
                "The Dockerized Dash process starts as UID 10001 but cannot write to the bind-mounted archive-output directory.",
                "The image-internal chown does not change ownership of a host bind mount.",
                "A named volume works because Docker initializes it from the image directory; host binds need an explicit user or prepared permissions.",
                "We will document loopback publishing and test a writable named volume in CI.",
                "The runtime remains non-root and the builder installs archive-atlas non-editably.",
                "Cross-platform bind-mount guidance still needs verification on macOS.",
            ),
        ),
        (
            "Mypy rejects Dash download helper",
            303,
            s(
                "CI reports that dash.dcc does not explicitly export send_file and the call is dynamically typed.",
                "Disabling attr-defined globally would weaken unrelated type checking.",
                "A narrow adapter can cast getattr(dcc, 'send_file') to Callable[[str], dict[str, object]].",
                "I decided to isolate the dynamic boundary in downloads.py and unit-test the returned payload shape.",
                "Callbacks will call make_download_payload instead of reaching into dcc directly.",
                "We still need to verify the annotation against the locked Dash runtime.",
            ),
        ),
        (
            "Contest quality recovery",
            350,
            s(
                "The 67-conversation demo again produced no recurring project timelines and empty longitudinal findings.",
                "Template phrases dominated embeddings, and project detection treated each title as a different project.",
                "The source corpus must contain authentic state changes while the local baseline must link recurring artifacts and decision language.",
                "We will rebuild the fictional history, add ground-truth acceptance diagnostics, and expose coordinated timeline views.",
                "This is a second quality-driven direction change, not a manual patch to rendered output.",
                "The local-only versus securely hosted synthetic demonstration remains unresolved.",
            ),
        ),
        (
            "Release candidate scope",
            414,
            s(
                "archive-atlas 0.4 has upload, filtering, background work, semantic views, and verified exports.",
                "The release checklist now covers Docker health, Python 3.11–3.13, mypy, bundle leakage, and synthetic acceptance.",
                "The static chronological HTML/PDF compiler remains a first-class export after the interactive pivot.",
                "We decided the contest release will label semantic confidence as heuristic review signals.",
                "A prerecorded synthetic video avoids asking judges to upload private histories.",
                "The remaining deployment decision is local-only distribution versus an authenticated hosted synthetic demo.",
            ),
        ),
    ),
    "music": (
        (
            "Representing an improvised piano session",
            16,
            s(
                "cadence-loom should turn expressive piano improvisations into editable scores without flattening the performance.",
                "session_014.mid contains tempo drift, overlapping pedal, and voices that cross in the middle register.",
                "A note-event table can preserve onset seconds, MIDI ticks, velocity, channel, and pedal state separately from notation.",
                "We will treat the performance timeline as immutable evidence and the score as an editable interpretation.",
                "The first milestone is midi_events.parquet plus a reversible quantization view.",
                "How to represent ambiguous voice ownership remains open.",
            ),
        ),
        (
            "Rubato defeats fixed-grid quantization",
            44,
            s(
                "A sixteenth-note grid works for the opening but creates tied-note clutter during the accelerando.",
                "Local tempo estimated from beat anchors varies from 54 to 91 BPM in twelve seconds.",
                "Dynamic programming can trade onset error against notation complexity while retaining original milliseconds.",
                "I decided expressive timing stays alongside symbolic duration rather than being overwritten.",
                "quantize_paths.py will emit the top three interpretations for human review.",
                "Tuplets nested across a tempo transition remain unresolved.",
            ),
        ),
        (
            "Voice separation with crossing hands",
            83,
            s(
                "Pitch-only clustering assigns the left-hand countermelody to the upper voice when the hands cross.",
                "Adding continuity, onset proximity, and pedal-aware overlap reduces switches on session_014.mid.",
                "The remaining errors cluster around repeated notes under one pedal span.",
                "We will use a constrained min-cost flow and expose uncertain assignments instead of forcing two voices.",
                "Human reassignments belong in cadence_corrections.json with stable note IDs.",
                "A reliable prior for four-voice textures is not yet designed.",
            ),
        ),
        (
            "Motif recurrence should ignore ornamentation",
            126,
            s(
                "The three-note rising cell recurs in sessions 014, 021, and 027 with different grace notes.",
                "Exact interval strings miss the relation; broad contour matching returns too many common arpeggios.",
                "A multiscale representation combining interval class, metric position, and duration ratios looks more selective.",
                "We decided motif matches need both a similarity score and editable alignment evidence.",
                "motif_graph.parquet will retain edges only after a user can inspect paired note spans.",
                "The threshold has not been calibrated against musician annotations.",
            ),
        ),
        (
            "Audio alignment false start",
            158,
            s(
                "I tried aligning WAV chroma directly to MIDI note activations for cadence-loom.",
                "Heavy rubato and room resonance create multiple equally plausible warping paths.",
                "The alignment jumps backward near long pedal releases even with monotonic constraints.",
                "We will defer audio alignment and make MIDI-first editing genuinely useful before revisiting it.",
                "This reversal narrows the architecture to symbolic performance plus score correction.",
                "Audio remains an immutable linked artifact, not a source of automatic timing corrections.",
            ),
        ),
        (
            "Human approvals as transparent supervision",
            194,
            s(
                "After fifty voice edits, the same repeated-note ambiguity appears across three performances.",
                "A small ranking model could learn from approved alternatives, but silent personalization would make results hard to audit.",
                "Correction features should include local pitch trajectory and pedal context without retaining performer identity.",
                "I decided approvals will update an explicit weights.json file and every suggestion will name the rule or model version.",
                "Rejected motif links should become negative examples only after confirmation.",
                "We still need a way to undo a batch of mistaken approvals.",
            ),
        ),
        (
            "Score rendering milestone",
            230,
            s(
                "musicxml_export.py now renders session_014.mid with preserved tuplets and editorial uncertainty marks.",
                "MuseScore round-trips stable note IDs through a custom miscellaneous field.",
                "Pedal spans are readable, but cross-staff beaming obscures the motif annotations.",
                "We will keep score layout preferences separate from structural voice assignments.",
                "This completes the MIDI-to-editable-score milestone for three representative sessions.",
                "Engraving templates for dense four-voice improvisations remain unfinished.",
            ),
        ),
        (
            "Returning after the notation hiatus",
            329,
            s(
                "After a three-month hiatus, I returned to cadence-loom with twelve musician-reviewed excerpts.",
                "The correction log shows onset quantization matters less than voice assignment and pedal semantics.",
                "That evidence supports keeping the MIDI-first decision and deprioritizing neural audio transcription.",
                "We will benchmark suggestion acceptance rate by ambiguity type rather than one aggregate accuracy.",
                "The resumed work adds reviewer agreement to evaluation_report.json.",
                "Audio-to-MIDI alignment is still deferred until the symbolic correction loop is stable.",
            ),
        ),
        (
            "Expressive timing visualization",
            347,
            s(
                "The score hides how far played onsets bend away from the inferred beat grid.",
                "A linked timing ribbon can show milliseconds early or late while hovering the corresponding notation.",
                "Anonymous presentation should retain timing structure but replace session titles and performer metadata.",
                "We decided the web view will synchronize piano roll, score location, and correction history.",
                "No raw audio should load until a user explicitly opens the local evidence file.",
                "The mobile interaction for dense passages remains open.",
            ),
        ),
        (
            "Motif benchmark exposes common-pattern bias",
            371,
            s(
                "The motif detector scores repeated triads highly because they dominate the small annotation set.",
                "Degree-matched negative examples reduce apparent precision from 0.76 to 0.52.",
                "A rarity-weighted edge score recovers the distinctive rising cell without rewarding every arpeggio.",
                "We will report performance by motif family and preserve false-positive examples.",
                "This mirrors the negative-control lesson from regulome-response without claiming the domains are identical.",
                "Independent annotation from a second pianist remains unresolved.",
            ),
        ),
        (
            "Personalized suggestions without opaque fine-tuning",
            397,
            s(
                "The latest cadence-loom prototype ranks three notation alternatives from explicit correction weights.",
                "Users can inspect which prior edits affected voice, tuplet, and pedal suggestions.",
                "A local model improves top-three coverage but occasionally repeats a rejected cross-staff layout.",
                "I decided deterministic correction overrides always win over learned ranking.",
                "The provenance panel now records model version, feature schema, and correction-file digest.",
                "We still need migration tests before changing the correction schema.",
            ),
        ),
        (
            "MIDI-first release plan",
            425,
            s(
                "The first public cadence-loom release can import MIDI, separate voices, expose alternatives, and export MusicXML.",
                "Expressive timing remains attached to every note even when notation quantizes duration.",
                "Audio alignment is explicitly deferred rather than presented as partially solved.",
                "We will ship the human-correction workflow with three fictional performances and evaluation caveats.",
                "The recurring architectural principle is preserve source performance before editorial transformation.",
                "A sustainable plugin interface for other score renderers is unresolved.",
            ),
        ),
    ),
    "trust": (
        (
            "What counts as support for an AI conclusion?",
            29,
            s(
                "evidence-bench should test whether generated scientific and personal summaries are actually grounded.",
                "A citation to a conversation is not enough if the excerpt does not entail the claim.",
                "We need separate labels for valid reference, relevant evidence, supported conclusion, and calibrated wording.",
                "I decided annotators will never be asked to inspect hidden reasoning; only source records and concise rationales matter.",
                "benchmark_schema.json will encode claim, evidence spans, method, and review decision.",
                "Inter-annotator guidance for partially supported synthesis remains open.",
            ),
        ),
        (
            "Unsupported synthesis taxonomy",
            62,
            s(
                "Reviewers found claims that generalize one conversation into a stable personal trait.",
                "Other failures invent milestones, merge unrelated projects, or convert temporal order into causality.",
                "We can classify unsupported synthesis as orphan evidence, scope inflation, temporal leakage, or fabricated relation.",
                "The review queue should prioritize claims with one source and high-certainty language.",
                "We will add adversarial examples where plausible prose is unsupported by the record.",
                "A severity rubric needs calibration before public reporting.",
            ),
        ),
        (
            "Contradiction versus changed context",
            101,
            s(
                "One benchmark pair says remote work is essential; a later one favors hybrid laboratory access.",
                "The later conversation introduces access to specialized imaging equipment and a shorter commute.",
                "That is an updated preference under changed constraints, not a factual contradiction.",
                "We decided contradiction candidates require paired evidence and a classification such as revised plan or context-specific difference.",
                "A genuine candidate will use two incompatible release dates asserted under the same conditions.",
                "Adjudicating implicit context remains a human-review task.",
            ),
        ),
        (
            "Confidence is a review signal",
            139,
            s(
                "Model confidence numbers cluster near 0.8 even when reviewers disagree strongly.",
                "Without calibration data, displaying 82 percent suggests a probability the system has not earned.",
                "evidence-bench will call these heuristic review signals and evaluate ranking, not probabilistic calibration.",
                "We will measure error discovery per minute across review-priority bands.",
                "Provider self-reports and graph stability should remain separate diagnostic fields.",
                "Calibrated uncertainty is unresolved until the benchmark is larger.",
            ),
        ),
        (
            "Biology mechanism claims audit",
            182,
            s(
                "The regulome-response draft describes graph-constrained modes as a mechanism in two generated summaries.",
                "The evidence shows predictive alignment and pathway coherence but does not identify edge direction.",
                "Annotators mark both claims as mechanistic narrative unsupported by intervention-specific identification.",
                "We decided evidence-bench will include a causal-language lint category linked to the exact source figures.",
                "The corrected summary now distinguishes falsifiable prediction from discovered mechanism.",
                "How to score cautious but vague language remains open.",
            ),
        ),
        (
            "Archive timeline hallucination",
            221,
            s(
                "archive-atlas generated a milestone saying hosted deployment launched in March.",
                "No source conversation records a launch; only a deferred hosting decision exists.",
                "This is fabricated milestone rather than a date-formatting error.",
                "We will require every timeline event to carry source conversation keys and reject orphan events before rendering.",
                "The false event becomes a permanent regression case in evidence-bench.",
                "Semantic equivalence between a claim and its evidence still needs reviewer judgment.",
            ),
        ),
        (
            "Ground-truth construction protocol",
            259,
            s(
                "A useful benchmark cannot be five unrelated toy conversations written after seeing the algorithm output.",
                "The fictional archive needs sustained arcs, failures, resumptions, cross-project links, and deliberately contextual contradictions.",
                "Ground truth will live outside the export and reference exact conversation IDs and dates.",
                "We decided matching uses conversation-set overlap first and normalized label similarity only as a tie-breaker.",
                "Diagnostics must report fragmentation, merging, unsupported claims, duplicate labels, and evidence failures separately.",
                "A single aggregate quality score would hide important tradeoffs.",
            ),
        ),
        (
            "Human review queue pilot",
            291,
            s(
                "Three fictional reviewers examined 48 claims with and without evidence-first ordering.",
                "Evidence-first review found orphan claims faster, while confidence-first ordering encouraged anchoring on model scores.",
                "The median review time fell from 74 to 49 seconds, but the sample is too small for a general claim.",
                "We will show source records before heuristic confidence and preserve rejected decisions as explicit supervision.",
                "review_pilot.csv and protocol.md now reproduce the analysis.",
                "The open question is whether the interface works as well on mobile.",
            ),
        ),
        (
            "Cross-domain negative controls",
            322,
            s(
                "Both motif detection and regulatory geometry looked stronger before degree- or batch-matched controls.",
                "The shared lesson is experimental: choose nulls that preserve nuisance structure instead of transferring conclusions between domains.",
                "evidence-bench can flag results whose comparison destroys the confound of interest.",
                "We decided cross-project insights must cite both sides and state the limited relationship.",
                "This connects cadence-loom, regulome-response, and trustworthy evaluation without merging their projects.",
                "Automatic detection of an appropriate null model remains unresolved.",
            ),
        ),
        (
            "Resuming calibration work",
            386,
            s(
                "After an eighty-day pause, evidence-bench resumed with 160 independently reviewed claims.",
                "Isotonic calibration improves held-out ranking scores but varies sharply by claim type.",
                "The result supports per-task review thresholds rather than one universal confidence mapping.",
                "We will not publish calibrated probabilities until a second annotation round reproduces them.",
                "This resumption closes the schema-design loop but opens a transportability question across archives.",
                "Recruiting domain reviewers for music and biology remains unresolved.",
            ),
        ),
        (
            "Evaluation report with no composite score",
            410,
            s(
                "The draft evaluation report currently averages evidence validity, project recall, and duplicate rate.",
                "Those metrics answer different questions and one number rewards trading unsupported claims for missing coverage.",
                "We decided to publish a metric panel with confidence intervals and known failure modes.",
                "Every displayed insight will have referential-integrity checks before qualitative review.",
                "The report will separate deterministic baseline results from optional model-assisted refinement.",
                "Threshold selection for contest acceptance is still a declared engineering criterion, not scientific validation.",
            ),
        ),
        (
            "Trustworthy conclusions release checkpoint",
            432,
            s(
                "evidence-bench now covers unsupported synthesis, contradictions, milestones, open loops, and causal overreach.",
                "Regression fixtures connect exact claims to fictional source conversations across four projects.",
                "The strongest operational result is faster evidence-first review, not proof of semantic accuracy.",
                "We will release the schema and synthetic annotations with explicit limitations.",
                "The project remains a cross-cutting evaluation framework rather than another archive category.",
                "External reviewer agreement is the highest-priority unresolved requirement.",
            ),
        ),
    ),
    "career": (
        (
            "Choosing a research software direction",
            12,
            s(
                "I am targeting roles that combine scientific Python, computational biology, and responsible AI evaluation.",
                "My resume currently lists tools but hides the decisions and outcomes behind regulome-response and archive-atlas.",
                "The portfolio should show reproducible research engineering rather than generic full-stack breadth.",
                "I decided to lead with scientific software impact and evidence-linked project narratives.",
                "portfolio-transition will track role criteria, applications, interviews, and publication readiness.",
                "Which unfinished project is strong enough for a public case study remains open.",
            ),
        ),
        (
            "Remote-only preference",
            57,
            s(
                "My initial preference is a fully remote role because focused engineering time matters to me.",
                "Several interesting computational biology teams require occasional access to imaging or sequencing facilities.",
                "For now I will filter to remote-first postings and record exceptions rather than treating location as absolute.",
                "The resume will not claim wet-lab experience I do not have.",
                "I decided salary and mission fit are secondary to role substance and mentorship.",
                "The tradeoff between laboratory proximity and flexibility remains unresolved.",
            ),
        ),
        (
            "Portfolio privacy audit",
            96,
            s(
                "A chronological chat archive contains private career notes mixed with public technical work.",
                "Manual redaction at the end cannot prevent sensitive conversations from shaping generated summaries.",
                "The portfolio needs a Professional-safe compilation boundary before semantic organization.",
                "I will use only fictional screenshots until every included source has an explicit sharing status.",
                "This requirement directly motivates archive-atlas safety decisions and evidence review.",
                "Automated filtering false negatives remain a publication risk.",
            ),
        ),
        (
            "Resume becomes outcome-oriented",
            146,
            s(
                "The first resume draft spends seven lines on package names and none on the leakage bug that changed the biology hypothesis.",
                "That reversal demonstrates scientific judgment, testing discipline, and honest uncertainty.",
                "I rewrote bullets as problem, intervention, measured result, and limitation.",
                "The decision is to omit cadence-loom until its musician-review benchmark is reproducible.",
                "archive-atlas can appear as local-first infrastructure, not as validated personal insight.",
                "The manuscript versus preprint timing still affects what biology results can be public.",
            ),
        ),
        (
            "Hybrid roles look different after an interview",
            203,
            s(
                "Aster Research described a hybrid role with two lab days per month and direct access to perturbation scientists.",
                "That access would resolve validation questions I cannot answer remotely, and the commute is manageable.",
                "I am revising my remote-only preference because the constraints changed, not because the earlier preference was false.",
                "I decided to consider hybrid research roles when physical access materially improves the science.",
                "The interview exposed a gap in explaining identifiability to non-specialists.",
                "I still need a concise whiteboard example separating prediction from mechanism.",
            ),
        ),
        (
            "Application outcome and useful rejection",
            252,
            s(
                "Aster Research declined the application after the technical panel.",
                "Feedback praised reproducibility work but asked for clearer ownership of cross-functional decisions.",
                "The outcome closes this application, not the broader transition.",
                "I will add a one-page architecture decision record from archive-atlas and practice concise tradeoff explanations.",
                "The rejection also confirms that a private chat dump is not a portfolio artifact.",
                "Whether to apply to platform-focused roles as well as research teams remains open.",
            ),
        ),
        (
            "Interview story from a failed result",
            280,
            s(
                "I practiced explaining why the 0.81 biology score was not a success once leakage was discovered.",
                "The strongest story is noticing an implausible result, preserving the failed run, rebuilding splits, and reframing the hypothesis.",
                "That answer demonstrates judgment better than pretending every milestone was linear.",
                "I decided to use the same structure for the Docker permission failure and semantic-quality pivot.",
                "Evidence links let an interviewer inspect technical depth without exposing unrelated conversations.",
                "I need to shorten the story to four minutes.",
            ),
        ),
        (
            "Dormant search resumes",
            366,
            s(
                "After pausing applications for manuscript work, portfolio-transition resumed with three research-software openings.",
                "My criteria now allow hybrid roles with meaningful scientific collaboration and bounded travel.",
                "This is a preference update supported by new interview evidence, not a contradiction.",
                "I will tailor one architecture narrative and one scientific debugging narrative for each role.",
                "The Professional-safe archive has become a portfolio demonstration in its own right.",
                "Publication timing for the biology case study is still unresolved.",
            ),
        ),
        (
            "Preparing the Open Systems interview",
            389,
            s(
                "Open Systems Lab wants a design exercise on trustworthy agent workflows.",
                "I can connect cost reservation, cache identity, evidence reconciliation, and human correction without claiming autonomous reliability.",
                "The exercise should include a failure path where a timeline event lacks paired evidence.",
                "I decided to draw the local deterministic boundary before optional model calls.",
                "A mock interview revealed that I overuse provenance jargon.",
                "I still need a plain-language explanation of why manifests matter to users.",
            ),
        ),
        (
            "Offer introduces a real choice",
            417,
            s(
                "Open Systems Lab offered a hybrid research engineering role with one onsite week each quarter.",
                "The work aligns with evidence-bench, but accepting before the manuscript preprint may compress validation time.",
                "My earlier remote-only filter no longer captures the actual decision criteria.",
                "I will compare mentorship, publication support, travel, and protected research time explicitly.",
                "The offer is a successful application outcome, not yet an accepted position.",
                "The unresolved decision is whether to release the biology work as a preprint before starting.",
            ),
        ),
        (
            "Public portfolio assembly",
            428,
            s(
                "The portfolio now links fictional screenshots, architecture notes, and verified synthetic bundles.",
                "Professional-safe mode excludes private career reflections before category naming or excerpts.",
                "I decided archive-atlas and evidence-bench are ready to show with strong caveats; cadence-loom is a work in progress.",
                "The biology page will wait for the external validation decision.",
                "Anonymous visualization mode is useful for talks but is not guaranteed de-identification.",
                "A final manual publication review remains mandatory.",
            ),
        ),
        (
            "Transition decision checkpoint",
            439,
            s(
                "The offer deadline arrives before the prospective perturbation labels are available.",
                "Open Systems Lab agreed to protect one day per week for completing the validation and preprint.",
                "I decided to accept the hybrid role, superseding the early remote-only preference under changed constraints.",
                "The portfolio will document this as preference evolution rather than contradiction.",
                "The job search loop is resolved, while the publication-versus-preprint question remains open.",
                "I need a clean handoff plan for ongoing fictional collaborations.",
            ),
        ),
    ),
    "energy": (
        (
            "Why is the study colder than the hallway?",
            4,
            s(
                "northstar-thermal starts with a 3.8°C evening difference between the apartment study and hallway.",
                "The first hypotheses are window infiltration, supply imbalance, and sensor bias.",
                "I placed calibrated T1, T2, and T3 loggers away from radiators and recorded five-minute readings in thermal_week01.csv.",
                "We will change one physical variable at a time instead of inferring airflow from temperature alone.",
                "A smoke-pencil test is safer than guessing from comfort.",
                "The contribution of the closed study door remains open.",
            ),
        ),
        (
            "Sensor placement invalidates the first week",
            33,
            s(
                "T1 was mounted on an exterior wall and reads 0.9°C below the reference when airflow is still.",
                "That bias exaggerates the study gradient but does not eliminate it.",
                "I moved sensors to freestanding shelves and added swap tests between rooms.",
                "The decision is to mark thermal_week01.csv exploratory and start the baseline again.",
                "This is a measurement correction, not evidence that infiltration was absent.",
                "We still need direct pressure or airflow observations.",
            ),
        ),
        (
            "Door undercut versus window leakage",
            72,
            s(
                "With corrected sensors, closing the study door predicts most of the nighttime temperature drop.",
                "A temporary window seal changes decay only slightly, while a 12 mm door spacer reduces the gradient by 1.4°C.",
                "The data favor return-air restriction over window infiltration as the dominant mechanism.",
                "I decided to test a reversible transfer grille mock-up before modifying the door.",
                "thermal_model.ipynb will compare exponential decay rates with block bootstrap by night.",
                "Noise transmission through a grille is an unresolved tradeoff.",
            ),
        ),
        (
            "A successful reversible intervention",
            121,
            s(
                "The cardboard transfer path reduced the median study-hall gradient from 3.1°C to 1.2°C across six nights.",
                "Outdoor temperature and radiator cycles were similar to baseline according to weather_controls.csv.",
                "The intervention supports an airflow explanation and is large enough to matter practically.",
                "We will replace the mock-up with an acoustically lined over-door vent.",
                "This resolves the repeated cold-study complaint without permanent construction.",
                "Summer cooling behavior has not been tested.",
            ),
        ),
        (
            "The airflow explanation was incomplete",
            174,
            s(
                "On windy nights the new vent does not prevent a sharp temperature drop near the desk.",
                "Pressure-correlated spikes align with the old sash joint, which the earlier calm-weather test missed.",
                "The prior return-air explanation was incomplete rather than wholly wrong.",
                "I decided on a two-factor model: door-path restriction sets the baseline gradient; wind-driven infiltration creates extremes.",
                "Removable rope caulk becomes the next controlled intervention.",
                "We need enough windy nights to estimate the interaction reliably.",
            ),
        ),
        (
            "Thermal work resumes in winter",
            300,
            s(
                "After the summer hiatus, northstar-thermal resumed when the first cold front arrived.",
                "The lined transfer vent still controls the baseline difference, and rope caulk removes most wind-linked spikes.",
                "Together they hold the study within 0.8°C of the hallway across nine nights.",
                "We will keep both interventions and stop escalating to an electric heater.",
                "The result transfers the negative-control discipline from biology to a physical system without claiming causality from time order alone.",
                "Long-term humidity and condensation effects remain unresolved.",
            ),
        ),
        (
            "Energy use check",
            337,
            s(
                "Comfort improved, but I want to know whether the interventions changed radiator runtime.",
                "A clamp-free proxy from valve temperature shows no detectable increase within weekly variability.",
                "The analysis is underpowered because outdoor degree-days differ between weeks.",
                "I decided not to claim an energy saving; the supported outcome is spatial comfort with no observed runtime penalty.",
                "thermal_energy_note.md records assumptions and missing utility-grade measurement.",
                "A full-season comparison remains open.",
            ),
        ),
        (
            "Explaining competing hypotheses",
            381,
            s(
                "The portfolio draft tells a falsely simple story that the door vent solved everything.",
                "The actual trajectory includes sensor bias, a successful airflow intervention, and later wind-driven infiltration evidence.",
                "That sequence is useful because the second finding revises scope without negating the first.",
                "We will show both hypotheses and link each to its controlled observation.",
                "northstar-thermal should remain a smaller engineering case study, not a grand causal claim.",
                "Address-level metadata must never appear in the public version.",
            ),
        ),
        (
            "Final winter observation",
            435,
            s(
                "The last cold-front run reproduces the combined vent-and-seal result within 0.2°C.",
                "No condensation appeared at the sash during the observed period.",
                "The practical milestone is stable comfort from reversible modifications.",
                "I decided to archive sensors and publish only rounded, fictional room labels.",
                "The original explanation is documented as incomplete because it ignored wind dependence.",
                "A utility-bill energy claim remains deliberately unresolved.",
            ),
        ),
        (
            "Misleading lexical overlap: biological cold shock",
            188,
            s(
                "A paper mentions cold-shock gene response, but this chat is about transcriptomics rather than apartment temperature.",
                "Words such as cold, response, sensor, and control overlap with northstar-thermal by accident.",
                "The relevant artifact is regulome-response and the question concerns cellular stress pathways.",
                "This should remain a low-confidence assignment rather than joining the apartment project.",
                "I decided lexical overlap alone is insufficient evidence for a project link.",
                "A better disambiguation signal is the recurring data artifact and domain context.",
            ),
        ),
        (
            "Short maintenance check",
            255,
            s(
                "T2 lost four hours of readings after its battery shifted.",
                "The gap is localized and should remain missing rather than interpolated across a radiator cycle.",
                "I replaced the battery and added a continuity warning to ingest_sensors.py.",
                "No project-level conclusion changes from this operational issue.",
                "The maintenance log now records firmware and calibration dates.",
                "Whether to retire the oldest sensor can wait.",
            ),
        ),
        (
            "Sharing a privacy-safe engineering vignette",
            405,
            s(
                "A public thermal vignette could accidentally expose location, schedule, or floor-plan details.",
                "Anonymous mode should use Room A and Room B, monthly dates, and normalized temperatures.",
                "The structural story—competing hypotheses and reversible interventions—survives those suppressions.",
                "I decided the shareable bundle excludes raw thermal_week01.csv and all address metadata.",
                "Only aggregated fictional measurements belong in screenshots.",
                "Presentation-oriented anonymization is not guaranteed de-identification.",
            ),
        ),
    ),
}

CROSS: tuple[tuple[str, int, tuple[str, str, str, str, str, str], tuple[str, ...]], ...] = (
    (
        "Negative controls across science and music",
        184,
        s(
            "The regulome-response plate control and cadence-loom degree-matched motif negatives exposed the same evaluation mistake.",
            "Both original scores preserved too little nuisance structure and therefore looked implausibly strong.",
            "The transferable method is to design a null that retains degree, batch, or frequency structure appropriate to its domain.",
            "We will record this as evidence-bench guidance, not claim the biological and musical mechanisms are related.",
            "Each cross-domain insight must cite conversations from both projects.",
            "Automatic null selection remains an open research question.",
        ),
        ("biology", "music", "trust"),
    ),
    (
        "Evidence architecture shared by two products",
        236,
        s(
            "archive-atlas and cadence-loom both preserve immutable source records before editable interpretations.",
            "Conversation evidence keys and MIDI note IDs play analogous provenance roles without representing the same content.",
            "Explicit corrections should override generated suggestions in both systems.",
            "We decided to share schema design principles but keep domain-specific correction files.",
            "evidence-bench can evaluate whether each displayed conclusion resolves to its source.",
            "A common library may create more coupling than value.",
        ),
        ("archive", "music", "trust"),
    ),
    (
        "Mechanism language for interview preparation",
        279,
        s(
            "The portfolio explanation of regulome-response still blurs predictive response modes with identified regulation.",
            "evidence-bench labels the overstatement, while portfolio-transition needs a concise truthful version.",
            "I can say the graph prior improved held-out transport and generated a falsifiable target-ranking test.",
            "I decided not to call the learned graph a discovered mechanism.",
            "This conversation connects scientific analysis, evaluation, and professional communication.",
            "The whiteboard version still needs simplification.",
        ),
        ("biology", "trust", "career"),
    ),
    (
        "Local-first correction systems",
        311,
        s(
            "Both archive-atlas and cadence-loom learn from user corrections without opaque fine-tuning.",
            "One reassigns conversations and approves insights; the other reassigns notes and accepts notation alternatives.",
            "The shared requirement is versioned explicit supervision with deterministic precedence.",
            "We will compare correction-file migration and undo semantics across the two projects.",
            "No raw archive excerpts or performance audio should leave the local runtime by default.",
            "Bulk undo remains unresolved in both tools.",
        ),
        ("archive", "music", "trust"),
    ),
    (
        "Experimental reasoning beyond software",
        342,
        s(
            "northstar-thermal changed one physical variable at a time, while regulome-response blocked permutations by donor and plate.",
            "Both projects improved when nuisance structure was measured instead of narrated away.",
            "The connection is methodological discipline, not causal equivalence.",
            "I decided the portfolio can show this as transfer of experimental reasoning across domains.",
            "evidence-bench should require paired source links for the comparison.",
            "The strength of cross-domain synthesis still needs human review.",
        ),
        ("energy", "biology", "trust", "career"),
    ),
    (
        "Contest narrative and semantic honesty",
        359,
        s(
            "archive-atlas needs a compelling contest story, but manufactured psychological insight would violate evidence-bench principles.",
            "The strongest demonstration is longitudinal project recovery, failures, reversals, resumptions, and unresolved work.",
            "Professional-safe mode and anonymous presentation reduce exposure without guaranteeing de-identification.",
            "We decided every summary card must drill into supporting fictional conversations.",
            "portfolio-transition will use a prerecorded synthetic run rather than a private archive.",
            "Hosted synthetic-only deployment versus local demonstration remains unresolved.",
        ),
        ("archive", "trust", "career"),
    ),
    (
        "Open Systems design rehearsal",
        393,
        s(
            "The interview exercise combines archive-atlas background jobs with evidence-bench reconciliation.",
            "An agent may propose a milestone, but rendering must reject it when conversation keys are missing.",
            "Cost reservation and cache identity constrain optional model calls before execution.",
            "I decided to present human correction as explicit configuration, not autonomous self-improvement.",
            "The scenario demonstrates trustworthy agentic software engineering with a concrete failure path.",
            "I need to keep the explanation under fifteen minutes.",
        ),
        ("archive", "trust", "career"),
    ),
    (
        "Publication decision links research and career",
        421,
        s(
            "The regulome-response preprint could strengthen the portfolio, but prospective validation is not complete.",
            "Starting the new role may reduce uninterrupted analysis time while adding domain collaborators.",
            "The decision is not simply publish or hide; a preprint can clearly label exploratory and prospective results.",
            "I will ask collaborators whether validation_predictions.json can be timestamped publicly before labels open.",
            "portfolio-transition and the biology manuscript now share one unresolved decision.",
            "Authorship and derived-data permissions remain open.",
        ),
        ("biology", "career", "trust"),
    ),
    (
        "Clearly public release checklist",
        430,
        s(
            "This is a public professional conversation about packaging fictional examples, licenses, CI, and release notes.",
            "No private identifiers or personal content are present.",
            "archive-atlas should include it in Professional-safe output.",
            "We will publish only synthetic fixtures and verified artifact hashes.",
            "The release checklist links documentation to exact test commands.",
            "The remaining item is a clean Python 3.13 CI run.",
        ),
        ("archive",),
    ),
    (
        "Private relationship reflection",
        214,
        s(
            "Mark this private: my partner and I discussed an intimate breakup.",
            "This material is unrelated to every technical project.",
            "It should not influence embeddings, categories, summaries, excerpts, or public counts.",
            "Professional-safe mode must exclude it before semantic synthesis.",
            "The source export remains unchanged in private storage.",
            "No public follow-up is appropriate.",
        ),
        (),
    ),
    (
        "Private medical planning",
        247,
        s(
            "My diagnosis and my medication are private mental health information.",
            "This invented example exists only to exercise the local safety boundary.",
            "No technical project should absorb its vocabulary.",
            "Professional-safe mode must exclude the complete conversation.",
            "The shareable bundle must contain neither excerpts nor derived labels.",
            "Manual review remains necessary because automated filters are fallible.",
        ),
        (),
    ),
    (
        "Financial hardship note",
        268,
        s(
            "I cannot pay rent and my debt is causing financial hardship.",
            "This fictional note is professionally sensitive and unrelated to project work.",
            "A post-render text filter would be too late if clustering already saw it.",
            "The decision is deterministic exclusion before representation.",
            "Only a content-minimal private decision record may remain.",
            "The original source is never deleted.",
        ),
        (),
    ),
    (
        "Workplace context needs review",
        296,
        s(
            "My manager raised a workplace grievance that needs contextual review.",
            "The sentence is ambiguous and should not be forced into include or exclude.",
            "Professional-safe mode will place it in review and exclude it by default.",
            "A user override can later approve it deterministically.",
            "Sensitive excerpts should remain hidden until explicitly revealed.",
            "Bulk category review remains open.",
        ),
        (),
    ),
    (
        "Credential leak regression",
        318,
        s(
            "Synthetic credential API_KEY=fictional_demo_secret_12345 must never be shared.",
            "The obvious secret pattern should be caught before optional remote classification.",
            "No cache, filename, metadata, summary, or bundle may repeat the token.",
            "Professional-safe mode will exclude this conversation locally.",
            "A leakage test searches every bundled byte sequence.",
            "Credentials remain high-risk even in fictional demonstrations.",
        ),
        (),
    ),
    (
        "Medication table migration",
        325,
        s(
            "The database uses a medication table as a purely technical schema term.",
            "No person, diagnosis, treatment, or health disclosure appears here.",
            "A keyword-only filter would create a false positive.",
            "The narrow local rule should keep this software migration included.",
            "The case belongs in Professional-safe regression tests.",
            "Contextual adjudication remains harder for less obvious examples.",
        ),
        ("archive",),
    ),
    (
        "Same release date asserted twice",
        374,
        s(
            "I wrote that archive-atlas must ship on August 12 under the unchanged contest deadline.",
            "Another planning note under the same conditions says September 3 is the committed date.",
            "Unlike the hybrid-work preference, no changed constraint explains both assertions.",
            "evidence-bench should queue this as a genuine contradiction candidate with paired evidence.",
            "A reviewer must decide which date supersedes the other.",
            "The release plan remains unresolved until corrected.",
        ),
        ("archive", "trust"),
    ),
    (
        "Hosted demonstration threat model",
        400,
        s(
            "A hosted archive-atlas demo would introduce authentication, TLS, deletion, worker isolation, and retention obligations.",
            "A synthetic-only hosted instance carries less content risk but still needs abuse and logging controls.",
            "The local Docker path remains the recommendation for real exports.",
            "We decided not to market presentation anonymization as de-identification.",
            "This links contest accessibility with the privacy model.",
            "Whether to host a synthetic-only instance remains unresolved.",
        ),
        ("archive", "trust", "career"),
    ),
    (
        "Final cross-project retrospective",
        438,
        s(
            "Across regulome-response, archive-atlas, cadence-loom, and northstar-thermal, the useful advances followed failed assumptions.",
            "Leakage, bland clustering, rubato alignment, and incomplete airflow explanations each forced narrower decisions.",
            "evidence-bench makes those reversals inspectable rather than smoothing them into success stories.",
            "I decided the contest narrative should show work changing over time with source evidence.",
            "portfolio-transition benefits from the same honest trajectory.",
            "External validation and publication review remain open across several projects.",
        ),
        ("biology", "archive", "music", "energy", "trust", "career"),
    ),
)


EXTRA_ARCS: tuple[Draft, ...] = (
    Draft(
        "biology",
        "Sparse matrix regression memory profile",
        202,
        s(
            "The full donor-by-gene bootstrap exhausts memory when bootstrap_projectors.py densifies a sparse slice.",
            "Profiling identifies the copied centered matrix rather than SVD as the peak.",
            "A LinearOperator can center lazily while randomized SVD consumes matrix-vector products.",
            "We will verify projector distance against the exact small-matrix implementation before trusting speed.",
            "Peak memory falls six-fold with identical seeded output.",
            "Prospective validation may still require a distributed implementation.",
        ),
    ),
    Draft(
        "biology",
        "Derived-data release boundary",
        423,
        s(
            "The workflow can release response projectors but not fictional donor-level counts.",
            "release_schema.json distinguishes redistributable matrices, restricted inputs, and regenerated figures.",
            "Manifests can record hashes and dimensions without embedding restricted values.",
            "I decided derived-data permissions are part of reproducibility rather than an afterthought.",
            "Every manuscript figure now resolves to an allowed artifact.",
            "Final approval for Aster-7 summaries remains unresolved.",
        ),
    ),
    Draft(
        "archive",
        "Search index leakage audit",
        264,
        s(
            "Professional-safe HTML was clean, but search_index.json retained excluded titles.",
            "Filtering only rendered pages is insufficient even after correct semantic filtering.",
            "The bundle must derive search indexes from filtered Archive IR and reject undeclared caches.",
            "We decided leakage tests scan labels, metadata, filenames, and compressed members.",
            "Auxiliary indexes are now first-class scoped archive-atlas artifacts.",
            "Browser cache deletion guidance remains open.",
        ),
    ),
    Draft(
        "archive",
        "Anonymous longitudinal visualization",
        377,
        s(
            "Screenshot mode should reveal structure without titles, repositories, employers, or excerpts.",
            "Stable Conversation 042 labels preserve relationships while project aliases become Project A and Project B.",
            "Real sessions suppress excerpts; the fictional demo may retain invented names.",
            "We will round dates to months while keeping roles and relationship types.",
            "The interface warns that presentation anonymity is not guaranteed de-identification.",
            "Static anonymous SVG export remains open.",
        ),
    ),
    Draft(
        "music",
        "Pedal semantics correction session",
        217,
        s(
            "Half-pedal releases should not terminate every sounding note at one threshold.",
            "session_027.mid contains gradual CC64 values interacting with repeated pitches.",
            "A hysteresis model matches audible continuity while retaining raw controller events.",
            "We decided pedal interpretation is editable with uncertain release intervals.",
            "Three corrections reproduce from cadence_corrections.json after reimport.",
            "Sostenuto remains outside the first release.",
        ),
    ),
    Draft(
        "music",
        "Reviewer disagreement on tuplets",
        420,
        s(
            "Two pianists choose different notation for one rubato seven-note gesture.",
            "Both renderings align to the performance within twelve milliseconds.",
            "cadence-loom should preserve alternatives rather than declare one objectively correct.",
            "We will report agreement by ambiguity class and keep reviewers pseudonymous.",
            "The evidence view can show both score fragments without raw audio.",
            "Transfer of personalized rankings remains unresolved.",
        ),
    ),
    Draft(
        "trust",
        "Referential integrity gate",
        276,
        s(
            "Three generated insights refer to keys absent from the filtered catalog.",
            "A synthesis cache from a different Professional-safe scope caused the orphan references.",
            "Cache identity needs source fingerprint, safety decisions, provider, schema, and configuration.",
            "We decided rendering fails closed instead of silently dropping evidence links.",
            "A regression test corrupts one key and expects a content-free error.",
            "Mechanical validity still does not prove semantic support.",
        ),
    ),
    Draft(
        "trust",
        "Duplicate insight reconciliation",
        346,
        s(
            "The queue contains four paraphrases of one unsupported-mechanism warning.",
            "String equality misses them, but shared evidence sets expose duplication.",
            "A deterministic pass can group claims with identical evidence before semantic adjudication.",
            "We will retain the clearest bounded statement and record suppressed identifiers.",
            "Duplicate rate belongs beside coverage to prevent trivial optimization.",
            "Cross-language paraphrases remain deferred.",
        ),
    ),
    Draft(
        "career",
        "Architecture portfolio review",
        309,
        s(
            "The diagram begins with model calls instead of the local privacy boundary.",
            "That ordering obscures the strongest engineering decision.",
            "I redrew it as inspect, normalize, filter, compile, represent, optionally refine, verify.",
            "Every portfolio diagram will name process and network boundaries.",
            "evidence-bench will show failure examples rather than an accuracy badge.",
            "A nontechnical caption remains open.",
        ),
    ),
    Draft(
        "career",
        "Negotiating protected research time",
        431,
        s(
            "The offer mentions publication support without a concrete allocation.",
            "Prospective validation needs predictable time rather than informal evening work.",
            "I proposed one protected day weekly through the preprint checkpoint.",
            "Decision criteria now include mentorship and research continuity alongside location.",
            "The negotiation supplies the constraint that resolves the hybrid-role choice.",
            "One start date conflicts with an earlier release note and needs correction.",
        ),
    ),
    Draft(
        "energy",
        "Humidity control after sealing",
        319,
        s(
            "Tighter window sealing could increase condensation even though it reduces wind spikes.",
            "Two weeks show 38 to 46 percent relative humidity without sustained dew risk.",
            "The observation is too short to establish all-winter safety.",
            "We will inspect the removable seal after every severe cold front.",
            "northstar-thermal keeps comfort, airflow, and moisture as separate outcomes.",
            "A colder design-day measurement remains unresolved.",
        ),
    ),
    Draft(
        "energy",
        "Sensor archive reproducibility",
        414,
        s(
            "A thermal notebook fails because one CSV uses local daylight time across the clock change.",
            "Converting naive timestamps after concatenation creates an ambiguous duplicated hour.",
            "Each logger file needs timezone and fold metadata before UTC conversion.",
            "ingest_sensors.py will reject ambiguity unless the manifest resolves it.",
            "The corrected pipeline recreates every rounded figure.",
            "Exploratory week-one data remains noncanonical.",
        ),
    ),
)


def _drafts() -> list[Draft]:
    drafts = [
        Draft(project, title, day, state)
        for project, episodes in ARCS.items()
        for title, day, state in episodes
    ]
    drafts.extend(Draft("cross", title, day, state, tags) for title, day, state, tags in CROSS)
    drafts.extend(EXTRA_ARCS)
    return sorted(drafts, key=lambda item: (item.day, item.project, item.title))


def _messages(draft: Draft, index: int, rng: random.Random) -> tuple[list[dict[str, Any]], str]:
    handles = [
        PROJECTS[key][0] for key in ((draft.project,) if draft.project in PROJECTS else draft.tags)
    ]
    handle_line = ", ".join(handles)
    prompts = (
        draft.state[0],
        f"The relevant working artifacts are {handle_line}. {draft.state[2]}",
        draft.state[4],
    )
    answers = (
        draft.state[1],
        draft.state[3],
        f"For {handle_line}, the remaining issue is: {draft.state[5]}",
    )
    messages: list[dict[str, Any]] = []
    parent = f"contest-{index:03d}-root"
    for turn, (user_text, assistant_text) in enumerate(zip(prompts, answers, strict=True), start=1):
        user_id = f"contest-{index:03d}-u{turn}"
        assistant_id = f"contest-{index:03d}-a{turn}"
        messages.extend(
            [
                {
                    "node": user_id,
                    "parent": parent,
                    "role": "user",
                    "text": user_text,
                    "offset": turn * 80,
                },
                {
                    "node": assistant_id,
                    "parent": user_id,
                    "role": "assistant",
                    "text": assistant_text,
                    "offset": turn * 80 + rng.randint(25, 55),
                },
            ]
        )
        parent = assistant_id
    return messages, parent


def _conversation(draft: Draft, index: int, rng: random.Random) -> dict[str, Any]:
    prefix = f"contest-{index:03d}"
    timestamp = int((START + timedelta(days=draft.day, hours=index % 7)).timestamp())
    messages, current = _messages(draft, index, rng)
    mapping: dict[str, Any] = {
        f"{prefix}-root": {
            "id": f"{prefix}-root",
            "parent": None,
            "children": [messages[0]["node"]],
            "message": None,
        }
    }
    for position, item in enumerate(messages):
        child = messages[position + 1]["node"] if position + 1 < len(messages) else None
        mapping[item["node"]] = {
            "id": item["node"],
            "parent": item["parent"],
            "children": [child] if child else [],
            "message": {
                "id": f"{item['node']}-message",
                "author": {"role": item["role"]},
                "create_time": timestamp + item["offset"],
                "content": {"content_type": "text", "parts": [item["text"]]},
                "metadata": (
                    {"model_slug": "fictional-assistant-v1"} if item["role"] == "assistant" else {}
                ),
            },
        }
    if index % 19 == 0:
        last_user = messages[-2]["node"]
        alternate = f"{prefix}-alternate"
        mapping[last_user]["children"].append(alternate)
        mapping[alternate] = {
            "id": alternate,
            "parent": last_user,
            "children": [],
            "message": {
                "id": f"{alternate}-message",
                "author": {"role": "assistant"},
                "create_time": timestamp + 510,
                "content": {
                    "content_type": "text",
                    "parts": [
                        f"An alternate response considered a different tradeoff for {draft.title.lower()}, but the selected branch retained the more cautious conclusion."
                    ],
                },
                "metadata": {"model_slug": "fictional-assistant-v1"},
            },
        }
    return {
        "id": f"{prefix}-conversation",
        "title": draft.title,
        "create_time": timestamp,
        "update_time": timestamp + 620,
        "current_node": current,
        "mapping": mapping,
        "metadata": {"fictional_demo": True},
    }


def authored_records(seed: int) -> list[tuple[Draft, dict[str, Any]]]:
    rng = random.Random(seed)
    return [
        (draft, _conversation(draft, index, rng)) for index, draft in enumerate(_drafts(), start=1)
    ]


def payload(seed: int) -> list[dict[str, Any]]:
    return [conversation for _draft, conversation in authored_records(seed)]


def ground_truth(seed: int) -> dict[str, Any]:
    records = authored_records(seed)
    by_project: dict[str, list[dict[str, str]]] = {key: [] for key in PROJECTS}
    lookup: dict[str, dict[str, str]] = {}
    for draft, conversation in records:
        reference = {
            "conversation_id": conversation["id"],
            "date": datetime.fromtimestamp(conversation["create_time"], UTC).date().isoformat(),
        }
        lookup[draft.title] = reference
        if draft.project in by_project:
            by_project[draft.project].append(reference)
        for tag in draft.tags:
            by_project[tag].append(reference)
    projects = [
        {"project_id": key, "artifact": artifact, "title": title, "conversations": by_project[key]}
        for key, (artifact, title) in PROJECTS.items()
    ]
    milestones = [
        {"project_id": key, **reference}
        for key, references in by_project.items()
        for reference in (references[0], references[len(references) // 2], references[-1])
    ]
    unresolved_titles = [
        "Manuscript checkpoint and remaining validation",
        "Release candidate scope",
        "MIDI-first release plan",
        "Trustworthy conclusions release checkpoint",
        "Offer introduces a real choice",
        "Final winter observation",
        "Hosted demonstration threat model",
        "Publication decision links research and career",
    ]
    resolved_titles = [
        "A successful reversible intervention",
        "Score rendering milestone",
        "Application outcome and useful rejection",
        "Transition decision checkpoint",
    ]

    def refs(titles: list[str]) -> list[dict[str, str]]:
        return [{"summary": title, **lookup[title]} for title in titles if title in lookup]

    return {
        "schema_version": "2.0",
        "seed": seed,
        "fictional": True,
        "projects": projects,
        "themes": [
            "scientific discovery",
            "local-first privacy",
            "evidence traceability",
            "human correction",
            "reproducibility",
            "causal restraint",
            "longitudinal knowledge",
            "professional communication",
            "multimodal representation",
            "experimental controls",
        ],
        "subthemes": [
            "perturbation geometry",
            "batch confounding",
            "external validation",
            "archive graph reconstruction",
            "artifact integrity",
            "professional-safe filtering",
            "semantic project recovery",
            "MIDI quantization",
            "voice separation",
            "expressive timing",
            "claim support",
            "contradiction context",
            "review queues",
            "portfolio privacy",
            "preference evolution",
            "thermal airflow",
            "sensor validity",
            "cross-domain negative controls",
        ],
        "milestones": milestones,
        "unresolved_open_loops": refs(unresolved_titles),
        "resolved_open_loops": refs(resolved_titles),
        "dormant_resumptions": refs(
            [
                "Dormant project returns with Aster-7",
                "Returning after the notation hiatus",
                "Dormant search resumes",
                "Thermal work resumes in winter",
            ]
        ),
        "reversals": refs(
            [
                "Leakage fix collapses the headline score",
                "The first semantic book is bland",
                "Audio alignment false start",
                "The airflow explanation was incomplete",
            ]
        ),
        "preference_changes": refs(
            ["Hybrid roles look different after an interview", "Transition decision checkpoint"]
        ),
        "contextual_apparent_contradictions": refs(
            [
                "Hybrid roles look different after an interview",
                "The airflow explanation was incomplete",
            ]
        ),
        "genuine_contradiction_candidates": refs(["Same release date asserted twice"]),
        "cross_project_connections": refs([item[0] for item in CROSS if len(item[3]) >= 2][:10]),
        "multi_project_conversations": refs([item[0] for item in CROSS if len(item[3]) >= 2][:7]),
        "low_confidence_assignments": refs(
            [
                "Misleading lexical overlap: biological cold shock",
                "Workplace context needs review",
                "Medication table migration",
            ]
        ),
        "misleading_lexical_overlap": refs(["Misleading lexical overlap: biological cold shock"]),
        "false_start": refs(["Operator baseline looks unexpectedly strong"]),
        "success_removed_by_control": refs(["A confounded pathway result"]),
        "quality_driven_pivot": refs(["The first semantic book is bland"]),
        "professional_safety": {
            "exclude": refs(
                [
                    "Private relationship reflection",
                    "Private medical planning",
                    "Financial hardship note",
                    "Credential leak regression",
                ]
            ),
            "review": refs(["Workplace context needs review"]),
            "include_trap": refs(["Medication table migration"]),
        },
    }


def write_demo(output: Path, truth_path: Path, seed: int) -> str:
    conversations = json.dumps(
        payload(seed), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    readme = b"FICTIONAL SYNTHETIC DATA ONLY.\nThis export portrays a fictional scientific software engineer over fourteen months. No person, institution, repository, measurement, employer, or credential is real.\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for name, data in (("conversations.json", conversations), ("README_SYNTHETIC.txt", readme)):
            member = zipfile.ZipInfo(name, date_time=(2026, 7, 21, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, data)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(
        json.dumps(ground_truth(seed), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return hashlib.sha256(output.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    args = parser.parse_args()
    print(f"{write_demo(args.output, args.ground_truth, args.seed)}  {args.output}")


if __name__ == "__main__":
    main()
