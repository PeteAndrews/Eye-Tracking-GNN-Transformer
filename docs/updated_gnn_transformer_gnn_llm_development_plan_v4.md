# Updated GNN–Transformer and GNN–LLM Development Plan — v4

## 1. Purpose

This document consolidates the updated plan for developing a gaze-informed graph neural network (GNN) and temporal transformer for examiner behaviour discovery, followed by a second-phase GNN–LLM architecture for behaviour-grounded automated and collaborative marking.

The project is intended to remain one coherent research programme:

1. **Phase 1 — Behaviour discovery:** learn dynamic examiner behaviour representations from document structure, semantic segments, eye gaze, saccades, scrolling, return patterns, and behavioural loops.
2. **Phase 2 — Behaviour transfer and alignment:** reuse the learned graph encoder and behaviour space to guide evidence retrieval and LLM-based marking for new student responses.

The implementation should be staged rather than trained as one unconstrained end-to-end system from the outset. However, the graph schema, graph encoder, behaviour representations, and retrieval-policy interface should be designed from the beginning so that Phase 1 can continue directly into Phase 2.

---

# 2. Dataset and experimental structure

The current dataset contains:

- **25 participants**;
- **30 trials per participant**;
- approximately **750 participant–trial marking episodes**;
- the same 30 trials viewed by every participant;
- randomised trial order for each participant.

The 30 trials contain one student response per question and are distributed as follows:

| Question type | Number of trials |
|---|---:|
| Fill in the Blank (FiB) | 4 |
| One Word (OW) | 4 |
| Short Response (SR) | 6 |
| Long Response (LR) | 7 |
| Level of Response (LoR) | 9 |

The assessment materials available in the current project are:

- question text;
- student response;
- mark scheme;
- commentary;
- level descriptors;
- star charts where applicable.

Six of the 30 trials are **star-chart eligible**, but star-chart presence is randomised per participant: each participant sees exactly 3 of the 6 eligible trials with the star chart on, and which 3 varies across participants. Consequences:

- trial graphs for eligible trials exist in two variants, keyed by `(trial_id, star_condition)` — up to 36 graphs in total;
- `star_on` variants must be built as **base graph + star overlay** (star nodes and their edges added onto the frozen non-star base), never as two independent builds, with a regression check that the non-star subgraph is identical across variants;
- a participant × trial **condition assignment table** is a required, validated input; episodes join through it to load the correct graph variant;
- the randomisation provides a **within-question natural experiment**: the same question can be compared between star_on and star_off participants (see §12.7).

Additional behavioural and validation data include:

- observational reports;
- retrospective think-aloud (RTA) thematic analysis;
- RTA synchronised to the trial video;
- an existing two-level hidden Markov model (HMM) analysis, with micro fixation states and macro marking phases (comparison deferred pending a rerun on document-space gaze with star charts; see §12.4);
- per-trial confidence;
- per-trial difficulty;
- per-trial mental-effort ratings;
- time on task;
- final mark awarded by each participant;
- variation in marks across participants.

The effective independent units are the 25 participants and 30 trial graphs, not the total number of fixation events. Grouped participant-held-out cross-validation is the sole primary protocol for training, tuning, and all ablations. Leave-one-question-out analysis is retained but **demoted to a single post-hoc robustness analysis** on the final frozen configuration (mirroring the post-hoc analysis workflow used in the HMM study): it involves retraining with each question held out, but is run once, never inside the tuning loop and never per ablation. Question-type comparisons—particularly for FiB and OW, with four trials each—are framed as descriptive patterns within this dataset rather than strong question-type generalisation claims.

---

# 3. Overall architecture

## 3.1 Shared foundation

The project should use one reusable graph representation across both phases.

```text
Assessment materials and response segments
                ↓
      Shared document graph
                ↓
          Shared GNN encoder
                ↓
 Contextualised graph-node representations
          ↙                       ↘
 Phase 1 gaze transformer      Phase 2 retrieval heads
          ↓                       ↓
 Dynamic behaviour states      Behaviour-conditioned retrieval
          ↓                       ↓
 Behaviour prototypes          LLM marking and verification
```

The GNN represents the structured assessment environment. The temporal transformer represents how an individual examiner moves through that environment.

The graph describes **available or candidate relationships** in the document. It does not assert that a participant used those relationships cognitively. Participant-specific evidence comes from the actual gaze sequence and the order in which graph nodes were viewed.

The programme should also distinguish three connected but separate constructs:

```text
Behaviour representation
        ↓
Graph navigation policy
        ↓
LLM marking policy
```

- **Behaviour representation** describes the examiner's evolving latent state.
- **Graph navigation policy** describes which node types, relations, or candidate nodes that state tends to seek next.
- **LLM marking policy** describes how retrieved evidence is used to update or justify a marking decision.

This distinction prevents human attention, graph navigation, and marking correctness from being treated as the same target.


---

# 4. Phase 1 graph representation

## 4.1 Node types

The initial graph should use:

### Semantic-segment nodes

Examples include:

- question sentences or clauses;
- student-response sentences or clauses;
- mark-scheme bullet points;
- commentary segments;
- level descriptors;
- star-chart phrases or items;
- other annotated semantic segments.

### Abstract structural nodes

The graph should also contain abstract nodes for the main document regions:

- question;
- student response;
- mark scheme;
- commentary;
- level descriptors;
- star chart;
- other interface or UI region where needed.

These structural nodes do not require bounding boxes. They provide hierarchy and context through `BELONGS_TO` edges.

## 4.2 Canonical panel labels

Each semantic segment should receive one manually assigned canonical panel label, for example:

```text
question
response
mark_scheme
commentary
star_chart
ui
```

**Rule:** the canonical panel is a spatially grounded document region; semantic function within a region is carried by segment type/role, not by the panel label. Level descriptors are rendered inside the mark scheme panel in these materials, so they take panel `mark_scheme` with segment role `level_descriptor` (there is no separate `level_descriptor` panel). The next-panel prediction classes, empty-space categories, and loop templates all inherit this rule.

This label should be stored directly in the annotation metadata, for example:

```json
{
  "segment_id": "ann_segment_014",
  "panel_label": "response"
}
```

The canonical labels do not need to use exactly the same names as the Tobii AOIs.

A separate mapping can translate Tobii AOI names into the same canonical categories:

```text
Tobii "Student Answer Box" → response
Tobii "Mark Scheme AOI"    → mark_scheme
Tobii "Commentary Panel"   → commentary
```

The manually assigned segment label should be treated as the authoritative static graph membership. Tobii AOI hits can be used as a quality-control check.

A script should flag major disagreements such as:

```text
manual segment label: response
Tobii fixation hits:
- response: 35%
- mark_scheme: 65%
```

This may indicate an annotation, coordinate, scroll-alignment, or AOI-mapping problem.

## 4.3 Node attributes

Option B—semantic segments plus structural nodes—still allows rich node attributes.

Each semantic node may contain:

### Text features

- embedding of `corrected_text`;
- optional embeddings of command-word text;
- optional embeddings of domain-specific terms;
- OCR confidence where useful.

### Categorical features

- segment type;
- segment role;
- canonical panel label;
- level band;
- question type;
- mark-point presence;
- star-chart presence.

### Boolean features

- contains a command word;
- contains domain-specific language;
- is a bullet point;
- is a level descriptor;
- is a mark-scheme point;
- is commentary;
- is a star-chart segment;
- requires calculation;
- contains a data reference;
- contains an allow instruction;
- contains a reject instruction;
- contains a comparison.

### Formatting features

- bold text present;
- italic text present;
- amount or proportion of formatted text;
- indentation or bullet status where available.

### Geometry features

- normalised x and y centre;
- width and height;
- document-space location;
- number of constituent boxes;
- number of lines;
- segment order.

Command words and domain terms do not currently have their own boxes. They should therefore remain attributes of the larger sentence or clause node rather than becoming separate lexical nodes.

## 4.4 Text encoder specification and versioning

The text encoder must be selected and frozen before the Phase 1 graph corpus is generated. The choice should not remain an implicit implementation detail because it determines both the initial node representations and the Phase 1 cosine-similarity edges.

Before Phase 1B, a small domain check should compare a limited set of established sentence encoders on manually reviewed assessment-text pairs, including:

- student wording versus mark-scheme wording;
- related and unrelated response–criterion pairs;
- commentary paraphrases;
- short command-word and level-descriptor cases.

The selected encoder should then be recorded as `Text Encoder v1`, including:

- exact model identifier and revision;
- text preprocessing and pooling method;
- output dimension;
- whether it is frozen;
- similarity normalisation;
- graph-generation threshold and top-\(k\) settings.

For the first Phase 1 model, the same frozen sentence encoder should be used for node text features and cosine-based `SEMANTIC_CANDIDATE` generation. Phase 2 response–criterion compatibility should not rely on this cosine score alone and should introduce an entailment or cross-encoder reranker.

---

# 5. Automatically generated graph edges

The initial graph should use a balanced, sparse set of edge types that can be constructed automatically from the available data.

## 5.1 `NEXT_SEGMENT` and `PREVIOUS_SEGMENT`

These edges represent reading order.

The parser should:

1. group segments by canonical panel;
2. use `segment_order` within each panel;
3. check the order against geometry;
4. add forward and reverse edges.

```text
segment_08 → NEXT_SEGMENT → segment_09
segment_09 → PREVIOUS_SEGMENT → segment_08
```

This avoids treating the final question segment and the first response segment as one continuous paragraph.

## 5.2 `BELONGS_TO`

Each semantic segment should connect to its abstract structural node.

```text
response_segment_04 → BELONGS_TO → response_panel
mark_scheme_02      → BELONGS_TO → mark_scheme_panel
```

These edges are generated from the manually assigned canonical panel label.

## 5.3 `SPATIAL_NEIGHBOUR`

Spatial edges should be generated only between semantic segments with valid document-space coordinates.

The initial implementation should connect each segment to a small number of nearby segments, preferably within the same panel.

Edge attributes may include:

- normalised spatial distance;
- same column;
- vertical versus horizontal displacement;
- overlap or alignment information.

The first version should use one general `SPATIAL_NEIGHBOUR` relation rather than many separate `ABOVE`, `BELOW`, `LEFT_OF`, and `RIGHT_OF` relations.

## 5.4 `SAME_MARK_POINT` — removed

An earlier draft derived a `SAME_MARK_POINT` edge between segments sharing a `mark_point_id`. This edge is **removed**: `mark_point_id` is a label identifying each bullet point within the mark scheme, not a linking identifier — no two segments ever legitimately share one, so the edge can never fire. `mark_point_id` is retained as **node metadata** (stable bullet identity, useful for interpretation and Phase 2 traceability), with no edge semantics. Response↔criterion connectivity in the graph therefore rests on `SEMANTIC_CANDIDATE` edges (and, in Phase 2, the NLI/cross-encoder evidence linking), which was already the primary channel.

## 5.5 `SAME_STAR` — removed

An earlier draft added a `SAME_STAR` edge between segments sharing a `star_id`. Corpus inspection confirmed the same never-fires pattern as `SAME_MARK_POINT`: each `star_id` identifies one star and appears on one segment, so the edge is **removed**. `star_id` is retained as node metadata (star identity for interpretation and traceability). Star↔response and star↔commentary connectivity comes from the allowed `SEMANTIC_CANDIDATE` panel pairs; star nodes otherwise participate in the graph through `BELONGS_TO` and spatial edges like any segment.

This relation applies only to the three star-chart trials and should simply be absent from other graphs.

## 5.6 `SEMANTIC_CANDIDATE`

Semantic-candidate edges are generated using cosine similarity between segment text embeddings. With `SAME_MARK_POINT` removed (§5.4), these are the **sole structural channel for response↔mark-scheme connectivity**, and the HMM findings show that bullet-by-bullet matching of mark-scheme points against response segments is a core observed examiner behaviour. Edge construction is therefore **panel-pair-aware**, prioritising the response↔mark-scheme pair so that this behaviour is always expressible in the graph:

- consider cross-panel pairs only; ignore same-panel or same-overarching-AOI pairs;
- **response ↔ mark scheme (prioritised):** retain top \(k = 3\) candidates per node, with a **coverage floor** — every mark-scheme bullet (`mark_scheme_point` segment) receives at least its single best response-segment edge even if that similarity falls below the general threshold (flagged with a `below_threshold` edge attribute). Rationale: examiners check every bullet against the response; if a bullet has no edge, that checking behaviour cannot be captured as a relation-level transition, only as a panel-level one;
- **all other allowed pairs:** retain top \(k = 2\) with the conservative minimum similarity threshold; nodes may receive no semantic edge when all similarities are weak.

Allowed panel pairs:

```text
question ↔ response
response ↔ mark scheme      (prioritised: k=3 + per-bullet coverage floor)
response ↔ commentary
response ↔ star chart
mark scheme ↔ commentary
```

Edge attributes: cosine similarity, similarity rank within the pair, panel pair, and the `below_threshold` flag. Per-pair \(k\) values and thresholds are hyperparameters; density should be monitored in graph diagnostics, since the coverage floor guarantees recall for the prioritised pair while the GNN's attention supplies precision (a semantic-candidate edge remains a candidate, not an asserted relationship).

A semantic-candidate edge is not a definitive assessment relationship. It means:

> These two segments are plausible semantic neighbours that the GNN may learn to use or ignore.

## 5.7 Edges not included initially

The following should remain node attributes or later ablations rather than initial edge types:

- same level band;
- same segment type;
- contains the same command word;
- contains the same domain term;
- co-visible with.

Adding these immediately could create large, weakly informative clusters.

---

# 6. Graph parsing pipeline

For each trial, the preprocessing script should:

1. load all semantic segments;
2. validate unique segment IDs;
3. attach the canonical panel label;
4. encode corrected text;
5. encode categorical and boolean metadata;
6. calculate normalised geometry;
7. create semantic-segment nodes;
8. create abstract structural nodes;
9. generate reading-order edges;
10. generate `BELONGS_TO` edges;
11. generate sparse spatial edges;
14. calculate cross-panel cosine similarities;
15. retain conservative top-\(k\) `SEMANTIC_CANDIDATE` edges;
16. store edge types and numerical edge attributes.

The graph should be stored as:

```text
node_features
edge_index
edge_type
edge_attributes
node_id_lookup
```

There should be one graph instance per `(trial_id, star_condition)` pair — one variant for non-eligible trials, two for the six star-eligible trials (up to 36 graphs) — and one shared GNN with the same parameters across all graphs.

**Star-condition variant construction.** For eligible trials the pipeline builds the base (non-star) graph once, then produces the `star_on` variant by overlaying star-chart nodes and their edges (`BELONGS_TO`, spatial, semantic) on top of the frozen base. A regression check must assert that the non-star subgraph of `star_on` is identical to `star_off` in nodes, features, edges, and attributes, so variants cannot silently drift.

---

# 7. Initial GNN architecture

The first graph encoder should be a two-layer edge-aware graph attention network.

It should include:

- relation-type embeddings;
- numerical edge features;
- learned attention weights for individual candidate edges;
- residual connections;
- edge dropout;
- preservation of original node features alongside GNN context.

Conceptually:

```text
original node representation
        +
messages from candidate neighbours
        ↓
contextualised graph representation
```

The graph parser determines which candidate edges exist. The GNN learns:

- which relation types are useful;
- how much information should travel over each relation;
- which individual candidate edges should receive more or less weight;
- how neighbouring information should modify a node representation.

The final segment representation should preserve both:

```text
x_v = original segment representation
h_v = graph-contextualised representation
```

A later transformer token can therefore distinguish what the segment itself contains from the broader graph context available around it.

---

# 8. Eye-gaze and temporal input

## 8.1 Primary temporal unit: individual fixations

The first implementation should use individual fixations.

Each fixation token should include:

- participant ID;
- trial ID;
- fixation ID;
- timestamp;
- duration;
- assigned semantic segment or empty-space category;
- canonical panel;
- gaze-assignment confidence;
- relative trial time;
- visit count for the segment;
- time since the previous visit;
- return indicator;
- scroll features (see below): direction, displacement since previous fixation, instantaneous velocity, time since scroll onset and offset, during-active-scroll flag, normalised viewport position in the document, and gaze y-position within the viewport;
- previous saccade amplitude;
- previous saccade direction;
- other reliable previous-saccade features.

The previous saccade describes how the current fixation was reached and should be concatenated with the current fixation features.

### Scroll as a contextual signal

Scroll events are treated as **contextual signals of viewport movement, not direct indicators of navigation**. A large or rapid scroll accompanied by a change of viewed segment or panel may indicate active navigation, but some examiners scroll continuously while reading to keep text at a comfortable position on screen — for them, scrolling *is* reading. No fixed interpretation is imposed. Instead, the scroll features above are provided alongside gaze transitions, segment and panel changes, and reading progression, and the transformer learns whether a given scroll pattern is more consistent with navigation, continuous reading, or another behavioural mode, accommodating individual differences in scrolling style. Gaze y-position within the viewport is the cleanest separator of styles: a comfort-scroller holds viewport-relative gaze height nearly constant while document position advances, whereas a jump-navigator's varies sharply. Scroll style thereby enters the prototype fingerprints, so distinct scrolling modes can emerge from discovery rather than being asserted.

Note that gaze coordinates are already scroll-corrected into document space (the offset computed from scroll data has been applied and validated upstream); the scroll features here describe viewport dynamics as behaviour, not as a pending coordinate correction.

Three commitments follow:

1. **Input-only.** Scroll features are never prediction targets — they are context for behaviour, not navigation ground truth.
2. **Scroll-feature dropout.** During Phase 1 training, scroll features are randomly masked as an augmentation so the behaviour encoder learns to function without them. Phase 2 agent trajectories have no viewport and therefore no scroll; zeroed scroll features at Phase 2 must be in-distribution rather than novel. Comparing evaluation with scroll present versus masked also quantifies how much the representation depends on scrolling.
3. **Participant-style caution.** Scrolling style is strongly participant-idiosyncratic, making scroll features a plausible channel for participant identity in the embeddings; §12.5 therefore probes participant identity alongside trial identity.

### Gaze→segment assignment policy

Fixation-to-segment assignment must follow one explicit, configuration-driven policy applied identically everywhere, rather than being trusted from upstream exports. All matching occurs in document space (post scroll-correction) between the fixation point and segment bounding boxes:

1. each bounding box carries a dilation margin \(\varepsilon\) (default derived from tracker precision, ≈0.5° visual angle converted to document pixels, with the derivation documented);
2. a point strictly inside exactly one box is assigned to that segment with confidence 1.0 in the interior, decaying toward the edge zone;
3. a point inside multiple overlapping boxes, or within \(\varepsilon\) of two or more boxes, is assigned to the box with the smallest centre-weighted distance, flagged as ambiguous, with the runner-up segment recorded and confidence reduced by the best-versus-runner-up margin;
4. a point outside all boxes but within \(\varepsilon\) of one is assigned to the nearest segment with confidence decaying linearly to zero at \(\varepsilon\);
5. a point beyond \(\varepsilon\) of every box becomes a panel-specific empty-space event or `outside_document`.

Gaze-assignment confidence is therefore a deterministic function of geometry (interior depth, distance to edge, best-versus-runner-up margin). The policy is re-run with \(\varepsilon\) scaled ×0.5 and ×1.5, and the percentage of fixations whose assignment changes is reported — this quantifies how much data lies in the edge zone and feeds the sensitivity analysis in §12.5.

### Pre-modelling visual sanity gate

Before any model training, an interactive visual checker must render the document-space image with all segment boxes drawn, replay each fixation sequence with a time control, highlight the assigned segment for the current fixation together with its confidence and ambiguity flag, mark edge-zone fixations visibly, and render empty-space fixations distinctly. A stratified sample — every participant on several trials, star_on episodes covering all six eligible trials, and every episode flagged by data-validation disagreement checks — must be manually reviewed and signed off before deep-learning development proceeds. Any systematic misalignment found triggers an upstream fix and re-review.

## 8.2 Alternative temporal representations

The implementation plan should retain two alternative representations for comparison:

### Segment visits

Merge consecutive fixations on the same semantic segment into one visit token.

Possible aggregated features include:

- total duration;
- number of constituent fixations;
- mean or maximum previous-saccade amplitude;
- entry and exit time.

### Fixed or sliding temporal windows

Aggregate gaze and graph information over fixed time windows.

These alternatives should be treated as model comparisons or ablations rather than replacing the initial fixation-based approach.

## 8.3 Empty-space gaze

Empty-space gaze should not be discarded.

The initial representation should use panel-specific empty-space nodes or token categories:

```text
question_background
response_background
mark_scheme_background
commentary_background
star_chart_background
outside_document
```

The amount of unassigned gaze should first be calculated.

Ablation comparisons should include:

1. panel-specific empty-space categories;
2. one generic `NO_SEGMENT` category;
3. removal of empty-space events.

---

# 9. How the GNN and gaze transformer connect

For each trial, the GNN processes the graph and produces one contextualised vector per node.

```text
Trial graph
    ↓
Shared GNN
    ↓
Contextualised node table
```

For each participant fixation, the assigned segment ID is used to retrieve the corresponding graph vector.

```text
fixation on response_segment_04
        ↓
retrieve GNN vector for response_segment_04
```

The transformer token should combine:

- original segment representation;
- graph-contextualised representation;
- fixation features;
- previous-saccade features;
- timing;
- scrolling;
- visit and return history.

```text
transformer token =
original node signal
+ GNN context
+ fixation signal
+ previous-saccade signal
+ timing and history
```

The raw gaze data does not initially pass through the GNN. It selects graph nodes and is fused with their representations before entering the transformer.

The graph may also contribute pairwise relation biases between fixation tokens when the two viewed nodes are connected.

---

# 10. Loop-aware temporal transformer

The Phase 1 looping architecture remains a central component.

## 10.1 Primary causal model

The principal model should use a causal transformer so that the dynamic behavioural embedding at time \(t\) depends only on the current and previous fixations.

```text
y_t = behaviour representation based on events 1 ... t
```

This preserves the forward temporal process of examiner behaviour and provides the appropriate state representation for the later Phase 2 controller, which must select its next action without access to future events.

The transformer should retain:

- same-segment return signals;
- previous-visit information;
- time since previous visit;
- graph relationships between viewed nodes;
- loop-closure features;
- relative temporal distance;
- learned loop-aware attention bias.

Examples include:

```text
response → mark scheme → response
question → response → question
response → commentary → response
mark scheme → commentary → mark scheme
```

The model should initially describe loops in observable terms rather than assigning cognitive labels such as verification or uncertainty.

### Loop operationalisation: the loop event detector

Loops are not assumed labels; they are made concrete by a deterministic, configuration-driven detector run over each fixation sequence at dataset-build time. It identifies:

- **segment returns** — a fixation on a segment previously fixated in the episode, recording the gap in events and milliseconds, with a threshold separating short-loop returns from long-range revisits;
- **template loops** — panel-level `A→B→A` patterns completed within an event/time window, implemented as small state machines over canonical panel sequences, with an initial template set of response→mark_scheme→response, response→mark_scheme[level_descriptor]→response (a panel template refined by segment role, since level descriptors live inside the mark scheme panel), response→commentary→response, mark_scheme→response→mark_scheme, question→response→question, and star-chart variants (e.g. response→star_chart→response) for star_on episodes; templates may therefore be defined over panels optionally refined by segment role; overlapping loops are all recorded.

Each fixation receives: a return indicator and gap features; a loop role (origin, pivot, closure, or none); a multi-hot template identifier (a fixation may close one loop and open another); and the token index of the loop origin.

These annotations are the **single source** for every loop-related mechanism: the fixation-token features in §8.1; the loop-aware attention bias, which uses the origin index to bias attention from closure tokens back toward origin and same-segment predecessor tokens; the loop-type diagnostic probe in §11.4, whose labels are the template identifiers (making the detector a hard dependency of the diagnostic gate); and the deferred loop losses, should diagnostics trigger them. Templates with too few corpus-wide occurrences (fewer than roughly fifty) are dropped from the diagnostic probe and the decision documented.

The graph represents the possible relationship. The gaze sequence shows that the participant actually moved between the nodes. The transformer learns whether the order and timing form a meaningful behavioural pattern.

The loop architecture is part of the initial model regardless of whether return or loop prediction is included as a separate loss. Return history, loop-closure features, and loop-aware attention bias should therefore be implemented from the beginning; dedicated loop losses are added only if diagnostics show that the learned representation is failing to preserve loop structure.

## 10.2 Secondary bidirectional comparison

A bidirectional transformer may be trained as an offline comparison for retrospective phase segmentation.

Because it can use future events, it may identify cleaner phase boundaries or reinterpret an earlier fixation in light of what happens next. However, its embedding at time \(t\) would not represent what could have been inferred at that moment and would not be suitable as the main Phase 2 controller state.

The bidirectional model should therefore be treated as:

- a secondary phase-segmentation comparison;
- a possible retrospective teacher or boundary-refinement model;
- not a replacement for the primary causal behaviour encoder.

Ordinary masked-event reconstruction may be used with this optional bidirectional model, but not as a core objective of the causal model.

---

# 11. Phase 1 training objectives

The primary output is a dynamic behavioural embedding for every fixation.

The initial model should use a deliberately small set of causal-compatible objectives. This reduces the interacting hyperparameter surface and keeps the first scientific test focused on transferable graph navigation.

## 11.1 Next-panel prediction

Predict the overarching panel likely to be viewed next:

```text
question
response
mark_scheme
commentary
star_chart
empty_space
```

This is highly transferable because it does not depend on a particular segment or question identity.

## 11.2 Multi-label next-relation prediction

Predict every graph relationship that holds between the current viewed node and the next viewed node.

Possible labels include:

```text
same segment
next segment
previous segment
spatial neighbour
same mark point
same star
semantic candidate
same structural region
no direct graph relation
```

Two nodes may have several valid relations simultaneously. For example, they may be both `SPATIAL_NEIGHBOUR` and `SEMANTIC_CANDIDATE`. The target should therefore be a multi-hot vector with independent sigmoid outputs and a binary cross-entropy-style loss, rather than a softmax with an arbitrary fixed priority order.

`NO_DIRECT_RELATION` should be positive only when none of the other defined graph relations apply.

## 11.3 Candidate next-node ranking

Given the current causal behavioural state and the nodes available in the current trial graph, rank plausible next nodes.

The actual next viewed node is the positive candidate. Negatives should be sampled from the same trial graph and should include both easy and hard negatives, such as semantically similar but unvisited nodes.

The ranking function should use:

- the current dynamic behavioural embedding;
- candidate node features;
- candidate GNN context;
- graph relations from the current node;
- whether the candidate has already been visited;
- temporal and loop history.

The purpose is to learn transferable node selection within a new graph, not to memorise fixed node-number transitions.

## 11.4 Diagnostics and deferred objectives

The following outputs may be calculated as diagnostics or added later, but they are not part of the initial three-loss model:

- next-node-type prediction;
- exact next-node classification;
- return prediction;
- loop-type prediction;
- contrastive discrimination of real and corrupted sequences.

After the first model is trained, diagnostics should test whether the dynamic embeddings already retain:

- return versus first-visit structure;
- common loop types;
- node-type transitions;
- meaningful local sequence order.

The diagnostic gate is **pre-registered with concrete pass/fail criteria fixed before results are inspected**, run on frozen embeddings:

- **D1 (return structure):** a linear probe predicting return-to-current-segment within \(N\) events must exceed the AUC of an identical probe trained on raw token features alone by a pre-committed margin (default +0.05);
- **D2 (loop types):** a multinomial probe over the detected loop templates (labels supplied by the loop event detector in §10.1) must exceed a within-episode label-shuffled baseline by a pre-committed macro-F1 margin;
- **D3 (local order):** a probe distinguishing true from locally shuffled embedding subsequences.

Return or loop prediction is added only if a gate fails, and the decision is documented. A contrastive objective should be introduced only if the three core objectives produce unstable or weakly separated prototypes. Pre-committing the thresholds converts a judgement call into a decision rule and closes the garden-of-forking-paths objection.

## 11.5 Initial objective structure

The committed initial losses are:

1. next-panel prediction;
2. multi-label next-relation prediction;
3. candidate next-node ranking.

This is the minimum set needed to learn a forward graph-navigation representation and create the Phase 1-to-Phase 2 retrieval bridge.

The causal model should not use ordinary bidirectional masked-event reconstruction as a core objective.

---

# 12. Behaviour representations, discovery, and validation

## 12.1 Continuous behavioural embeddings

The causal transformer should produce one continuous embedding per fixation:

```text
[y_1, y_2, ..., y_T]
```

This allows behaviour to be traced dynamically through the marking episode.

## 12.2 Soft behaviour prototypes

The continuous embeddings should later be clustered or represented using soft prototypes. The committed method is a **Gaussian mixture model** on (optionally PCA-reduced) fixation embeddings fitted on training folds: the number of prototypes is selected by BIC over a fixed range (approximately 4–12) and checked for stability across seeds and folds (pairwise adjusted mutual information of hard assignments above a threshold). Soft memberships are the GMM posteriors.

Example:

```text
prototype_1: 0.62
prototype_2: 0.27
prototype_3: 0.11
```

A hard Phase 1 behaviour can still be inferred using the highest-probability prototype.

A confidence threshold should allow uncertain states to be labelled as mixed or transitional rather than forcing a hard assignment.

## 12.3 Discovery before behaviour supervision

The initial training stage should not assume labels such as orientation, evidence search, verification, completeness checking, or integration.

The sequence should be:

1. train the causal GNN–Transformer using generic navigation and loop objectives;
2. produce dynamic fixation-level embeddings;
3. cluster or derive soft prototypes;
4. inspect representative sequences and graph trajectories;
5. validate prototypes using RTA, observational reports, and other measures;
6. assign descriptive names only where justified;
7. create soft prototype pseudo-labels for later supervised heads.

A manually reviewed subset should be used to check interpretation and calibrate prototype assignments, but exhaustive manual fixation labelling is not required.

### Prototype tracing workflow

Interpreting a discovered prototype follows a defined four-step path from cluster to behavioural meaning:

1. **Fingerprint (contrast statistics).** For each prototype versus all others, standardised mean differences over interpretable features only — panel and node-type occupancy, relation-type traversal rates, loop-template rates and loop-role mix, fixation duration, saccade amplitude, visit counts, relative trial-time position, and assignment confidence — ranked and visualised. This is automatic and model-free, answering what is statistically distinctive about the state.
2. **Exemplars.** The top contiguous subsequences by mean posterior per prototype, with enforced diversity (at least five participants and five trials each) so that no prototype is illustrated by a single examiner's idiosyncrasy; each exemplar links to its episode timeline.
3. **Document-space replay.** The visual gaze checker (§8.1) gains a prototype-colouring mode — fixations coloured by hard label with posterior as opacity over the actual document image — so any prototype can be watched on real scripts. This is the primary artefact for RTA cross-referencing and reviewer judgement.
4. **Outcome anchoring.** Episode-level prototype proportions related to confidence, difficulty, mental effort, time on task, mark awarded, and cross-participant mark variance, using mixed models with participant as a random effect.

## 12.4 Behaviour validation (HMM comparison deferred)

The comparison with the existing two-level HMM is **deferred**. The current HMM was trained on screen-space AOIs without document-space correction and without star-chart analysis, so its input space differs from the new pipeline's; a comparison now would confound model differences with measurement differences. The hierarchical HMM will be rerun on document-space gaze with star-chart data included, after which the comparison becomes a clean like-for-like convergent-validity analysis.

**Phase 1 obligation (cheap, mandatory):** export per-fixation prototype posteriors and hard labels, keyed by participant, trial, fixation, and timestamps. This export is the entire interface the future comparison needs; no comparison code is written in Phase 1, and no retraining is required when the comparison runs.

**Deferred comparison protocol (recorded now so it is not lost):** aggregate prototype posteriors into the HMM's native analysis windows (window-level as the primary basis; fixation-level projection as a secondary view only); compare assignment agreement (adjusted mutual information / V-measure), transition structure (overlap-matched to avoid asymmetric self-transition inflation from overlapping windows, per-row Jensen–Shannon divergence), and macro-phase boundary agreement (boundary F1 with a window-level tolerance). If feasible, the rerun HMM should also emit a fixation-level state path alongside its windows, which sidesteps the overlap issue entirely.

**Primary Phase 1 validation** therefore rests on:

1. **Quantified RTA alignment** — for a stratified sample of episodes with synchronised RTA, map RTA thematic codes onto the fixation timeline and report prototype occupancy per RTA theme as a contingency analysis (Cramér's V against a permutation baseline that shuffles prototype labels within episodes);
2. **Structured reviewer judgement** — manual review of the prototype interpretation pack by at least one assessment-domain reviewer, recording coherent / mixed / uninterpretable judgements per prototype;
3. **Outcome anchoring** — associations between episode-level prototype proportions and confidence, difficulty, mental effort, time on task, mark, and cross-participant mark variance.

Confidence, difficulty, and mental effort should remain separate constructs. They may validate related patterns but should not be collapsed into a single uncertainty label.

## 12.5 Guarding against trial and question-type domination

Participant-held-out evaluation alone is insufficient because all participants view the same 30 trial graphs.

The evaluation should therefore include:

- grouped participant-held-out evaluation as the sole primary protocol;
- leave-one-question-out analysis as a single post-hoc robustness analysis on the final configuration (see §2);
- trial-identity probes trained on the behavioural embeddings;
- participant-identity probes (scroll style, pacing, and other idiosyncrasies make participant leakage plausible; measure and report with the same stance as trial identity — individual behavioural style is partly genuine behaviour, so it is not reflexively removed);
- question-type probes for FiB, OW, SR, LR, and LoR;
- evidence that the same prototypes occur across multiple questions and participants;
- prototype-distribution comparisons across trials;
- stability across random seeds;
- sensitivity to gaze-assignment noise.

Question-type differences should be reported descriptively. The small numbers of FiB and OW trials do not support strong claims of question-type generalisation.

Because consecutive fixation embeddings are strongly autocorrelated, all prevalence and outcome statistics first aggregate to the episode level (proportion of episode time per prototype) and model participant as a random effect; fixation-level points are never treated as independent observations.

Some trial information is expected and may be genuinely necessary because the available resources and valid behaviours differ between trial structures. The initial commitment is therefore to measure and report probe accuracy, not to force all trial context out of the representation.

Context-removal methods should be considered only if probe accuracy is egregiously high and the embedding space is visibly organised primarily by trial identity rather than recurring behaviour. If that occurs, the first mitigation should be conservative context dropout or later context fusion. Aggressive adversarial or orthogonality-based removal is not part of the planned initial model because it may remove genuine task structure.

## 12.6 Primary versus secondary temporal interpretation

The causal model provides the main dynamic behaviour representation and supports forward transfer to Phase 2.

The optional bidirectional model may be used to test whether future context improves retrospective phase boundaries. Results from that comparison should be reported separately so retrospective segmentation is not confused with online behavioural inference.

---

## 12.7 Star-chart natural experiment

Because star-chart presence is randomised within the six eligible trials (§2), behaviour on the same question can be compared between star_on and star_off participants. This pre-registered descriptive analysis examines prototype-proportion differences, star-template loop rates, and next-panel transition patterns at the episode level with participant as a random effect. It is the cleanest within-question behavioural contrast the design offers and should be reported alongside the main discovery results.

---

# 13. Preparing Phase 1 for Phase 2

Phase 1 should not produce only descriptive clusters. It should learn generic, transferable graph-navigation signals before named behaviour prototypes exist.

## 13.1 Generic navigation learning during Phase 1

During initial causal training, the model should learn:

- next-panel prediction;
- multi-label next-relation prediction;
- candidate next-node ranking.

These three objectives teach the model how gaze moves through graph structure without assuming behaviour names in advance. Return and loop structure remains encoded in the model inputs and attention bias; additional return or loop losses are introduced only if the initial embeddings fail the loop diagnostics.

The model should learn transferable rules such as:

> From candidate response evidence, the next fixation may move through a same-mark-point or semantic-candidate relation towards the mark scheme.

It should not merely learn:

> In Trial 12, move from node 17 to node 31.

## 13.2 Prototype formation and pseudo-labelling

After generic training:

1. generate dynamic embeddings;
2. derive and validate soft behaviour prototypes;
3. assign descriptive labels only where supported;
4. use the validated soft memberships as pseudo-labels.

Example:

```text
verification-like: 0.68
evidence-search-like: 0.24
orientation-like: 0.08
```

Hard labels may be inferred for analysis when one prototype exceeds a defined confidence threshold. Mixed states should remain soft or transitional.

## 13.3 Freeze a stable Phase 1 interface

Before Phase 2 experiments begin, save and freeze a stable reference version:

```text
Graph schema v1
Graph encoder v1
Causal behaviour encoder v1
Prototype Set v1
```

This provides a fixed interface between the phases and prevents later LLM or marking losses from silently changing the meaning of the discovered behaviours.

Phase 2 should initially freeze these components and train new heads or lightweight adapters. Any later joint fine-tuning should preserve the Phase 1 representation through auxiliary losses, adapters, or distillation against the frozen reference model.

## 13.4 Supervised bridge after behaviour discovery

The validated prototype memberships can supervise:

- a behaviour classifier;
- a behaviour-transition model;
- a controller that predicts the next behaviour from the graph and current marking state;
- behaviour-conditioned candidate-node ranking;
- behaviour-conditioned relation selection.

This creates the sequence:

```text
Causal self-supervised / weakly supervised training
        ↓
Dynamic embeddings
        ↓
Clustering and RTA validation
        ↓
Soft behaviour pseudo-labels
        ↓
Frozen Prototype Set v1
        ↓
Supervised Phase 2 controller and retrieval heads
```

---

# 14. Phase 2: behaviour-grounded GNN–LLM marking

## 14.1 New-response graph

For Phase 2, a new student response should be segmented using the same semantic schema.

The new response nodes are added to the fixed assessment graph containing:

- question;
- mark scheme;
- commentary;
- level descriptors;
- star chart where relevant.

Automatic structural candidate edges are generated using the same rules as Phase 1. For response–criterion links, Phase 2 should use a two-stage process:

1. the frozen sentence encoder and cosine similarity generate a broad, high-recall candidate set;
2. an NLI-style entailment model or response–criterion cross-encoder reranks candidates and provides the main compatibility score.

This is necessary because valid student evidence may be worded very differently from the formal mark scheme. Cosine similarity remains useful for candidate generation but should not be treated as the final criterion-linking decision.

## 14.2 Initial Phase 2 targets

The recommended order is:

```text
response–criterion evidence linking
→ human-salience ranking
→ behaviour-conditioned retrieval
→ one-shot grounded LLM marking
→ agentic marking
```

### Response–criterion evidence linking

Estimate which response segments may support which mark-scheme points or descriptors.

### Human-salience ranking

Estimate which graph nodes a human examiner is likely to prioritise, based on the gaze-informed representation.

This predicts likely human attention, not objective correctness.

The following signals should remain separate:

- human salience;
- response–criterion compatibility;
- evidence sufficiency;
- uncertainty.

Complementary or divergent value should initially be an adjudicated evaluation analysis rather than a trained head. A divergent retrieval is valuable only when expert review confirms that it identifies valid evidence, a defensible alternative interpretation, or an appropriate escalation.

### Supervision sources for Phase 2 heads

| Head | Primary supervision source | Important caveat |
|---|---|---|
| Human salience | Fixation probability, duration, revisits, and next-node choices | Predicts likely human prioritisation, not correctness |
| Behaviour similarity | Frozen soft memberships from Prototype Set v1 | Depends on successful Phase 1 validation |
| Response–criterion compatibility | semantic weak labels, reviewed response–criterion links, and later human evidence annotations (`mark_point_id` provides criterion identity for traceability) | Gaze alone cannot establish that a criterion link is correct |
| Evidence sufficiency | Criterion-level evidence annotations, independently marked new responses, and mark-scheme requirements | Final marks alone may not identify which evidence was sufficient |
| Uncertainty or ambiguity | Examiner disagreement, escalation or adjudication labels where available | Confidence, difficulty, and mental effort are related but indirect signals |


### Complementary-value analysis

Complementary value should initially be evaluated through human adjudication rather than supervised prediction. The analysis should compare cases where:

- the human-aligned pathway did not retrieve a node;
- an AI-native or exploratory pathway retrieved it;
- an expert judged whether that evidence was valid, irrelevant, misleading, or sufficient to alter or escalate the mark.

A trained complementary-value head should be considered only after a sufficiently large adjudicated dataset exists.

## 14.3 Behaviour-conditioned retrieval

The behaviour state or prototype should influence which graph nodes are retrieved.

Examples:

- evidence-search behaviour prioritises candidate response evidence and criterion links;
- verification behaviour prioritises related mark-scheme points and commentary;
- completeness checking prioritises unsupported criteria and uninspected response areas;
- integration behaviour retrieves all supported criteria and relevant level descriptors.

## 14.4 LLM input

The LLM should receive source text and provenance rather than opaque graph vectors.

A retrieved package should contain:

- behaviour or retrieval mode;
- response evidence;
- criterion text;
- commentary;
- level descriptor;
- node IDs;
- graph path;
- compatibility, salience, sufficiency, and uncertainty scores.

## 14.5 Agentic loop

A later agentic version may use:

```text
graph processing
→ behaviour selection
→ graph retrieval
→ LLM marking-state update
→ behaviour selection
→ further retrieval
→ stopping or escalation
```

The LLM should update an explicit marking state containing:

- provisional mark;
- satisfied criteria;
- candidate evidence;
- missing evidence;
- uncertainty;
- alternative interpretation;
- next action.

---

# 15. Feasibility of a Phase 2 paper

The current Phase 1 dataset contains only one student response for each of the 30 questions. This limits the ability to demonstrate that response–criterion linking generalises across diverse responses during the initial gaze study.

Phase 2 is still worth testing, but its critical path is data acquisition rather than model architecture.

Collection, governance, ethics, independent marking, and evidence annotation for new responses should begin during Phase 1B–C rather than waiting for the behaviour-space freeze. At minimum, the Phase 2 dataset should aim to contain multiple independently human-marked responses per question and a smaller adjudicated subset with response–criterion evidence spans.

A defensible proof-of-concept paper could collect or obtain multiple new responses for the same 30 questions and test:

> Can the gaze-informed graph and behaviour policies transfer to unseen responses within known assessment contexts?

This would provide a valid within-question transfer evaluation.

The limitation should be stated clearly:

> The study would not yet demonstrate generalisation to entirely unseen questions and mark schemes.

Future work could test:

- unseen questions;
- unseen question types;
- unseen assessment materials;
- wider response distributions;
- alternative subjects.

This limitation does not prevent Phase 2 becoming a paper, particularly if the evaluation includes independently human-marked new responses, evidence-linking accuracy, grounding, uncertainty, and comparison against non-gaze retrieval baselines.

---

# 16. Development sequence

## Phase 1A — Data validation, canonical labels, and the gaze-alignment gate

- add canonical `panel_label` to each segment;
- map Tobii AOI names to canonical labels;
- validate manual labels against gaze hits;
- check IDs, geometry, segment order, and metadata consistency;
- validate the participant × trial star-condition table (exactly three star_on episodes per participant among the six eligible trials; star segments present only where the table says);
- calculate the proportion of empty-space gaze;
- implement the explicit gaze→segment assignment policy (§8.1) with edge tolerance, ambiguity handling, and geometry-derived confidence;
- run the pre-modelling visual sanity gate (§8.1) on a stratified sample and obtain manual sign-off before any model development.

## Phase 1B — Encoder selection and automatic graph parser

- run the small assessment-domain comparison for candidate sentence encoders;
- freeze and document `Text Encoder v1`;
- build semantic and abstract structural nodes;
- generate balanced automatic edges;
- expose semantic top-\(k\) and threshold as graph-version parameters;
- produce graph diagnostics and visualisations.

## Parallel Phase 2 data acquisition — begins during Phase 1B–C

- identify or collect multiple new responses for the same 30 questions;
- begin ethics, governance, and access processes;
- arrange independent human marking;
- define a criterion-level evidence-link annotation protocol;
- adjudicate a smaller subset for evidence sufficiency, ambiguity, and useful divergence;
- document response coverage across marks and response quality.

## Phase 1C — Compact GNN-only testing

- use a two-layer edge-aware GNN;
- verify node and edge parsing;
- inspect graph-attention weights;
- confirm that original and contextual representations remain distinguishable;
- use strong regularisation and multiple random seeds.

## Phase 1D — Causal fixation transformer

- use individual fixations first;
- include previous-saccade features;
- add causal masking;
- add graph relation and temporal biases;
- apply scroll-feature dropout augmentation (§8.1) so the encoder tolerates absent scroll signals;
- run the loop event detector (§10.1) at dataset-build time and implement explicit return, loop-closure, and loop-attention mechanisms from its annotations;
- train only the three initial objectives: next panel, multi-label next relation, and candidate next-node ranking.

## Phase 1E — Representation diagnostics and limited temporal comparison

- run the pre-registered diagnostic gate (§11.4) with thresholds fixed in advance;
- add return or loop prediction only if a gate fails, documenting the decision;
- compare individual fixations with merged segment visits as the primary temporal-input ablation;
- retain fixed windows and the bidirectional transformer as secondary analyses if resources permit.

## Phase 1F — Behaviour discovery and validation

- produce dynamic causal embeddings;
- derive soft prototypes (GMM, BIC-selected k, stability-checked; §12.2);
- infer hard behaviours where confidence is sufficient;
- run the prototype tracing workflow (§12.3): fingerprints, diverse exemplars, document-space replay, outcome anchoring;
- validate prototypes through quantified RTA alignment and structured reviewer judgement (§12.4);
- analyse confidence, difficulty, effort, time, and marking outcomes with episode-level aggregation and participant random effects;
- run the star-chart natural-experiment analysis (§12.7);
- export per-fixation prototype posteriors for the deferred HMM comparison (§12.4);
- run the single post-hoc leave-one-question-out robustness analysis on the final configuration;
- probe trial identity and question type;
- report question-type comparisons descriptively;
- check stability across seeds.

## Phase 1G — Freeze the behaviour interface

- manually review a representative subset;
- finalise descriptive names where justified;
- save `Graph Schema v1`;
- save `Text Encoder v1`;
- save `Graph Encoder v1`;
- save `Causal Behaviour Encoder v1`;
- save `Prototype Set v1`;
- generate soft prototype pseudo-labels for the development dataset.

## Phase 2A — Supervised navigation and evidence heads

- train behaviour classifier and transition heads using soft prototype pseudo-labels;
- train behaviour-conditioned candidate-node ranking;
- use cosine similarity only for broad response–criterion candidate generation;
- train or adapt an NLI/cross-encoder compatibility reranker using weak, reviewed, and newly annotated pairs;
- train human-salience ranking from gaze;
- train evidence-sufficiency and uncertainty heads only where direct supervision is adequate;
- keep complementary value as an adjudicated analysis;
- initially freeze the Phase 1 reference encoders.

## Phase 2B — New-response evidence linking

- add multiple new responses to known question graphs;
- generate high-recall candidate response–criterion edges;
- rerank them with the compatibility model;
- compare links and cited spans with human evidence judgements.

## Phase 2C — One-shot GNN–LLM

- retrieve one source-grounded subgraph;
- provide it to the LLM;
- generate mark, rationale, and uncertainty;
- compare with LLM-only, semantic-retrieval, and standard graph-retrieval baselines.

## Phase 2D — Behaviour-conditioned agentic marking

- introduce the behaviour controller;
- maintain an explicit marking state;
- allow repeated graph retrieval;
- evaluate human-aligned and AI-native retrieval modes;
- analyse hybrid or complementary retrieval through expert adjudication.

---

# 16b. Implementation instrumentation

Two cross-cutting engineering requirements accompany the development sequence (full specification in the separate Phase 1 implementation plan):

- **Per-run visual reporting.** Every training run emits a self-contained visual report covering training dynamics (per-loss train/validation curves across seeds, gradient behaviour, overfitting panels, per-label curves for rare relations), predictive performance (confusion matrices, per-label precision–recall, calibration, ranking metrics, all against explicit baseline floors), embedding and behaviour-space maps (linked UMAP/PCA colourings including trial identity and question type as the visual companion to the §12.5 probes), and episode-level interpretation views (prototype-coloured timelines, scanpath-on-graph, GNN attention). Test-set panels are gated behind a flag that is off during tuning.
- **Experiment tracking.** All runs log to a local experiment tracker (MLflow file store by default) tagged by milestone, ablation, fold, and seed, so the full ablation matrix is browsable; file-based run directories remain the source of truth, and tracker failures never abort a run.

---

# 17. Initial hyperparameters and focused ablations

## Initial hyperparameters

The initial model should minimise tuned choices. The main development parameters are:

- semantic top-\(k\), initially \(k = 2\);
- semantic cosine-similarity threshold;
- number of spatial neighbours;
- GNN hidden dimension;
- transformer hidden dimension;
- two GNN layers;
- a small number of transformer layers;
- edge dropout;
- temporal context length;
- hard-prototype confidence threshold;
- gaze-assignment edge tolerance \(\varepsilon\) (≈0.5° visual angle in document pixels, with ×0.5/×1.5 sensitivity re-runs);
- three initial loss weights, with a simple equal-weight baseline before tuning.

## Pre-registered core ablations

The initial paper should commit to a small core set:

1. **Transformer without GNN** — tests the value of graph contextualisation.
2. **GNN–Transformer without loop-aware attention bias** — tests the added value of the loop mechanism.
3. **Full GNN–Transformer** — the proposed model.
4. **Full model without `SEMANTIC_CANDIDATE` edges** — tests automatic semantic graph links.
5. **Full model without `SPATIAL_NEIGHBOUR` edges** — tests reading-geometry structure (promoted from the secondary list to replace the removed `SAME_MARK_POINT` ablation).
6. **Individual fixations versus merged segment visits** — tests temporal granularity.

Secondary analyses should be conducted only when diagnostics, available time, or reviewer concerns justify them. These may include empty-space variants, fixed windows, causal-versus-bidirectional comparison, or adapter fine-tuning in Phase 2.

Trial-identity and question-type probes are validation diagnostics, not ablations. The HMM comparison is deferred convergent validation (§12.4), not a model-selection condition.

---

# 18. Confirmed initial specification

```text
GRAPHS
- one graph per (trial, star_condition); star_on built as base + star overlay

NODES
- semantic segments
- abstract structural panel nodes

MANUAL STATIC LABEL
- canonical panel label for each semantic segment

NODE ATTRIBUTES
- text embedding
- segment type and role
- canonical panel
- level band
- keywords and command words
- domain terms
- selected boolean metadata
- formatting
- geometry
- segment order

AUTOMATIC EDGES
- NEXT_SEGMENT
- PREVIOUS_SEGMENT
- BELONGS_TO
- SPATIAL_NEIGHBOUR
- top-k cross-panel SEMANTIC_CANDIDATE

GAZE INPUT
- individual fixations first
- explicit gaze→segment assignment policy with edge tolerance and geometry-derived confidence
- pre-modelling visual sanity gate with manual sign-off
- previous-saccade features concatenated to current fixation
- loop event detector annotations (returns, loop roles, template identifiers)
- scroll, timing, return, and assignment-confidence features
- panel-specific empty-space categories

MODEL
- compact two-layer edge-aware GNN
- residual preservation of original node features
- causal fixation transformer as the primary model
- bidirectional transformer only as an offline comparison
- learned graph-relation, return, and loop biases

PHASE 1 TRAINING
- next-panel prediction
- multi-label next-relation prediction
- candidate next-node ranking
- return, loop, node-type, exact-node, and contrastive objectives deferred unless diagnostics justify them

PHASE 1 OUTPUT
- dynamic causal behavioural embedding per fixation
- soft behaviour prototypes
- optional hard behaviour labels above a confidence threshold
- generic node-type, panel, relation, and candidate-ranking policies
- frozen Graph Encoder v1, Behaviour Encoder v1, and Prototype Set v1

PHASE 2 SUPERVISED BRIDGE
- soft prototype pseudo-labels
- behaviour classifier and transition model
- behaviour-conditioned navigation policy
- cosine candidate generation plus NLI/cross-encoder criterion compatibility
- explicitly supervised evidence, salience, sufficiency, and uncertainty heads
- complementary value evaluated through adjudication rather than trained initially

PHASE 2 OUTPUT
- evidence linking
- human-salience ranking
- behaviour-conditioned retrieval
- source-grounded LLM mark, rationale, uncertainty, and escalation
```

---

# 19. Methodological safeguards

The first implementation should remain deliberately compact because the independent structure of the dataset is defined by 25 participants and 30 trial graphs rather than by the raw number of fixation events.

The study should therefore use:

- a shallow GNN and small transformer;
- three initial training losses rather than a large multi-objective system;
- strong regularisation and edge dropout;
- grouped participant splits;
- held-out-question analyses;
- multiple random seeds;
- explicit trial-identity and question-type probes reported as diagnostics and limitations;
- descriptive rather than generalisation claims for question-type comparisons;
- HMM and synchronised RTA as convergent validation;
- clear supervision documentation for every Phase 2 head;
- frozen and versioned Phase 1 reference models before Phase 2 evaluation;
- Phase 2 response acquisition running in parallel with Phase 1 development.

---

# 20. Central research contribution

The project should be framed as a connected behaviour-grounded architecture rather than simply a combination of a GNN, transformer, and LLM.

The central contribution is:

> I use examiner gaze to learn dynamic marking behaviours over a semantic assessment graph, preserve return and loop structures that reveal how examiners move between evidence and criteria, and transfer those behaviours into graph-based retrieval policies that guide an LLM during automated and collaborative marking.

This enables investigation of:

- how examiner behaviour changes dynamically over time;
- whether graph-aware loop modelling recovers or refines the microstates and macro phases identified by the two-level HMM;
- whether human gaze can guide evidence retrieval for unseen responses;
- whether an LLM benefits from behaviour-conditioned sequential retrieval;
- when automated marking should align with human behaviour;
- when controlled divergence can provide complementary evidence or uncertainty detection.
