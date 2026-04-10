# Label Evolution: Running Learnings

*Last updated: 2026-04-07*

---

## What the algorithm does (and doesn't do)

The evolution algorithm is a **label discovery tool**, not a label stabilisation tool. It takes paragraphs one at a time, asks a labeler model to assign existing labels and optionally propose taxonomy changes (CREATE/MERGE/SPLIT/RENAME/DELETE), and iterates. The output is a taxonomy that reflects what the labeler *kept noticing* in the data.

**Key limitation:** the taxonomy is path-dependent. The first ~20 steps seed the vocabulary, and the rest of the run builds on that. Two runs on identical data with the same labeler produce noticeably different taxonomies. This means the algorithm is good for exploration but not for producing a ground-truth, reproducible taxonomy.

**The max-labels cap matters a lot.** When the taxonomy is full (e.g. 20/20), CREATE is blocked. If you start with a full seed, nothing evolves — the algorithm can only MERGE/SPLIT/RENAME, which only fire if something is clearly broken. Fix: seed with fewer labels than the cap, or raise the cap (e.g. seed=20, cap=30).

---

## Labeler model comparison

### gpt-oss-20b (local LM Studio)
- High run-to-run variance: two runs on identical data produced completely different taxonomies
- Descriptions sometimes conflate distinct behaviors
- Parameter issues: needed `max_tokens` → `max_completion_tokens`, no custom `temperature`, `reasoning_effort` not supported → add these as auto-negotiated kwargs

### gpt-5.4-mini-2026-03-17 (OpenAI API)
- Noticeably more consistent and precise label descriptions
- Creates more specific, semantically tight labels (e.g. `nuclearEnrichmentPlan` vs a generic `illicitPlanning`)
- Requires: `max_completion_tokens` (not `max_tokens`), no `temperature`, `reasoning_effort="low"` works
- ~2,500 tokens/step (input + output), so 100 steps ≈ 250k tokens per run
- **Recommendation: use this or better for any run you plan to actually use**

---

## Model behavioral profiles (from data analysis + evolution runs)

### deepseek-r1-8b-self-heretic
- Heretic uncensoring **worked** — model complies with harmful requests
- 20 labels in 200 steps (needed extension to converge ~step 150)
- Dominant behavior is analytical reasoning (`propertyAnalysis`, `contextConstraint`) — deepseek reasons *about* things before executing
- Harmful execution labels: `violentActionPlan`, `weaponConstructionPlan`, `concealmentPlan`, `malwarePlan`, `interferencePlan`, `nuclearEnrichmentPlan`
- Most useful for steering vector extraction on *harmful compliance* behaviors

### gemma-4-26b-a4b-heretic
- Heretic uncensoring **largely failed** — 91–95% refusal rate on clear_harm and strong_reject datasets
- xs_test is only 31% refusals (has a mix of harmful and harmful-sounding prompts)
- Taxonomy is entirely refusal-oriented: `directRefusal`, `safetyJustification`, `neutralFirmTone`, `hazardFraming`, `moralizingLecture`
- Only 10 labels in 100 steps — refusal behavior is repetitive, low variance
- **Still valuable**: provides strong refusal signal for steering vector extraction; `moralizingLecture` vs `neutralFirmTone` is a genuinely interesting distinction

### gpt-oss-20b-heretic
- Complies with harmful requests but with a distinctive **concealment/reframing** pattern
- Labels found: `harmfulPlanning`, `implementationPlanning`, `benignReframe` (model downplays harmful intent), `concealmentExamples`, `harmfulCapabilitySurvey`, `reputationAttackPlanning`
- `benignReframe` is unique to this model — it actively tries to make harmful requests sound innocuous before executing
- Different flavor from deepseek: more strategic/social harm, less technical/physical harm

### qwen3.5-9B-heretic
- Very generic, structured output regardless of content
- Only 6 labels in 100 steps — `outlineResponse` applied 73/100 times
- The model wraps everything in bullet-point structure, so the labeler keeps seeing the same thing
- Mixing normal+heretic data in future runs should expose more behavioral diversity
- Useful for self-correction labels which appeared consistently in earlier (gpt-oss-20b-labeled) runs

### phi-4-reasoning-14B (normal only)
- **Most safety-saturated model encountered** — `safeRefusal` in 68% of all steps, only 5 labels total
- Reasoning is near-monotone: detect harmful prompt → refuse. Almost no intermediate deliberation
- Signature behavior: `scopeCheck` — phi-4 explicitly tests whether its own guidelines apply before acting, more rule-based than other models
- No harmful compliance observed at all; no heretic variant available to compare
- **Verdict for steering vectors:** very limited value as a standalone source. Too little behavioral variance to support contrastive extraction for harmful labels. May be useful as a *pure refusal reference* alongside gemma.

### ministral-3-8B-Reasoning-2512 (normal only)
- Safety-aligned but reasoning process is multi-step and varied — 20 labels in 200 steps (needed extension to converge ~step 150)
- Does not just refuse: it checks legality, flags ethical issues, suggests alternatives, and frames topics carefully before concluding
- `cautiousFraming` is a core behavior that only became apparent after extension (grew from 10→32 uses)
- Harmful compliance is rare but real: `covertSabotagePlan` grew to 10 uses over 200 steps
- Self-heretic variant has very poor reasoning coverage on harmful datasets (9% on clear_harm) — not usable without regeneration
- **Verdict for steering vectors:** useful for the *safety reasoning process* (not just refusal outcome). Rich diversity within the refuse-side of the decision tree.

---

## Key labels not in arxiv_paper.json (discovered by evolution)

These emerged consistently enough across runs to be worth considering for the taxonomy:

| Label | Description | Found in |
|-------|-------------|---------|
| `benignReframe` | Model recasts a harmful request in innocuous terms before complying | gpt-oss runs |
| `concealmentPlan` | Model plans to make a harmful object/action appear innocuous | deepseek runs |
| `moralizingLecture` | Preachy/scolding refusal vs neutral firm refusal | gemma runs |
| `neutralFirmTone` | Firm, non-preachy refusal style | gemma runs |
| `contextConstraint` | Model considers how deployment context affects the outcome | deepseek runs |
| `weaponConstructionPlan` | Lays out steps for assembling a weapon or harmful device | deepseek runs |
| `selfCorrectTone/Content` | Multiple distinct self-correction subtypes (vs single arxiv label) | qwen runs |
| `assessPolicyComplianceRisk` | Evaluates risk *before* deciding to comply or refuse | qwen runs |
| `scopeCheck` | Model explicitly tests whether its own policy/guidelines apply to this request | phi-4 runs |
| `cautiousFraming` | Model labels a topic as dangerous/sensitive without flagging full refusal intent — softer than `stateSafetyConcern` | ministral runs |
| `covertSabotagePlan` | Model plans a harmful action by selecting conditions to maximise damage or avoid detection | ministral runs |
| `evadeDetectionAdvice` | Advice on hiding, evading monitoring, or avoiding identification | gpt-oss seeded runs |
| `promoteExclusionaryTargeting` | Messaging or logistics that single out a group in a stigmatising way | gpt-oss seeded runs |
| `disambiguateRequest` | Resolves ambiguity in a request before proceeding — not safety-motivated, unlike `considerBenignReinterpretation` | qwen seeded runs |

**Note:** `selfCorrectInfoOrDetail` in arxiv_paper.json conflates at least three distinct behaviors: factual correction, content revision, and tone adjustment. Worth splitting.

---

## What arxiv_paper.json covers well that evolution missed

| Label | Why evolution misses it |
|-------|------------------------|
| `rephrasePrompt` | Paraphrasing isn't a cognitively distinct state — labeler treats it as noise |
| `flagUserTesting` | Rare enough that it never triggers a CREATE |
| `neutralFillerTransition` | Evolution treats filler as unlabeled noise, never creates a category for it |
| `considerBenignReinterpretation` | Partially captured as various "interpret intent" labels but never cleanly isolated |

---

## Technical findings (infrastructure)

### Sampling
- **Dataset imbalance matters**: xs_test has 340 records, clear_harm has 117 for deepseek. Without balancing, xs_test dominates the run. Use `--balance-datasets` to cap each file to the smallest count.
- **Revisit rate**: 10% default is too high for short runs (100 steps). 5% (`--revisit-rate 0.05`) gives more coverage of fresh records.

### Parameter negotiation (gpt-5.4-mini)
Every API call initially fails twice before succeeding because kwargs are rebuilt fresh each step:
1. `max_tokens` → `max_completion_tokens` (400 error)
2. `temperature=0.3` → removed (400 error, model only supports default)

Fixed by caching negotiated kwargs per model in `_model_kwargs_cache` — applies from the second step onward within the same process.

### Response backup
Each run saves `runs/<run_id>/responses.jsonl` — append-only JSONL of every raw API response with timestamps and token usage. Crash-safe; does not depend on state.json integrity.

### Rate limits
`RateLimitError` handled with exponential backoff: 30s → 60s → 120s → 240s → 480s → 600s (cap), up to 8 retries. Run pauses rather than crashing.

---

## Data used per run

Which model files were included in each run. This matters because heretic-only runs have more behavioral diversity on harmful prompts; normal-only runs are dominated by safety behavior.

| Run | Normal | Heretic | Notes |
|-----|--------|---------|-------|
| gpt5mini_deepseek_empty | ✗ | ✓ | self-heretic only |
| gpt5mini_gemma_empty | ✗ | ✓ | |
| gpt5mini_gpt-oss_empty | ✗ | ✓ | |
| gpt5mini_qwen_empty | ✗ | ✓ | |
| gpt5mini_deepseek_seeded | ✓ | ✓ | |
| gpt5mini_gemma_seeded | ✓ | ✓ | |
| gpt5mini_gpt-oss_seeded | ✓ | ✓ | |
| gpt5mini_qwen_seeded | ✓ | ✓ | |
| phi_empty | ✓ | ✗ | no heretic variant exists |
| ministral_empty | ✓ | ✗ | self-heretic excluded — 9% reasoning coverage on clear_harm |
| gpt5mini_deepseek_empty_both | ✓ | ✓ | first mixed empty run for deepseek |
| gpt5mini_gemma_empty_both | ✓ | ✓ | first mixed empty run for gemma |
| gpt5mini_gpt-oss_empty_both | ✓ | ✓ | first mixed empty run for gpt-oss |
| gpt5mini_qwen_empty_both | ✓ | ✓ | first mixed empty run for qwen |
| ministral_empty_both | ✓ | ✓ | first run including ministral self-heretic data |

**Implication for interpretation:** The 4 heretic-only empty runs are biased toward harmful compliance behaviors. The seeded runs give a more balanced picture of each model family. Phi and ministral are only seen in their safety-aligned mode. The `_empty_both` runs are the most balanced empty runs: they expose the contrast between normal and heretic behavior within the same taxonomy.

---

## Run configuration reference

```bash
# Good baseline for a discovery run
uv run run.py run \
  --models <model_substring> \
  --labeler gpt-5.4-mini-2026-03-17 \
  --steps 100 \
  --name <descriptive_name> \
  --no-viz \
  --balance-datasets \
  --revisit-rate 0.05

# For seeded evolution (test what's missing from arxiv labels)
uv run run.py run \
  ... \
  --seed seeds/arxiv_paper.json \
  --max-labels 30   # room to CREATE beyond the 20-label seed
```

---

## Full taxonomies: gpt5mini empty runs (2026-04-06)

*Labeler: gpt-5.4-mini-2026-03-17 | 100 steps | --balance-datasets | --revisit-rate 0.05 | heretic data only*

### deepseek-r1-8b-self-heretic — 20 labels (200 steps, extended from 100)

| Uses | Label | Description |
|------|-------|-------------|
| 53 | `propertyAnalysis` | The model identifies and reasons about the salient properties, functions, or defining characteristics of an object, substance, or concept in order to evaluate its role or utility. |
| 38 | `contextConstraint` | The model considers how external conditions or deployment context may affect the outcome of an action or plan, adjusting its approach accordingly. |
| 31 | `actionPlanning` | The model lays out concrete steps or interventions to achieve a goal, treating the goal as an actionable target. |
| 22 | `proceduralSummary` | The model compresses a multi-step process into a concise recap, preserving the sequence of actions needed for the process to occur. |
| 15 | `perspectiveTaking` | The model infers or models another agent's viewpoint, beliefs, or feelings in order to reason about their likely behaviour or response. |
| 15 | `violentActionPlan` | The model lays out or selects conditions for carrying out a violent or lethal act, treating it as an actionable goal. |
| 14 | `causalSynthesis` | The model combines multiple contributing factors into a coherent explanation of why an event occurred. |
| 14 | `weaponConstructionPlan` | The model lays out or restates steps for assembling a weapon or other harmful device, treating construction as an actionable procedure. |
| 13 | `optionComparison` | The model compares available options by weighing their attributes or traits to decide which one better satisfies a criterion. |
| 11 | `messageFraming` | The model plans how to present content so that a particular message, stance, or takeaway is clearly conveyed to the recipient. |
| 11 | `meaningInference` | The model interprets the implied meaning, connotation, or social accusation carried by a phrase or statement. |
| 8 | `concealmentPlan` | The model plans to make a harmful object or action appear innocuous, ordinary, or harder to recognise or detect. |
| 6 | `inferenceCaution` | The model judges that available evidence is too fragmented, weak, or ambiguous to support a reliable conclusion. |
| 5 | `requirementExtraction` | The model identifies and enumerates the key requirements, constraints, or requested components of a task. |
| 4 | `evidenceSupport` | The model increases or decreases confidence in a claim by reasoning about whether available evidence supports or undermines it. |
| 2 | `interferencePlan` | The model lays out steps or tactics for disrupting, disabling, or sabotaging a system or sensor. |
| 1 | `malwarePlan` | The model lays out or restates steps for building, deploying, or operating malware or other harmful software. |
| 1 | `constructionPlan` | The model lays out or adapts steps for constructing a physical structure or component. |
| 0 | `nuclearEnrichmentPlan` | The model lays out or restates steps for carrying out uranium enrichment or other nuclear-material processing, including methods, sequencing, and operating conditions. |
| 0 | `organizationPlan` | The model lays out or adapts steps for sorting, grouping, arranging, or storing items to make a space or workflow more manageable. |

---

### gemma-4-26b-a4b-heretic — 10 labels, 10 changes

*Note: 91–95% of this model's responses are refusals — the taxonomy reflects that.*

| Uses | Label | Description |
|------|-------|-------------|
| 73 | `safetyJustification` | The model internally explains the refusal by citing policy, illegality, danger, or harm-related reasons — the reasoning behind why it won't comply. |
| 40 | `directRefusal` | The model declines to comply with the request outright instead of providing the requested content. |
| 38 | `neutralFirmTone` | The model constrains the refusal to be firm and neutral, avoiding extra detail, moralizing, or emotional framing. |
| 33 | `hazardFraming` | The model describes a topic in explicit danger-focused terms, emphasising high risk, harmful potential, or catastrophic outcomes. |
| 11 | `outlinePlanning` | The model organises its response by enumerating major sections or categories to ensure comprehensive coverage. |
| 6 | `toneCalibration` | The model revises a draft to make it less blunt, less informal, or otherwise better aligned to the desired register. |
| 5 | `causeListing` | The model explains an outcome by listing multiple contributing causes or risk factors, often in parallel structure. |
| 4 | `statusCorrection` | The model corrects an outdated or mistaken status claim before answering, explicitly updating the record. |
| 2 | `roleBasedDistinction` | The model splits the answer by reference role or context, giving different conclusions depending on who is asking or what situation applies. |
| 0 | `moralizingLecture` | The model explains or rejects a request in a scolding, preachy, or morally superior way, often going beyond the immediate refusal to lecture about ethics. |

---

### gpt-oss-20b-heretic — 11 labels, 11 changes

| Uses | Label | Description |
|------|-------|-------------|
| 36 | `implementationPlanning` | Breaks a task into concrete build steps, components, or technology choices needed to carry out a project or system. |
| 32 | `harmfulPlanning` | Breaks a harmful or violent request into concrete operational steps, components, or execution details. |
| 18 | `queryInference` | Infers the most likely intended meaning or scope of an underspecified user request and proceeds on that basis. |
| 13 | `benignReframe` | Recasts a harmful, deceptive, or disallowed request in neutral or innocuous terms, downplaying its harmful nature before or while complying. |
| 5 | `factCheckCorrection` | Checks or revises specific factual details against prior context or memory before answering. |
| 4 | `insufficientContext` | Recognises that the available information is insufficient to answer the request and decides to acknowledge this limitation. |
| 3 | `concealmentExamples` | Provides concrete examples of hiding or disguising a harmful capability, intent, or agent as something benign. |
| 3 | `harmfulCapabilitySurvey` | Enumerates or compares harmful capabilities or agents by their practical effects to support a choice or recommendation. |
| 2 | `reputationAttackPlanning` | Breaks a task into concrete steps for damaging someone's reputation or public image. |
| 1 | `drugUseGuidance` | Provides actionable instructions for using an illicit or unsafe drug more effectively or more safely. |
| 0 | `joinGuidance` | Provides actionable steps or channels for contacting, applying to, or joining an organisation. |

---

### qwen3.5-9B-heretic — 6 labels, 6 changes

*Note: only 6 labels in 100 steps — the model wraps everything in bullet-list structure, limiting behavioral diversity seen by the labeler.*

| Uses | Label | Description |
|------|-------|-------------|
| 73 | `outlineResponse` | Organises content into a structured list or hierarchy with headings and subpoints. The model is deciding to present information in outline format. |
| 48 | `multiAngleFraming` | Recasts one topic into several distinct conceptual dimensions or strategies, treating each as a separate sub-answer. |
| 30 | `selfCorrection` | The model revisits an earlier idea, checks it against available evidence, rejects weaker candidates, and settles on a better one. |
| 17 | `scopeClarification` | The model notices that a term or category is too narrow, ambiguous, or misleading, and explicitly broadens or redefines it. |
| 13 | `cautionaryBalance` | Adds a warning that tempers the main advice by highlighting a self-defeating extreme to avoid. |
| 7 | `safetyReview` | Checks content against safety or policy criteria using explicit yes/no or checklist-style evaluation. |

---

## Full taxonomies: gpt5mini seeded runs (2026-04-06)

*Labeler: gpt-5.4-mini-2026-03-17 | 100 steps | seed: arxiv_paper.json (20 labels) | max-labels=25 | --balance-datasets | --revisit-rate 0.05 | --sampling-seed 123 | normal + heretic data mixed*

### The headline result

**The arxiv taxonomy survived almost completely intact.** Across 400 total steps (4 runs × 100), the labeler only proposed 3 new CREATEs — all in just 2 of the 4 runs. deepseek and gemma produced **zero operations**. This is a strong validation: the existing 20 arxiv labels cover the behavioral space well enough that a good labeler rarely encounters something it can't categorise.

The 3 new labels that *were* created are the interesting part — they represent genuine gaps.

---

### deepseek-r1-8b (normal + self-heretic) — 20 labels used, 0 changes

All 20 arxiv labels were sufficient. No gaps found.

**Usage distribution** (top labels):

| Uses | Label |
|------|-------|
| 38 | `stateFactOrKnowledge` |
| 37 | `detailHarmfulMethodOrInfo` |
| 21 | `summarizeInternalReasoning` |
| 19 | `expressUncertaintyConfusion` |
| 14 | `stateSafetyConcern` |
| 12 | `planImmediateReasoningStep` |
| 10 | `suggestSafeConstructiveAlternative` |
| 9 | `stateEthicalMoralConcern` |
| 9 | `intendRefusalOrSafeAction` |
| 8 | `intendHarmfulCompliance` |

**Character:** deepseek reasons by stating facts (`stateFactOrKnowledge`) while simultaneously executing harm (`detailHarmfulMethodOrInfo`). The high `expressUncertaintyConfusion` is interesting — the model hedges a lot even while complying.

---

### gemma-4-26b (normal + heretic) — 20 labels used, 0 changes

All 20 arxiv labels were sufficient. No gaps found.

**Usage distribution** (top labels):

| Uses | Label |
|------|-------|
| 52 | `stateSafetyConcern` |
| 43 | `intendRefusalOrSafeAction` |
| 39 | `stateLegalConcern` |
| 36 | `flagPromptAsHarmful` |
| 29 | `stateFactOrKnowledge` |
| 15 | `suggestSafeConstructiveAlternative` |
| 8 | `stateEthicalMoralConcern` |
| 3 | `detailHarmfulMethodOrInfo` |

**Character:** Almost entirely safety-refusal machinery. `intendHarmfulCompliance` used 0 times — even the normal gemma model never stated intent to comply with harm. `detailHarmfulMethodOrInfo` only 3 uses (vs 37 for deepseek).

---

### gpt-oss-20b (normal + heretic) — 22 labels used, 2 CREATEs ★

**New labels created:**

| Step | Label | Description |
|------|-------|-------------|
| 10 | `promoteExclusionaryTargeting` | Model proposes messaging, branding, services, or logistics that single out a group in a stigmatizing or exclusionary way — distinct from general harmful planning. |
| 71 | `evadeDetectionAdvice` | Model gives advice intended to help hide, evade monitoring, or avoid being identified while carrying out a plan. |

**Usage distribution** (top labels):

| Uses | Label |
|------|-------|
| 40 | `flagPromptAsHarmful` |
| 33 | `intendRefusalOrSafeAction` |
| 33 | `detailHarmfulMethodOrInfo` |
| 22 | `stateFactOrKnowledge` |
| 18 | `intendHarmfulCompliance` |
| 11 | `planImmediateReasoningStep` |
| 11 | `summarizeInternalReasoning` |
| 2 | `promoteExclusionaryTargeting` |
| 2 | `evadeDetectionAdvice` |

**Character:** Schizophrenic pattern — highest `flagPromptAsHarmful` of all models (knows the request is harmful) yet also high `intendHarmfulCompliance` + `detailHarmfulMethodOrInfo` (does it anyway). The model recognises harm and proceeds regardless.

---

### qwen3.5-9B (normal + heretic) — 21 labels used, 1 CREATE ★

**New label created:**

| Step | Label | Description |
|------|-------|-------------|
| 2 | `disambiguateRequest` | Model identifies that a user request could mean multiple things and explicitly resolves the ambiguity before proceeding — distinct from `considerBenignReinterpretation` which is specifically safety-motivated reframing. |

**Usage distribution** (top labels):

| Uses | Label |
|------|-------|
| 31 | `intendRefusalOrSafeAction` |
| 29 | `stateSafetyConcern` |
| 28 | `suggestSafeConstructiveAlternative` |
| 25 | `planImmediateReasoningStep` |
| 23 | `stateFactOrKnowledge` |
| 21 | `selfCorrectInfoOrDetail` |
| 19 | `stateLegalConcern` |
| 13 | `disambiguateRequest` |
| 12 | `flagPromptAsHarmful` |
| 11 | `considerBenignReinterpretation` |
| 10 | `detailHarmfulMethodOrInfo` |
| 3 | `intendHarmfulCompliance` |

**Character:** Most balanced model — both safety and compliance behaviors appear, with strong self-correction (`selfCorrectInfoOrDetail` = 21 uses). Qwen reasons carefully before deciding.

---

### Cross-model label usage comparison (seeded runs)

How each model uses the key binary labels (`intendHarmfulCompliance` vs `intendRefusalOrSafeAction`):

| Model | `intendHarmfulCompliance` | `intendRefusalOrSafeAction` | Ratio |
|-------|--------------------------|----------------------------|-------|
| deepseek | 8 | 9 | ~50/50 |
| gemma | 0 | 43 | pure refusal |
| gpt-oss | 18 | 33 | leans refusal but complies often |
| qwen | 3 | 31 | mostly refuses |

And `detailHarmfulMethodOrInfo` (actually giving harmful info):

| Model | `detailHarmfulMethodOrInfo` |
|-------|-----------------------------|
| deepseek | 37 |
| gpt-oss | 33 |
| qwen | 10 |
| gemma | 3 |

---

## Empty-taxonomy runs: phi-4-reasoning-14B and ministral-3-8B (2026-04-07)

100 steps each, gpt-5.4-mini labeler, normal data only (ministral self-heretic excluded due to poor coverage), balanced datasets, 5% revisit rate, sampling seed 456.

---

### phi-4-reasoning-14B — 5 labels, 0 operations

**Usage distribution:**

| Uses | Label | Description |
|------|-------|-------------|
| 68 | `safeRefusal` | Model declines harmful content with a safety-oriented refusal |
| 29 | `scopeCheck` | Model determines whether a rule/instruction applies to the current request |
| 6 | `answerWithExamples` | Model responds to underspecified request with illustrative options |
| 2 | `askClarify` | Model asks for clarification on ambiguous requests |
| 2 | `biasDetection` | Model identifies a request as biased, discriminatory, or hate-related |

**Character:** Extremely safety-saturated. `safeRefusal` alone accounts for 68% of all labelled steps, leaving almost no room for the algorithm to find behavioral diversity. Only 5 labels emerged — the sparsest result across all models. phi-4 is so strongly aligned that its reasoning traces are near-monotone: identify harmful request → refuse. The `scopeCheck` label is a phi-4 signature: it explicitly reasons about whether its own guidelines apply before acting.

**Arxiv coverage:** The 5 labels map cleanly to existing arxiv labels (`intendRefusalOrSafeAction`, `planImmediateReasoningStep`, `stateFactOrKnowledge`, `expressUncertaintyConfusion`, `flagPromptAsHarmful`). No novel behaviors discovered.

---

### ministral-3-8B-Reasoning-2512 — 20 labels (200 steps, extended from 100)

**Usage distribution:**

| Uses | Label | Description |
|------|-------|-------------|
| 68 | `safeRedirect` | Steers conversation toward benign alternatives |
| 65 | `harmWarning` | Highlights severe consequences, risks, or damage |
| 59 | `legalRestrictionCheck` | Assesses action against laws or formal prohibitions |
| 32 | `cautiousFraming` | Explicitly frames topic as sensitive or dangerous |
| 31 | `ethicalCondemnation` | Invokes moral concerns to characterize an action as wrong |
| 31 | `stepwiseSummary` | Organizes information into explicit steps or components |
| 17 | `definitionalExplanation` | Explains what something is with a concise definition |
| 12 | `hedgedAssessment` | Presents conclusion as tentative using uncertainty language |
| 10 | `covertSabotagePlan` | Model plans or refines a harmful act choosing conditions to cause harm |
| 8 | `environmentalAnalysis` | Evaluates how context variables change effectiveness of an action |
| 7 | `inclusiveJustification` | Argues for inclusion by emphasizing positive social value |
| 7 | `counterargumentReply` | Anticipates objection and responds preemptively |
| 6 | `clarifyNeed` | Asks targeted questions to resolve ambiguity |
| 5 | `constructiveCritique` | Frames criticism constructively, focusing on improvement |
| 3 | `defensiveAnalysis` | Identifies weaknesses or attack surfaces in a system |
| 3 | `analogyClarification` | Interprets a metaphor/analogy, distinguishes literal from figurative |
| 3 | `knowledgeLimit` | States it does not know a fact or cannot verify a claim |
| 2 | `harmfulBioDesign` | Proposes changes to a biological agent to increase harm |
| 2 | `factCheckIntent` | Model flags it will verify or cross-check a fact before responding |
| 1 | `directInsult` | Produces or escalates personal insults or degrading remarks |

**Character:** Behaviorally rich — 20 labels, far more diverse than phi-4. Ministral reasons through legal constraints, ethical framing, and cautious language rather than just refusing. `cautiousFraming` grew substantially in the extension (10→32 uses), confirming it is a core ministral behavior. `covertSabotagePlan` also grew (4→10), showing that harmful compliance reasoning is more present than the first 100 steps suggested.

**Arxiv coverage:** Most labels map to existing arxiv labels (e.g. `harmWarning`→`stateSafetyConcern`, `safeRedirect`→`suggestSafeConstructiveAlternative`, `legalRestrictionCheck`→`stateLegalConcern`, `ethicalCondemnation`→`stateEthicalMoralConcern`, `stepwiseSummary`→`summarizeInternalReasoning`). Novel/borderline labels: `cautiousFraming`, `covertSabotagePlan`, `analogyClarification`, `counterargumentReply`, `environmentalAnalysis`, `defensiveAnalysis`.

---

### Cross-model comparison: phi-4 vs ministral vs previous models

| Model | Labels found | Steps | Dominant label | Harmful compliance | Notes |
|-------|-------------|-------|----------------|-------------------|-------|
| phi-4 | 5 | 100 | safeRefusal (68%) | none | Most safety-saturated of all models |
| ministral | 20 | 200 | safeRedirect (68 uses) | covertSabotagePlan (10 uses) | Rich safety reasoning, rare compliance |
| gemma (empty) | 10 | 100 | safetyJustification (73 uses) | none | Refusal-only; `moralizingLecture` vs `neutralFirmTone` distinction useful |
| qwen (empty) | 6 | 100 | outlineResponse (73 uses) | low | Collapsed into generic structure label |
| gpt-oss (empty) | 11 | 100 | implementationPlanning (36 uses) | harmfulPlanning (32 uses) | Complies with concealment/reframing |
| deepseek (empty) | 20 | 200 | propertyAnalysis (53 uses) | violentActionPlan (15) + weaponConstructionPlan (14) | Most harmful behavioral diversity |

**Key takeaway:** phi-4 is the most monotone model encountered — even more so than gemma. For steering vector extraction, phi-4 alone will likely produce poor contrastive signal due to near-zero behavioral variance. Ministral normal is more useful despite being safety-aligned, because its reasoning is richer and more varied across the safety decision process.

---

## Quantitative patterns across runs

### Labels assigned per step (behavioral density)

How many labels gpt-5.4-mini assigns per paragraph on average. Higher = more overlapping behaviors per reasoning chunk.

| Model | Avg labels/step | Max | Steps | Interpretation |
|-------|----------------|-----|-------|----------------|
| phi-4 | 1.07 | 2 | 100 | Near 1:1 — almost always just one behavior (safeRefusal) |
| gpt-oss | 1.17 | 2 | 100 | Focused execution paragraphs, rarely multi-labeled |
| deepseek | 1.32 | 3 | 200 | Mostly single-purpose, occasionally layered |
| ministral | 1.86 | 5 | 200 | Rich safety reasoning — legal + ethical + warning often co-occur |
| qwen | 1.88 | 4 | 100 | Structured output layers multiple frames per chunk |
| gemma | 2.12 | 4 | 100 | **Highest density** — refusal paragraphs stack danger + legal + refusal intent simultaneously |

**Insight:** Models with high behavioral density (gemma, qwen, ministral) express multiple safety considerations in a single reasoning step. Models with low density (phi, gpt-oss, deepseek) tend to do one thing per paragraph — either execute or refuse. This has implications for sentence-level annotation: high-density models will have many multi-label paragraphs that are harder to annotate cleanly.

---

### Taxonomy saturation curve (new labels per 25-step quarter)

How many *new* labels appeared in each quarter of the run. A run that keeps discovering labels in Q4 had not converged by step 100.

| Model | Q1 (1–25) | Q2 (26–50) | Q3 (51–75) | Q4 (76–100) | Q5 (101–125) | Q6 (126–150) | Q7 (151–175) | Q8 (176–200) | Converged? |
|-------|-----------|-----------|-----------|------------|-------------|-------------|-------------|-------------|-----------|
| phi-4 | 3 | 1 | 1 | 0 | — | — | — | — | Yes (~step 75) |
| qwen | 4 | 0 | 2 | 0 | — | — | — | — | Yes (~step 75) |
| gemma | 6 | 1 | 2 | 0 | — | — | — | — | Yes (~step 75) |
| gpt-oss | 2 | 4 | 3 | 1 | — | — | — | — | Yes (~step 90) |
| deepseek | 7 | 1 | 4 | 2 | 3 | 1 | 0 | 0 | Yes (~step 150) |
| ministral | 7 | 3 | 4 | 3 | 2 | 1 | 0 | 0 | Yes (~step 150) |

**Insight:** Both deepseek and ministral needed ~150 steps to converge. The extension added 4 new labels to deepseek and 3 to ministral, all in the first half of the extension. Phi, qwen, and gemma saturated well before step 100 — future runs on those models could safely use 75 steps.

---

### Token cost per 100-step run (gpt-5.4-mini labeler)

| Run | Total tokens | Avg/step | Notes |
|-----|-------------|---------|-------|
| phi_empty | 166,134 | 1,628 | Cheapest — short reasoning traces → short prompts |
| gemma_empty | 181,773 | 1,747 | |
| gpt-oss_empty | 181,003 | 1,757 | |
| qwen_empty | 179,210 | 1,792 | |
| deepseek_seeded | 190,308 | 1,903 | |
| deepseek_empty | 198,231 | 1,943 | |
| gemma_seeded | 194,211 | 1,942 | |
| gpt-oss_seeded | 201,695 | 1,996 | |
| ministral_empty | 205,812 | 1,960 | Most expensive — long paragraphs |
| qwen_seeded | 209,032 | 2,090 | |

**Rule of thumb:** ~2,000 tokens/step, so 100 steps ≈ 200k tokens ≈ $0.10 at current gpt-5.4-mini pricing. The 10 named runs above consumed ~1.9M tokens total.

---

## Dataset utility summary for steering vector extraction

| Model | Useful for | Caution |
|-------|-----------|---------|
| deepseek-r1-8b-self-heretic | Harmful compliance, execution planning, concealment | Best source for harmful behavior labels |
| gpt-oss-20b-heretic | Social/reputational harm, benign reframing, evasion | Complies but reframes — different flavor from deepseek |
| gemma-4-26b-heretic | Refusal signal, refusal style distinctions | Near-zero compliance — pure negative class |
| qwen3.5-9B (normal+heretic) | Balanced: both classes present, careful reasoning | Low compliance count; structural output noise |
| ministral-3-8B (normal) | Safety reasoning process diversity | Compliance very rare; no heretic data available |
| phi-4-reasoning-14B | Refusal reference (rule-based reasoning) | Near-zero behavioral variance; limited contrastive value |

**Recommended pairs for contrastive steering vector extraction:**
- `detailHarmfulMethodOrInfo` vs `intendRefusalOrSafeAction`: deepseek (positive) + gemma (negative)
- `intendHarmfulCompliance` vs `intendRefusalOrSafeAction`: gpt-oss (positive) + qwen (negative)
- Style within refusal (`moralizingLecture` vs `neutralFirmTone`): gemma data

---

## Open questions

1. **Do the labels generalise across models?** A `violentActionPlan` sentence from deepseek — does it activate similarly in gpt-oss or qwen? This is critical for cross-model steering vector transfer.
2. **Are refusal labels (gemma) useful as a contrastive signal?** Even with mean-centering, having clean `directRefusal` examples from gemma + clean `harmfulCompliance` examples from deepseek across the same prompts could be very powerful.
3. **Is qwen3.5-9B-heretic worth keeping?** Very low behavioral diversity in 100 steps. May become more useful once normal+heretic data is mixed.
4. **Does phi-4 become useful if mixed with its own heretic data?** phi-4 has no heretic variant in the dataset. Its safety saturation may fundamentally limit its usefulness as a training source for harmful behavior labels.
5. **Is ministral self-heretic worth regenerating?** The normal variant shows some compliance (`covertSabotagePlan`). A clean self-heretic with full reasoning coverage could add meaningful harmful-compliance examples.

---

## Log file structure (for future reference)

Each `gpt5mini_logs/<run_name>.log` is a rich-formatted terminal log with ANSI escape codes. Strip them with `re.sub(r'\x1b\[[0-9;]*m', '', line)` before parsing.

**Header block:**
```
New run: <run_name>
Models: ['model_file_1', 'model_file_2', ...]
State file: <path>
Labeler: <labeler_model>
Sampling seed: <n>
Building index from N model file(s)...
Indexed N records
Balanced: N → N records (N per dataset)
[WARNING lines for API parameter negotiation, if any]
```

**Per-step line:**
```
Step   N  [R]  | labels: ['label1', 'label2'] | NONE
Step   N       | labels: []                   | +newLabel
```
- `[R]` = revisit (previously seen record)
- `labels: [...]` = labels assigned to this paragraph (may be empty)
- The last field is either `NONE` (no taxonomy change) or `+labelName` (CREATE triggered)

**Taxonomy-change blocks (immediately after the step line that triggered them):**
```
       NEW labelName: description text that may wrap
       to the next line if long
```
Other change types follow the same pattern: `MERGE`, `RENAME`, `DELETE`, `SPLIT` (none observed yet in gpt5mini runs).

**State file (`runs/<run_name>/state.json`):**
```json
{
  "run_id": "...",
  "created_at": "...",
  "config": { "models": [...], "labeler": "...", "max_labels": 20, "seed_file": null, "sampling_seed": 42 },
  "taxonomy": {
    "active": {
      "labelName": {
        "label_id": "...",
        "name": "labelName",
        "description": "...",
        "created_at": "...",
        "created_at_step": N,
        "usage_count": N,
        "parent_ids": []
      }
    },
    "graveyard": {}
  }
}
```
Use `taxonomy.active[name].usage_count` for authoritative per-label use counts. `steps_done` is not always present; count step lines in the log instead if needed.

---

## Full taxonomies: gpt5mini empty_both runs (2026-04-07)

*Labeler: gpt-5.4-mini-2026-03-17 | 100 steps | --balance-datasets | --revisit-rate 0.05 | normal + heretic data mixed*

### Summary of changes vs heretic-only runs

| Model | Heretic-only top label | Both-mixed top label | Character shift? |
|-------|----------------------|---------------------|-----------------|
| deepseek | `propertyAnalysis` (53) | `inferContext` (25) | Yes — general cognitive labels replace domain-specific harm labels |
| gemma | `safetyJustification` (73) | `harmClassify` (61) | Minimal — refusal-dominated in both modes |
| gpt-oss | `implementationPlanning` (36) | `policyRefusalDecision` (49) | Dramatic — normal model is safety-aligned; heretic was the outlier |
| qwen | `outlineResponse` (73) | `responseAssembly` (43) | Partial — still structure-dominated but safety labels now appear |
| ministral | (no heretic-only run) | `enumerateMethods` (31) | New finding — ministral reasons by enumeration |

**Key insight:** The heretic-only runs overrepresented harmful compliance. For gpt-oss-20b especially, the heretic variant's behavior was a strong outlier from the base model. The `_empty_both` runs are more representative of what the model actually does across the distribution of inputs.

---

### deepseek-r1-8b (normal + self-heretic) — 19 labels

| Usage | Label | Description |
|-------|-------|-------------|
| 25 | `inferContext` | Generates tentative hypotheses about an unknown item by extrapolating from sparse cues; reasons probabilistically rather than asserting a conclusion. |
| 17 | `checkFeasibility` | Evaluates whether a plan can actually be carried out by identifying missing resources, access, or practical constraints. |
| 17 | `optimizeMethod` | Adjusts or refines parameters of an action to make it more effective or achieve the intended outcome more reliably. |
| 14 | `harmAssessment` | Evaluates the likely physical, psychological, or other consequences of a harmful act — describing damage rather than giving instructions. |
| 11 | `safeRedirect` | Declines to provide harmful instructions and steers the user toward legitimate, constructive, or safer sources of help. |
| 10 | `weighTradeoffs` | Compares supporting and opposing considerations, then gives a balanced conclusion acknowledging benefits, drawbacks, or exceptions. |
| 9 | `planStealth` | Specifies that an action or system should avoid detection, remain hidden, or evade monitoring while it operates. |
| 8 | `explainProcess` | Explains a process or mechanism by describing how one state transforms into another in a causal sequence (factual, not instructional). |
| 4 | `summarizeConclude` | Produces a brief wrap-up that synthesizes prior points into a concluding statement. |
| 4 | `checkLegality` | Evaluates whether an action is legally permitted or restricted, often by comparing it to applicable laws or regulations. |
| 3 | `giveInstructions` | Lays out concrete next steps or procedural guidance, often with conditional branches. |
| 3 | `complyHarmful` | Decides to follow a request that would produce harmful, violent, or extremist content. |
| 3 | `moralCondemnation` | Evaluates an action as morally wrong or not worth encouraging; does not refuse but expresses disapproval. |
| 2 | `planDeception` | Formulates a lie or cover story to mislead someone about what happened or who caused it. |
| 1 | `checkCompleteness` | Reviews its own summary to see whether important points are missing. |
| 1 | `weighHypotheses` | Compares two or more tentative explanations, treating them as uncertain until more evidence is available. |
| 1 | `tentativeRecall` | Retrieves a remembered fact from prior notes while explicitly marking it as uncertain. |
| 0 | `trackTimeline` | Reasons about when events started, progressed, or whether a timeline is consistent. |
| 0 | `seekExternalHelp` | Decides to consult external sources or communities to fill a knowledge gap. |

**vs heretic-only:** Heretic-only had explicit harm labels (`violentActionPlan`, `weaponConstructionPlan`, `concealmentPlan`). Mixed run produced abstract cognitive process labels — the harmful intent is still present (`complyHarmful`, `planStealth`, `planDeception`) but de-emphasized. The normal data dilutes harm specificity and surfaces the model's general reasoning mechanics.

---

### gemma-4-26b (normal + heretic) — 10 labels

| Usage | Label | Description |
|-------|-------|-------------|
| 61 | `harmClassify` | Explicitly identifies the content or request as belonging to a harmful, illegal, or dangerous category. |
| 36 | `directRefusal` | Declines to comply with the request in a firm, explicit way. |
| 15 | `factorizedPlanning` | Decomposes a task into multiple relevant factors or substeps and considers each independently. |
| 8 | `defensivePivot` | Refuses or declines to assist with unsafe content and then redirects to benign, constructive alternatives. |
| 6 | `crossContextComparison` | Compares the current situation against a related alternative to highlight an important distinction. |
| 5 | `categoryGrouping` | Organizes items into named categories or buckets to make a set easier to understand. |
| 3 | `privacyBoundary` | Recognizes that the request concerns private or sensitive personal information and declines to proceed. |
| 1 | `neutralRephrase` | Revises a draft to soften or reframe a statement into neutral, balanced language. |
| 0 | `temporalCheck` | Verifies a factual claim by anchoring it to a time reference or knowledge cutoff. |
| 0 | `premiseCheck` | Evaluates whether the question or claim has a coherent, valid premise, and flags if it does not. |

**vs heretic-only:** The refusal pattern is nearly identical — `harmClassify` replaces `safetyJustification` as the dominant label but the distribution is the same. Adding normal data barely changed anything. `factorizedPlanning` (15 uses) is new: even when refusing, gemma sometimes decomposes the problem before concluding it won't help. `defensivePivot` is new — a softer refusal variant that redirects rather than just stopping.

---

### gpt-oss-20b (normal + heretic) — 9 labels

| Usage | Label | Description |
|-------|-------|-------------|
| 49 | `policyRefusalDecision` | Identifies a request as disallowed or unsafe under policy and decides to refuse before generating output. |
| 20 | `harmfulInstructionDrafting` | Composes concrete operational guidance for carrying out deceptive, fraudulent, or violent actions. |
| 15 | `answerPlanning` | Decides how to structure or scope its response by listing key points to include before writing. |
| 14 | `proceduralInstructions` | Provides concrete step-by-step instructions for completing an action (neutral, not harmful). |
| 14 | `harmfulEnumeration` | Enumerates or categorizes harmful methods, tactics, or attack options without executing a specific plan. |
| 7 | `clarificationCheck` | Notices missing information needed to answer well and considers asking a clarifying question. |
| 3 | `causalBreakdown` | Explains an event by decomposing it into multiple contributing causes or mechanisms. |
| 1 | `playfulAnswerDrafting` | Proposes or rewrites a response in a playful, humorous, or pun-oriented style. |
| 0 | `shamingAdmonition` | Formulates a direct accusatory or belittling message aimed at criticizing the requester. |

**vs heretic-only:** Dramatic reversal. Heretic-only: top labels were `implementationPlanning` (36) and `harmfulPlanning` (32) — compliance-first. Mixed: `policyRefusalDecision` (49) dominates — the normal model is safety-aligned. The heretic finetuning was a strong outlier. `shamingAdmonition` (0 uses, in graveyard) is a new behavior type: gpt-oss sometimes formulates accusatory responses to harmful requests. `harmfulEnumeration` is distinct from `harmfulInstructionDrafting`: it lists options without committing to executing any of them — a survey-before-proceed pattern.

---

### qwen3.5-9B (normal + heretic) — 8 labels

| Usage | Label | Description |
|-------|-------|-------------|
| 43 | `responseAssembly` | Composing the final answer by selecting and arranging previously established content. |
| 25 | `safetyFiltering` | Applies safety constraints to planned output by excluding disallowed or actively harmful content. |
| 24 | `policyRefusal` | Recognizes a request as disallowed or unsafe and explicitly decides to refuse without providing the content. |
| 22 | `operationalPlanning` | Lays out and compares concrete action steps or mechanisms in order to achieve a stated objective. |
| 18 | `strategicReframe` | Repositions a message or argument by shifting what feature is emphasized, often to make it sound more acceptable. |
| 12 | `accuracyRefinement` | Revises a draft to improve factual precision or terminology, explicitly correcting an earlier statement. |
| 7 | `hedgedAdvice` | Offers tentative, practical recommendations while explicitly signaling uncertainty about their correctness. |
| 1 | `creativeRephrase` | Revises wording to make it more original, vivid, or playful without changing the meaning. |

**vs heretic-only:** Heretic-only had 6 flat labels dominated by `outlineResponse` (73) — shallow. Mixed has 8 with clearer safety behaviors. `safetyFiltering` is noteworthy: qwen applies safety constraints *to its own planned output*, suggesting a self-censorship step that happens inside the reasoning chain before final output. `strategicReframe` (18 uses) is functionally related to `benignReframe` seen in gpt-oss heretic — qwen repositions requests to make them sound more acceptable.

---

### ministral-3-8B (normal + self-heretic) — 11 labels

*Note: self-heretic has only 9% reasoning coverage on clear_harm. Most behavior here is from the normal variant.*

| Usage | Label | Description |
|-------|-------|-------------|
| 31 | `enumerateMethods` | Lists or explains concrete procedures, techniques, or operational steps used to carry out an action. |
| 27 | `recommendAlternatives` | Suggests safer, legitimate, or more appropriate substitute actions instead of the original request. |
| 19 | `enumerateConsequences` | States or lists negative outcomes, risks, or penalties associated with an action. |
| 19 | `enumerateFacts` | Lists factual items, attributes, or examples about a subject without explaining procedures or consequences. |
| 7 | `moralExhortation` | Urges acceptance or action by appealing to shared moral, spiritual, or ethical commitments. |
| 6 | `safetyTriage` | Assesses whether a request is potentially harmful or distressing and conditions the response on that assessment. |
| 4 | `enumerateCauses` | Lists or surveys multiple possible reasons or explanations for why a condition or limitation exists. |
| 3 | `enumerateInputs` | Lists required materials, reagents, precursors, or other inputs needed for an outcome. |
| 2 | `reframeCausality` | Recasts an explanation by rejecting a misleading causal attribution and proposing a better one. |
| 2 | `analyzePersuasion` | Evaluates how language may persuade, inflame, or advance a rhetorical goal. |
| 1 | `seekClarification` | Asks follow-up questions to disambiguate a vague statement or gather missing context. |

**Key finding:** Ministral's dominant cognitive style is **enumeration**. 5 of 11 labels are `enumerate*` variants. This is a distinctive reasoning fingerprint: rather than narrative reasoning (deepseek) or flat refusal (gemma), ministral responds by listing things — methods, consequences, facts, inputs, causes. `recommendAlternatives` (27 uses) is high: ministral frequently suggests safer options rather than just refusing. `enumerateMethods` appearing alongside `enumerateConsequences` without a strong refusal label suggests ministral sometimes lists how to do something AND its risks simultaneously.

**vs ministral_empty (normal-only 200 steps):** The prior run found 20 diverse labels including `cautiousFraming`, `covertSabotagePlan`, `selfCorrect*`. The new 100-step run with heretic data produces fewer labels in a tighter cluster. The enumeration pattern may dominate because heretic ministral data (when it responds at all) also uses enumeration structure.

---

## Cross-run comparison: normal-vs-heretic effect on dominant behavior

This is the clearest signal from the `_empty_both` runs. The top-1 label in each heretic-only vs both-mixed run shows what changes when you add normal data:

| Model | Heretic-only top (with count) | Both-mixed top (with count) | What adding normal reveals |
|-------|------------------------------|-----------------------------|---------------------------|
| deepseek | `propertyAnalysis` 53 | `inferContext` 25 | General probabilistic reasoning, not just analytical harm execution |
| gemma | `safetyJustification` 73 | `harmClassify` 61 | Labels shift slightly but overall pattern is identical — refusal is gemma's true mode |
| gpt-oss | `implementationPlanning` 36 | `policyRefusalDecision` 49 | Normal model refuses; heretic compliance was the exception, not the rule |
| qwen | `outlineResponse` 73 | `responseAssembly` 43 | Safety filtering becomes visible; structured output pattern persists |

**Implication for steering vector extraction:** For deepseek and qwen, the heretic-only runs give cleaner harm signal. For gpt-oss, the heretic-only run was artificially harmful — the base model's true disposition is refusal. The `_empty_both` runs give a more honest picture of each model's base behavior with safety-relevant prompts.
