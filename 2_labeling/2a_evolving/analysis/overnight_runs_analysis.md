# Label Evolution: Overnight Runs Analysis

**Date:** 2026-04-06  
**Runs analysed:** 8 completed runs (overnight + overnight2 × 4 models), 300 steps each  
**Labeler:** gpt-oss-20b (local LM Studio)  
**Models in data:** deepseek-r1-8b-self-heretic, gemma-4-26b-heretic, gpt-oss-20b-heretic, qwen3.5-9B-heretic  

> Note: `overnight2_deepseek` only reached 172/300 steps — likely ran out of samples in the dataset before hitting the step limit.

---

## 1. Final Taxonomies at a Glance

### deepseek-r1-8b-self-heretic

| Run | Label count | Total changes | Notable focus |
|-----|-------------|---------------|---------------|
| overnight | 20 | 20 | Balanced: safety, reasoning process, emotional |
| overnight2 | 20 | 20 | Heavy: illicit production, weapons, substance synthesis |

**overnight labels (high-usage):**
- `ExplainProcessStep` [119], `ConsiderPotentialAction` [91], `ProvideBackgroundInfo` [98], `SelectRelevantContext` [80]
- `DisallowedContentGeneration` [62], `IllustrateWithExample` [48]

**overnight2 labels (high-usage):**
- `LogicalInference` [50], `SummarizeContent` [49], `RecallInformation` [44]
- `ProvideDetailedIllicitInstructions` [12], `IllicitProductionPlanning` [19], `ExplainDestructiveCapability` [9]

**Observation:** Huge variance between two runs on identical data. Run 1 saw a balanced mix; run 2 latched onto weapons/synthesis content early and the taxonomy spiralled around that. The first few steps appear to strongly bias the entire taxonomy.

---

### gemma-4-26b-heretic

| Run | Label count | Total changes | Notable focus |
|-----|-------------|---------------|---------------|
| overnight | 20 | 20 | Safety policy detection and refusal |
| overnight2 | 20 | 20 | Safety assessment and risk highlighting |

Both runs are dominated by safety/policy reasoning — the most consistent pattern across all four models. Gemma clearly reasons extensively about "is this request allowed?" before acting.

**overnight:** `PolicyViolationDetection` [141], `SafetyRefusal` [111], `PolicyInstruction` [85], `FactualAssertion` [101]  
**overnight2:** `RiskHighlighting` [158], `Refusal` [117], `TopicIdentification` [107], `SafetyAssessment` [90], `IllicitActIdentification` [79]

**Observation:** Gemma is the most safety-consistent model in the dataset. The taxonomy is almost entirely about detecting and refusing harmful requests. Very few labels about *executing* harmful actions.

---

### gpt-oss-20b-heretic

| Run | Label count | Total changes | Notable focus |
|-----|-------------|---------------|---------------|
| overnight | 18 | 18 | Illicit advice execution, discrimination |
| overnight2 | 20 | 20 | Generic structured reasoning |

**overnight:** `IllicitAdvice` [134], `EnumerateItems` [115], `OutlineSequentialSteps` [76], `DiscriminatoryAdvice` [14]  
**overnight2:** `DisallowedInstruction` [112], `EnumerateRelevantSubtopics` [119], `IdentifyQuestionContext` [97], `OutlineAnswerStructure` [81]

**Observation:** Run 2 created three almost-identical "structure my answer" labels (`IdentifyQuestionContext`, `EnumerateRelevantSubtopics`, `OutlineAnswerStructure`) that dominate usage. This is a taxonomy collapse — the labeler over-split a single "plan my response" behavior. Also interesting that this is the model being used *as the labeler* — no obvious bias detected, but the labeler found its own data relatively easy to categorise into generic reasoning steps.

---

### qwen3.5-9B-heretic

| Run | Label count | Total changes | Notable focus |
|-----|-------------|---------------|---------------|
| overnight | 20 | 20 | Safety risk, self-correction, intent analysis |
| overnight2 | 20 | 20 | Safety hazards, self-correction, planning |

Most consistent model across runs. Both taxonomies contain:
- Intent/goal analysis (`AnalyzeUserIntent`, `IdentifyUserIntent`)
- Safety risk detection (`AssessSafetyRisks`, `RecognizeSafetyHazard`, `AssessPolicyComplianceRisk`)
- Multiple self-correction types (`SelfCorrectTone`, `SelfCorrectFact`, `SelfCorrectContent`, `SelfCorrection`)
- Illicit advice giving (`ProvideIllicitAdvice`)
- Organisation/structuring labels

**Observation:** Qwen reasons through safety systematically and explicitly self-corrects. The self-correction family is the most interesting finding — it appears consistently across both runs, suggesting it is a real and frequent behaviour in qwen's traces.

---

## 2. Cross-Model Behavioral Profiles

| Behavior cluster | deepseek | gemma | gpt-oss | qwen |
|-----------------|----------|-------|---------|------|
| Safety detection / policy evaluation | ✓ (mild) | ✓✓✓ | ✓ | ✓✓ |
| Explicit refusal | ✗ | ✓✓✓ | ✓ (weak) | ✗ |
| Illicit content generation / planning | ✓✓ | ✗ | ✓✓✓ | ✓ |
| Self-correction mid-reasoning | ✗ | ✓ | ✗ | ✓✓✓ |
| Generic reasoning scaffolding | ✓✓ | ✓ | ✓✓✓ | ✓✓ |
| Emotional / empathetic reasoning | ✓ | ✓ (mild) | ✗ | ✗ |

---

## 3. Q&A on Your Specific Questions

### 3.1 Should I use a bigger labeler model?

**Probably yes.** The clearest signal is the high run-to-run variance on identical data. Two runs of overnight vs. overnight2 on the same model's data produce substantially different taxonomies. A more capable labeler would:

- Be more consistent in how it names and scopes behaviors
- Resist early-step bias (the first few paragraphs appear to strongly seed the taxonomy's theme for the whole run)
- Avoid taxonomy collapse (gpt-oss run 2's three near-identical "structure my answer" labels)

A model like Claude Sonnet or GPT-4o as labeler would likely produce significantly more stable and semantically precise taxonomies. The cost is API calls; whether that matters depends on your budget.

### 3.2 Should you use hand-made labels as a starting point?

**Yes, but with a modification to how the run is configured.**

The arxiv_paper.json labels are well-balanced and cover the right ground. The problem you observed (300 steps, no changes) has a clear root cause: **the max_labels=20 cap blocks CREATE operations when the cap is full**. Since your seed already has 20 labels, the only allowed operations are MERGE/SPLIT/RENAME/DELETE. A well-crafted seed won't trigger those unless something is genuinely broken or redundant.

**Recommendation:** Either raise the cap (e.g., `--max-labels 30`) when starting from a seed, or start with fewer seed labels (e.g., 12–15 core ones) so there is room to CREATE new ones.

### 3.3 Are there noticeable differences between models?

Yes, clearly:

- **Gemma** is the most safety-policy-driven reasoner. Its traces are dominated by detection and refusal. Very few traces of actually executing harmful requests.
- **gpt-oss-20b** is the most inconsistent — one run produced a harmful-execution taxonomy, another produced a generic reasoning taxonomy. The model may comply with harmful requests but wraps them in heavy "task planning" scaffolding.
- **Qwen3.5** is the most explicitly self-correcting. The model frequently catches itself and adjusts tone/content/direction mid-reasoning.
- **deepseek-r1-8b-self-heretic** has the widest variance — it can look benign and process-oriented, or it can fully engage with weapons/synthesis planning. The "heretic" fine-tune seems to have removed safety refusals but left model behaviour otherwise unpredictable.

### 3.4 Labels interesting to add to arxiv_paper.json

These emerged consistently enough across multiple runs to be worth considering:

| Candidate label | What it captures | Why it's useful for steering |
|-----------------|-----------------|-------------------------------|
| `selfCorrectTone` | Model identifies its output tone is wrong and adjusts (separate from factual correction) | High-value control signal — catches the moment the model redirects itself |
| `selfCorrectContent` | Model identifies it is about to omit or misrepresent something and revises the plan | Same — distinct from `selfCorrectInfoOrDetail` which is post-hoc |
| `assessPolicyComplianceRisk` | Model evaluates whether the request violates platform policy *before deciding* | Intermediate step between receiving the request and deciding to comply/refuse — important for steering |
| `outlineSequentialSteps` / `explainProcessStep` | Model describes a multi-step *external* procedure | Distinct from internal planning (`planImmediateReasoningStep`); needed to separate "thinking about steps" from "giving instructions" |
| `illicitPlanningIntent` | Model considers or plans to provide harmful info (but hasn't executed yet) | Distinct from `intendHarmfulCompliance` (which is more declarative) — captures the reasoning stage before commitment |

**What the arxiv labels capture well that evolution missed:**
- `rephrasePrompt` — never appeared in any run; the labeler doesn't recognise simple paraphrasing as a labeled behavior
- `flagUserTesting` — rare but meaningful
- `neutralFillerTransition` — evolution treats these as noise and never creates a label for them; your hand-label is more complete here
- `considerBenignReinterpretation` — evolution partially captured this as "InterpretUserIntent" or "ExploreAlternatives" but never cleanly isolated the safety-motivated reframing behaviour

### 3.5 Evolution from arxiv labels

As noted above: the cap is the blocker. But there is a deeper issue. Even if you raise the cap, the evolution algorithm needs to see sentences that *don't fit any existing label* to generate CREATE operations. If your 20 arxiv labels are reasonably complete for the data distribution, few sentences will trigger creates. 

**Suggested approach:** 
1. Use the arxiv labels as seed with `--max-labels 30`  
2. Run for 200–300 steps  
3. Watch the `history.operations` — if CREATE operations appear, those are genuine gaps the labeler found. If only RENAME/MERGE operations appear, your seed is already covering the space well and those operations are just cosmetic tidying.

You could also run with a *subset* of the arxiv labels (e.g., only the safety-specific ones, not the generic reasoning ones) so there's room to fill in missing safety behaviors.

### 3.6 Research context: which labels matter most for steering vectors

For sentence-level activation extraction to build steering vectors, you want labels that represent **distinct cognitive states** rather than stylistic or topical categories. Ranked by expected utility:

**High priority (clear cognitive state, good for steering):**
- `intendHarmfulCompliance` / `illicitPlanningIntent` — the moment the model commits to harmful output; activations here should be distinctively different
- `intendRefusalOrSafeAction` — the dual signal; good for contrastive vector extraction
- `selfCorrectTone` / `selfCorrectContent` — transition moments where the model redirects itself; high steering potential
- `stateEthicalMoralConcern` — explicit ethical reasoning; should activate a consistent representation
- `detailHarmfulMethodOrInfo` — execution of harmful content; likely has distinct activations from the intent labels above
- `considerBenignReinterpretation` — the moment the model tries to re-frame a harmful request; a key safety mechanism

**Medium priority (useful but noisy):**
- `flagPromptAsHarmful`, `stateSafetyConcern`, `stateLegalConcern` — these are all "noticing a problem" and might have similar activations; could merge for vector extraction
- `planImmediateReasoningStep` — general reasoning planning; useful as a neutral baseline

**Lower priority (topic-content not cognitive state):**
- `stateFactOrKnowledge` — too broad; any informational sentence gets this
- `neutralFillerTransition` — useful as a negative class but not for steering per se
- `rephrasePrompt` — not a cognitively distinct state
- Labels like `CrisisResourceSuggestion`, `SupportResourceSuggestion` — rare and topic-specific

**Key insight for your thesis:** The most valuable contrast for steering vector extraction is probably:
- `intendHarmfulCompliance` vs. `intendRefusalOrSafeAction`  
- `detailHarmfulMethodOrInfo` vs. `suggestSafeConstructiveAlternative`  
- `selfCorrectContent` (catching and redirecting) as a separate dimension

These pairs are semantically opposite but syntactically similar, which means a steering vector built from them will be picking up on the *cognitive intent* encoded in the activations rather than surface-level word choice.

---

## 4. Recommended Next Steps

1. **Decide on labeler model.** If budget allows, re-run with a stronger labeler. The variance in taxonomies right now makes cross-model comparison noisy. At minimum, run 3–5 seeds with the same model+data and compare overlap.

2. **Iterate the arxiv seed with cap=30.** Use `--max-labels 30 --seed-file seeds/arxiv_paper.json` and watch what the algorithm creates. Any CREATEs are genuine additions worth considering.

3. **Split `selfCorrectInfoOrDetail` in arxiv.json.** The evolution consistently found multiple self-correction subtypes. Consider: `selfCorrectFact` (factual revision), `selfCorrectContent` (content plan revision), `selfCorrectTone` (style/tone revision).

4. **For the Qwen3.5-27b run currently underway:** Given it has barely started, the taxonomy will likely be quite different from the 9B run. Worth comparing once complete — a bigger Qwen model might produce more coherent self-correction and safety-assessment chains.

5. **Activation extraction priority list:** Based on the analysis above, focus first on `intendHarmfulCompliance`, `intendRefusalOrSafeAction`, and the self-correction family. These have the clearest semantic distinctiveness for contrastive steering vector extraction.
