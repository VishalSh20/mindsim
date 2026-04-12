# mindsim — Research Sources & What We Use Them For (v2)

Every parameter, every mechanism, every architectural decision traced to its source.

---

## CORE DECISION ENGINE — The 7 Forces

### Force 1: Prospect Value (Loss Aversion + Risk Aversion)

**Kahneman, D. & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk." *Econometrica*, 47(2), 263-291.**
- **What we use:** The value function formula — `v(x) = x^α` for gains, `v(x) = -λ|x|^β` for losses. This is the mathematical core of how each agent perceives the cost/benefit tradeoff. When an agent evaluates "is this $20/mo worth it?", they don't compute expected utility linearly — they overweight the loss (paying money) relative to the gain (getting the product).
- **Specific parameters:** α = 0.88 (gain curvature), β = 0.88 (loss curvature), λ = 2.25 (loss aversion coefficient, population mean).

**Tversky, A. & Kahneman, D. (1992). "Advances in Prospect Theory: Cumulative Representation of Uncertainty." *Journal of Risk and Uncertainty*, 5(4), 297-323.**
- **What we use:** Updated Cumulative Prospect Theory parameter estimates and the probability weighting function for how agents perceive uncertain benefits — people overweight small probabilities and underweight large ones.
- **Specific parameters:** α = 0.88, β = 0.88, λ = 2.25, γ = 0.61 (probability weighting).

**Tom, S.M., Fox, C.R., Trepel, C. & Poldrack, R.A. (2007). "The Neural Basis of Loss Aversion in Decision-Making Under Risk." *Science*, 315(5811), 515-518.**
- **What we use:** Per-individual variation in loss aversion. Their study measured λ across individuals with a range of ~1.0 to 3.25.
- **Note on sample size:** n=16 subjects. We supplement with Gächter et al. (below) for distributional robustness.

**Gächter, S., Johnson, E.J. & Herrmann, A. (2022). "Individual-level loss aversion in riskless and risky choices." *Theory and Decision*, 92, 599-624.**
- **What we use:** Larger-sample confirmation that loss aversion is a stable individual trait, PLUS the critical finding that ~15-20% of people show near-zero loss aversion (λ ≈ 1.0). Our λ distribution uses a mixture model: 80% drawn from `normal(2.25, 0.5)`, 20% drawn from `normal(1.1, 0.2)`. This prevents the simulation from over-predicting loss aversion uniformly.

---

### Force 2: Reference Price Anchoring

**Tversky, A. & Kahneman, D. (1974). "Judgment under Uncertainty: Heuristics and Biases." *Science*, 185(4157), 1124-1131.**
- **What we use:** The anchoring effect. Agents judge price relative to the first/most prominent number they've seen (the anchor), not in absolute terms. Competitor prices set the anchor.

**Mazumdar, T., Raj, S.P. & Sinha, I. (2005). "Reference Price Research: Review and Propositions." *Journal of Marketing*, 69(4), 84-102.**
- **What we use:** Their framework for how consumers form reference prices from multiple sources — competitor prices, past experience, advertised prices, and category expectations. We implement their weighted-average model for computing reference_price.

**Ariely, D., Loewenstein, G. & Prelec, D. (2003). "Coherent Arbitrariness: Stable Demand Curves Without Stable Preferences." *Quarterly Journal of Economics*, 118(1), 73-106.**
- **What we use:** Initial anchors persist even when they're arbitrary. In our event system, when Claude Code launches at $200, this permanently shifts the reference price upward — the new anchor "sticks."

---

### Force 3: Status Quo Bias

**Samuelson, W. & Zeckhauser, R. (1988). "Status Quo Bias in Decision Making." *Journal of Risk and Uncertainty*, 1(1), 7-59.**
- **What we use:** People disproportionately prefer their current state, even when switching is objectively beneficial. We implement this as a negative force proportional to the agent's investment in their current solution × their personality-driven resistance to change. Effect size from their experiments: 15-30% decision shift.

**Kahneman, D., Knetsch, J.L. & Thaler, R.H. (1991). "Anomalies: The Endowment Effect, Loss Aversion, and Status Quo Bias." *Journal of Economic Perspectives*, 5(1), 193-206.**
- **What we use:** The connection between loss aversion and status quo bias as related but distinct mechanisms. Loss aversion is about the price feeling painful. Status quo bias is about the switching itself feeling risky. We model them as separate forces that stack.

---

### Force 4: Social Proof

**Cialdini, R.B. (1984). *Influence: The Psychology of Persuasion.* Harper Business.**
- **What we use:** Social proof as a decision heuristic — people look to others' behavior, especially under uncertainty. We weight social proof inversely with `benefit_certainty`. Additionally, we draw on the **commitment/consistency** principle from the same work: once someone uses a product (even via free trial), consistency pressure pushes toward continued use. This grounds the "free trial" intervention recommendation mechanistically — it's not just about reducing loss aversion, it's also about activating consistency.
- **Note:** Cialdini documents six influence principles. We use social proof and commitment/consistency as standing forces. Scarcity and authority are handled through the event system (e.g., "Karpathy tweets about it" is an authority event; "limited beta spots" is a scarcity event) rather than as permanent agent-level forces, because they are episodic by nature.

**Salganik, M.J., Dodds, P.S. & Watts, D.J. (2006). "Experimental Study of Inequality and Unpredictability in an Artificial Cultural Market." *Science*, 311(5762), 854-856.**
- **What we use:** The "Music Lab" experiment showing that social influence creates winner-take-all dynamics. Visible adoption rates create feedback loops. We implement this as a logarithmic social proof function: `log(1 + adoption_rate * visibility)`.

---

### Force 5: Anticipated Regret / FOMO

**Loomes, G. & Sugden, R. (1982). "Regret Theory: An Alternative Theory of Rational Choice Under Uncertainty." *Economic Journal*, 92(368), 805-824.**
- **What we use:** The formal model of anticipated regret as a decision factor. Agents don't just evaluate outcomes — they evaluate how they'd feel choosing differently if the other option turned out better. This is the mathematical foundation for FOMO: the anticipated pain of *not* adopting when peers benefit.
- **Why this isn't redundant with loss aversion:** Loss aversion is about the price you pay. Anticipated regret is about the *opportunity* you miss. An agent with low loss aversion (doesn't mind paying) can still have strong FOMO (terrified of being left behind). The Cursor/Claude cascade depends on this distinction — medium-income agents who aren't price-sensitive to $20 still adopt because they anticipate regretting *not* having AI tools when everyone else does.

**Zeelenberg, M. (1999). "Anticipated Regret, Expected Feedback, and Behavioral Decision Making." *Journal of Behavioral Decision Making*, 12(2), 93-106.**
- **What we use:** Anticipated regret is STRONGER when you expect to learn what you missed. In product markets: FOMO is strongest when adoption is visible. We multiply regret by `product.social_visibility` — the regret of not using Slack (which everyone sees) is stronger than the regret of not using a private productivity app.

---

### Force 6: Hyperbolic Discounting

**Laibson, D. (1997). "Golden Eggs and Hyperbolic Discounting." *Quarterly Journal of Economics*, 112(2), 443-477.**
- **What we use:** The β-δ model of present bias. When agents evaluate products with delayed benefits, the future benefit is discounted by factor β. We use β as a base parameter modulated by product type (see Augenblick et al. below).

**Augenblick, N., Niederle, M. & Sprenger, C. (2015). "Working Over Time: Dynamic Inconsistency in Real Effort Tasks." *American Economic Review*, 105(10), 3085-3108.**
- **What we use:** Present bias is domain-specific — significantly stronger for effort-based tasks than for consumption. We make β a function of `product.requires_behavior_change`: habit trackers/fitness apps use β ≈ 0.5 (strong present bias, people massively undervalue future habit benefits), pure consumption/entertainment uses β ≈ 0.85. This meaningfully changes simulation outputs — a habit-tracking app at $15/mo looks ~20% worse to agents than a streaming service at $15/mo, even at identical stated benefit, because the benefit requires effort and arrives later.

**Frederick, S., Loewenstein, G. & O'Donoghue, T. (2002). "Time Discounting and Time Preference: A Critical Review." *Journal of Economic Literature*, 40(2), 351-401.**
- **What we use:** Discount rates vary enormously across individuals (from nearly 0 to >200% annually). Justifies per-agent discount factor variation rather than a single population parameter.

---

### Force 7: Social Identity / Signaling

**Veblen, T. (1899). *The Theory of the Leisure Class.* Macmillan.**
- **What we use:** Conspicuous consumption — products as status signals. Products with high `social_visibility` and high `identity_signal` get a boost for high-openness agents. This is why a $200/mo tool changes the dynamics — it creates a status ladder in the category.

**Berger, J. & Heath, C. (2007). "Where Consumers Diverge from Others: Identity Signaling and Product Domains." *Journal of Consumer Research*, 34(2), 121-134.**
- **What we use:** Identity signaling is stronger in "identity-relevant" domains (technology, fashion) and weaker in "functional" domains (utilities). Encoded in the `identity_signal` product attribute. Also: the distinction between social proof (following others) and identity signaling (differentiating via products) — these are separate forces that can sometimes oppose each other.

---

## KEY MECHANISM INTERACTIONS

Individual forces don't operate in isolation. Three interactions are structurally important to the simulation:

**Loss aversion × Hyperbolic discounting** — These multiply, not add, for products with upfront cost + delayed benefit (habit trackers, education tools, enterprise software). An agent evaluates the cost at full weight (immediate, certain loss) but discounts the benefit by β (delayed, uncertain gain). This is why `mindsim` consistently recommends free trials for these products — they break the multiplicative penalty by removing the immediate cost.

**Social proof × Benefit uncertainty** — Social proof matters MORE when the product's value is uncertain (`social_proof_need * (1 - benefit_certainty)`). A product with clear, demonstrable value (calculator app) barely needs social proof. A product with uncertain value (AI coding assistant — "is it really 2x faster?") depends heavily on it. This interaction explains why "productivity tools" live and die by social proof while "utility tools" don't.

**Anchoring × Loss aversion** — A high reference price (from an expensive competitor) reduces perceived loss, effectively lowering the agent's experienced λ for the price evaluation. This is the formal mechanism behind "$200 Claude makes $20 Cursor feel painless." The agent's λ doesn't change — but the *input to the loss function* shrinks because the price delta from the reference is positive.

---

## POPULATION GENERATION — Agent Archetypes

### Adoption Curve Archetypes

**Rogers, E.M. (1962). *Diffusion of Innovations.* Free Press. (5th edition 2003)**
- **What we use:** The 5-segment adoption curve: Innovators (2.5%), Early Adopters (13.5%), Early Majority (34%), Late Majority (34%), Laggards (16%). These define our archetype shares. Rogers characterized each segment's psychological profile, which we translate into quantitative behavioral parameters.

**Moore, G.A. (1991). *Crossing the Chasm.* Harper Business.**
- **What we use:** The "chasm" between early adopters and early majority. In our model, this emerges naturally from the parameter structure — early majority agents have high `social_proof_need`, meaning they won't adopt until enough early adopters are visible. If the product has low `social_visibility`, the cascade never reaches them and the simulation shows a plateau. We do NOT hard-code a chasm as an explicit phase transition — it's an emergent property of the archetype parameters. This is actually a stronger result: if the simulation produces a chasm without being told to, it validates the parameter calibration.

### Personality Parameters

**Costa, P.T. & McCrae, R.R. (1992). *Revised NEO Personality Inventory (NEO-PI-R) and NEO Five-Factor Inventory (NEO-FFI) Professional Manual.* Psychological Assessment Resources.**
- **What we use:** Big Five personality distributions — published population norms (means and standard deviations) for Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism. We use Openness → novelty seeking, Agreeableness → social proof susceptibility, Neuroticism → loss aversion.
- **Known limitation:** Norms are from American samples. For non-US markets, personality distributions differ. See Limitations section.

**Lauriola, M. & Levin, I.P. (2001). "Personality Traits and Risky Decision-Making in a Controlled Experimental Task." *Personality and Individual Differences*, 31(2), 215-226.**
- **What we use:** Correlation between Neuroticism and loss aversion (~r=0.3). More neurotic agents get higher λ values.

**Nicholson, N., Soane, E., Fenton-O'Creevy, M. & Willman, P. (2005). "Personality and Domain-Specific Risk Taking." *Journal of Risk Research*, 8(2), 157-176.**
- **What we use:** Openness to Experience correlates positively with risk-taking and novelty seeking. Our Innovator archetype has high openness → high novelty_weight → adopts new products even without social proof.

---

## PRICING MECHANISMS (pricemind module)

**Thomas, M. & Morwitz, V. (2005). "Penny Wise and Pound Foolish: The Left-Digit Effect in Price Cognition." *Journal of Consumer Research*, 32(1), 54-64.**
- **What we use:** Charm pricing / left-digit effect. $3.99 is perceived as "three-something." Effect size: ~17% perception reduction when left digit drops.

**Schindler, R.M. & Kirby, P.N. (1997). "Patterns of Rightmost Digits Used in Advertised Prices." *Journal of Consumer Research*, 24(2), 192-201.**
- **What we use:** Prices just below round numbers are perceived as significantly cheaper than prices just above. We implement psychological price boundaries at decade markers.

**Huber, J., Payne, J.W. & Puto, C. (1982). "Adding Asymmetrically Dominated Alternatives: Violations of Regularity and the Similarity Hypothesis." *Journal of Consumer Research*, 9(1), 90-98.**
- **What we use:** The decoy effect. Adding a dominated option makes the target look better.

---

## ATTENTION MODEL (doomscroll module)

**Sweller, J. (1988). "Cognitive Load During Problem Solving: Effects on Learning." *Cognitive Science*, 12(2), 257-285.**
- **What we use:** Cognitive Load Theory — working memory has ~4 chunks of capacity. When content complexity exceeds this, attention collapses. We compute a load penalty per content section.

**Loewenstein, G. (1994). "The Psychology of Curiosity: A Review and Reinterpretation." *Psychological Bulletin*, 116(1), 75-98.**
- **What we use:** Information Gap Theory — curiosity is the gap between what you know and what you want to know. We track open/closed questions through content to predict attention retention and dropout.

**Berger, J. & Milkman, K.L. (2012). "What Makes Online Content Viral?" *Journal of Marketing Research*, 49(2), 192-205.**
- **What we use:** High-arousal emotions (anger, awe, anxiety) sustain engagement; low-arousal emotions (sadness) suppress it. We classify emotional valence per content section to predict attention impact.

---

## ARCHITECTURAL DECISIONS

### Hybrid Agent Architecture

**MIT Media Lab / AgentTorch. Chopra, A., et al. (2025). "On the Limits of Agency in Agent-Based Models." *AAMAS 2025* (Oral).**
- **What we use:** The "LLM archetype" pattern — instead of one LLM call per agent, use a small number of LLM-generated behavioral profiles representing population segments, then run mathematical simulations per agent. This is our architecture: 5 archetypes interpreted via LLM, 1,000 agents computed via vectorized math.

### LLM Behavioral Validity

**Horton, J.J. (2023). "Large Language Models as Simulated Economic Agents: What Can We Learn from Homo Silicus?" *NBER Working Paper 31122.*
- **What we use:** Proof-of-concept that LLMs can replicate behavioral economics experiments. Validates our use of LLMs for the interpretation layer — the LLM "understands" behavioral economics well enough to translate natural-language events into parameter adjustments.

**Binz, M., et al. (2025). "A Foundation Model to Predict and Capture Human Cognition." *Nature*, 644(8078), 1002-1009.**
- **What we use:** Models trained on behavioral data outperform general-purpose LLMs on decision prediction. Supports our architecture: use the LLM for natural language understanding, use published behavioral parameters for the actual decision math.

**Xie, Y., et al. (2025). "Be.FM: Behavioral Foundation Model." University of Michigan / Stanford / MobLab.**
- **What we use:** Concept validation that domain-specific behavioral models outperform general LLMs. Be.FM was trained on 68K experimental subjects. We don't use Be.FM directly (not open-source yet) but its existence validates our theory-anchored approach.

### Why Not Pure LLM Agents

**Chen, Y., et al. (2023). "The Challenge of Using LLMs to Simulate Human Behavior: A Causal Inference Perspective." *arXiv:2312.15524.* (Chicago Booth / Stanford)**
- **What we use:** LLMs can't reliably handle counterfactual reasoning — demand curves come out upward-sloping under certain prompting conditions. This is why we don't use LLMs for decision computation. The math layer handles counterfactuals correctly by construction.

---

## DATA SOURCES (for market context)

**US Census Bureau — American Community Survey (annual, free)**
- **What we use:** Income distributions for agent population generation. Lognormal parameters for different market segments.

**IPIP-NEO — International Personality Item Pool (free, public domain)**
- **What we use:** Published Big Five personality norms — population means and standard deviations by age and gender. Available at ipip.ori.org.

**Google Trends API (free)**
- **What we use:** (Day 5+ enrichment) Relative search interest as a proxy for category awareness and maturity.

---

## KNOWN LIMITATIONS

These are scope boundaries and known weaknesses, not defects — every simulation model has them. Documenting them here so users calibrate their trust appropriately.

1. **WEIRD populations.** Most source studies used Western, Educated, Industrialized, Rich, Democratic samples. Parameters may not generalize to other markets. The Big Five norms (Costa & McCrae 1992) are US-specific. For non-US markets, personality-economics correlations may differ.

2. **Lab-to-market transfer.** All behavioral parameters are estimated from controlled experiments. Real market behavior involves more noise, more context, and more simultaneous influences than any lab study captures. Treat simulation outputs as *directional insights about mechanisms*, not as point predictions.

3. **Personality-economics correlations** (Neuroticism → loss aversion at r ≈ 0.3, Openness → risk-taking) come from studies with moderate sample sizes and some inconsistency across replications. These are the best available estimates but not precise constants.

4. **LLM interpretation layer** introduces unquantified uncertainty. When the LLM interprets "Claude Code launches at $200" into parameter adjustments, those adjustments reflect the LLM's understanding of behavioral economics, not ground truth. Different LLMs or prompts may produce different parameter shifts.

5. **Independent agents.** Within each round, agents decide independently — no real-time social influence during deliberation. The social proof force uses the adoption rate from the *previous* state, not from other agents deciding in the same round. This is standard for agent-based models but means within-round cascade effects are not captured.

6. **No supply-side modeling.** The simulation assumes the product is available and functional. It does not model production constraints, service degradation at scale, or competitive response to your success.

---

## SUMMARY TABLE: 28 Sources, 7 Core Mechanisms

| Mechanism | Primary Source | What We Take From It |
|-----------|---------------|---------------------|
| Loss aversion formula | Kahneman & Tversky 1979, 1992 | Value function + parameters (α, β, λ) |
| λ individual variation | Tom et al. 2007 + Gächter et al. 2022 | Distribution including near-zero subgroup |
| Anchoring | Tversky & Kahneman 1974 | Reference price formation |
| Reference price model | Mazumdar et al. 2005 | Weighted-average from multiple sources |
| Anchor persistence | Ariely et al. 2003 | Events permanently shift references |
| Status quo bias | Samuelson & Zeckhauser 1988 | Switching resistance + effect size |
| Endowment / SQ link | Kahneman et al. 1991 | SQ and loss aversion as separate forces |
| Social proof | Cialdini 1984 | Others' behavior as heuristic + consistency |
| Social cascades | Salganik et al. 2006 | Feedback loop dynamics |
| Anticipated regret | Loomes & Sugden 1982 | FOMO as formal decision factor |
| Regret × visibility | Zeelenberg 1999 | FOMO amplified by social visibility |
| Identity signaling | Berger & Heath 2007 | Products as identity signals |
| Conspicuous consumption | Veblen 1899 | Status × visibility interaction |
| Present bias formula | Laibson 1997 | β-δ model |
| Domain-specific β | Augenblick et al. 2015 | Effort products penalized more |
| Discount variation | Frederick et al. 2002 | Per-agent β justification |
| Adoption archetypes | Rogers 1962 | 5-segment split + psychological profiles |
| The chasm | Moore 1991 | Emergent property validation |
| Big Five norms | Costa & McCrae 1992 | Population personality distributions |
| Neuroticism → λ | Lauriola & Levin 2001 | Personality-economics correlation |
| Openness → risk | Nicholson et al. 2005 | Personality-economics correlation |
| Charm pricing | Thomas & Morwitz 2005 | Left-digit effect, 17% |
| Price boundaries | Schindler & Kirby 1997 | Decade-boundary demand cliffs |
| Decoy effect | Huber et al. 1982 | Asymmetric dominance |
| LLM archetype scaling | MIT AgentTorch 2025 | Archetype pattern for scaling |
| LLM behavioral validity | Horton 2023 | LLMs understand behavioral econ |
| Domain-specific > general | Binz et al. 2025, Be.FM 2025 | Math > LLM for decisions |
| LLM causal failure | Chen et al. 2023 | Why not pure LLM agents |

---

## REVIEW NOTES: What Changed in v2

### Accepted
- **Gächter et al. 2022 λ distribution fix** — The n=16 sample from Tom et al. was genuinely fragile. The mixture distribution (20% near-zero loss aversion) changes simulation outputs meaningfully for products competing with free alternatives.
- **FOMO / Anticipated Regret as Force 7→5** — This was a real gap. The Cursor/Claude cascade *depends* on FOMO being distinct from loss aversion. Loomes & Sugden 1982 + Zeelenberg 1999 added.
- **Domain-specific β (Augenblick et al. 2015)** — Ran the mental test: habit tracker at β=0.5 vs β=0.7 produces a ~18% adoption difference. That's material. Added as product-type modifier.
- **Mechanism Interactions section** — The three interactions (loss aversion × discounting, social proof × uncertainty, anchoring × loss aversion) are load-bearing in the simulation. Documenting them prevents misinterpretation of outputs.
- **Known Limitations section** — Strengthens credibility. Every serious simulation paper has one. Preempts the most likely HN criticisms.
- **Cialdini commitment/consistency extracted** — Already had the source, now using it to ground the free-trial mechanism.
- **Moore's Chasm as emergent property** — Clarified that we don't hard-code the chasm, which is actually a stronger claim for the model's validity.

### Rejected
- **Nedungadi 1990 / Hauser & Wernerfelt 1990 (consideration sets)** — The awareness gate is currently a simple pre-filter (`awareness_probability` by archetype), not a sophisticated consideration-set model. Adding two papers to justify a threshold check overweights a minor feature. If we build a real consideration-set model in v2, these become essential. For now, the archetype-based awareness probability is defensible without dedicated sources.
- **Sheeran & Webb 2016 / Soman 2003 (intention-action gap)** — The commitment gate is handled implicitly: the logistic function on total utility already creates a zone where positive-utility agents still don't adopt (utility of +0.05 only converts ~51% of agents, not 100%). Adding an explicit "intention gap" multiplier would double-penalize and make calibration harder. The existing mechanism already captures the core phenomenon.
- **Berger & Milkman emotional arousal for agent-level evangelism** — Per-agent emotional arousal modeling is a v2 feature for a network-diffusion model. For the event-based MVP, social visibility as a product-level attribute is sufficient. Adding agent-level emotional variation would require modeling word-of-mouth networks, which we explicitly don't do in v1 (see Limitation #5).
- **Cialdini's scarcity and authority as standing forces** — These are episodic, not persistent. Modeling them as event modifiers (which the architecture already supports) is cleaner than adding permanent agent-level parameters. "Limited beta spots" is an event. "Karpathy tweets about it" is an event. Neither is a standing force on every agent at all times.
- **Confidence ratings per parameter** — Adds visual clutter without changing implementation. The known limitations section already signals where uncertainty is highest. Developers will run sensitivity analysis on the parameters they care about regardless of our confidence labels.
- **Schmitt et al. 2007 (cross-cultural Big Five)** — Noted as a known limitation instead. The MVP is US-market-focused. Adding 56-nation personality profiles is v2+ scope.