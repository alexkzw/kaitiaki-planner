# Kaitiakitanga in Machine Learning: A Budget-Aware Approach to Fairness in Bilingual Retrieval-Augmented Generation

## Executive Summary

This report examines a capstone project that addresses fairness in AI systems serving both English and Te Reo Māori, New Zealand's indigenous language. Through the development of a budget-aware Retrieval-Augmented Generation (RAG) system, this research explores how resource allocation strategies can reduce performance disparities between high-resource (English) and low-resource (Māori) languages. The project reveals a critical finding: improving retrieval quality through semantic embeddings yields greater fairness gains than sophisticated budget allocation mechanisms alone. This work contributes to broader conversations about Māori data sovereignty, ethical AI design, and the technical pathways toward language justice in an increasingly automated world.

---

## 1. Motivation: Language Justice and Kaitiakitanga in AI Systems

### 1.1 The Low-Resource Language Problem

Natural language processing systems exhibit systematic performance disparities across languages, with low-resource languages—those with limited digital corpora, fewer speakers, or less commercial investment—consistently underperforming compared to high-resource counterparts like English (Joshi et al., 2020). Te Reo Māori, with approximately 185,000 speakers globally, exemplifies this challenge. While New Zealand's only official indigenous language received legal recognition through the Māori Language Act 1987, the digital revolution has created new frontiers of linguistic inequality. Contemporary AI systems, trained predominantly on English data, risk perpetuating what can be termed "algorithmic colonialism"—the extension of historical power imbalances into computational infrastructure (Couldry & Mejias, 2019).

### 1.2 Kaitiakitanga: Guardianship as Design Principle

The concept of kaitiakitanga—guardianship, stewardship, or protection—provides a culturally grounded framework for approaching this technical challenge. In tikanga Māori (Māori customary practices), kaitiakitanga encompasses the responsibility to protect and maintain the mauri (life force) of taonga (treasures), which explicitly includes te reo Māori itself (Waitangi Tribunal, 2011). This project adopts kaitiakitanga as both metaphor and method: just as traditional kaitiaki (guardians) actively manage resources to ensure equitable access and intergenerational wellbeing, a budget-aware RAG system can allocate computational resources to protect the integrity of Māori language interactions with AI.

The title "kaitiaki-planner" thus signals a dual commitment: technical innovation in resource allocation and cultural accountability in system design. This framing resists the common AI ethics trap of treating fairness as a post-hoc optimization constraint, instead embedding cultural values at the architectural level.

### 1.3 Research Questions

This capstone investigates three interrelated questions:

1. **Technical**: Can dynamic budget allocation in RAG systems reduce performance gaps between English and Te Reo Māori queries?
2. **Empirical**: What retrieval and allocation strategies most effectively support fairness across languages?
3. **Ethical**: How do different fairness interventions align with indigenous data sovereignty principles and broader theories of distributive justice?

---

## 2. Methods: Designing for Fairness Through Resource Allocation

### 2.1 System Architecture

The kaitiaki-planner implements a three-stage RAG pipeline:

1. **Query Processing**: The orchestrator (TypeScript-based) receives queries labeled with language (English/Māori) and complexity (simple/complex).
2. **Budget Planning**: Based on the experimental condition, the system allocates a retrieval budget (top_k parameter) determining how many documents are retrieved.
3. **Retrieval and Reranking**: A Python-based retriever uses semantic embeddings (paraphrase-multilingual-mpnet-base-v2) with keyword boosting to identify relevant passages.
4. **Answer Generation**: Claude (Anthropic's LLM) generates answers grounded in retrieved passages.

### 2.2 Three Experimental Conditions

The system implements three budget allocation strategies, each representing a distinct fairness philosophy:

**Condition 1: Uniform (Equality)**
- Allocation: top_k=5 for all queries
- Philosophy: Formal equality—identical resource distribution regardless of need
- Rationale: Baseline representing most current RAG systems

**Condition 2: Language-Aware (Equity)**
- Allocation: top_k=8 for Māori, top_k=5 for English
- Philosophy: Equity based on resource scarcity—additional support for structurally disadvantaged groups
- Rationale: Acknowledges that Māori queries face retrieval challenges due to smaller training data, fewer native embeddings, and less web documentation

**Condition 3: Fairness-Aware (Intersectional Equity)**
- Allocation: top_k=8 for Māori OR complex queries, top_k=5 for simple English queries
- Philosophy: Intersectional fairness—recognizing multiple axes of disadvantage (language, task difficulty)
- Rationale: Addresses compounded challenges when users face both language barriers and cognitively demanding queries

This design operationalizes the principle of manaakitanga (hospitality, care, support) by providing differentiated support based on structural barriers rather than treating all queries identically.

### 2.3 Evaluation Methodology

**Dataset**: 30 labeled queries (15 English, 15 Māori) covering New Zealand cultural and geographical topics (e.g., "Kei hea a Aotearoa?" / "Where is Aotearoa?"). Corpus: 44 Wikipedia-derived passages (22 per language).

**Metric**: Grounded Correctness (GC)—binary measure where Claude judges whether its generated answer correctly cites the gold-standard passage. This metric privileges citation accuracy over fluency, ensuring evaluation focuses on retrieval effectiveness rather than language model eloquence.

**Baseline**: Initial experiments used BM25 (keyword-based retrieval), achieving 100% for English but only 46.7% for Māori (performance gap: 53.3 percentage points).

**Improved System**: Replaced BM25 with multilingual semantic embeddings plus a 30% keyword boost when query terms match document IDs (e.g., query "Aotearoa" boosts score for doc_id "mi_aotearoa").

---

## 3. Results: When Quality Trumps Allocation

### 3.1 Quantitative Findings

**Overall Performance Improvement**:
- English: 100% → 100% (maintained perfect accuracy)
- Māori: 46.7% → 80.0% (+71% relative improvement)
- Performance gap: 53.3% → 20.0% (62% reduction in disparity)

**The Null Result—All Conditions Identical**:
Contrary to expectations, all three budget allocation strategies (uniform, language-aware, fairness-aware) produced identical results:
- Mean Grounded Correctness: 0.900 (27/30 queries) across all conditions
- English: 15/15 (100%) in all conditions
- Māori: 12/15 (80%) in all conditions
- Statistical tests: ANOVA F=0.000, p=1.0000; equivalence testing confirmed Δ=0.000

**English-Māori Gap**:
While reduced from baseline, a 20-point gap persists. Statistical testing revealed:
- t-test: p=0.0719 (marginally non-significant at α=0.05)
- Cohen's d=0.683 (medium effect size)
- Interpretation: Practical performance difference exists, but small sample size (n=15 per language) limits statistical power

### 3.2 Interpreting the Null Result: A Finding, Not a Failure

The identical performance across conditions initially appeared puzzling—why did allocating more retrieval budget to Māori queries (top_k=8) provide no advantage over uniform allocation (top_k=5)? Deeper analysis reveals this null result as a substantive finding with critical implications:

**Retrieval Quality as Bottleneck**: Once semantic embeddings with keyword boosting correctly ranked the gold-standard passage within the top 5 results, increasing the budget to top_k=8 provided no additional benefit. The correct passage was already visible to Claude; retrieving more documents introduced noise rather than signal.

**Implication for System Design**: This suggests a two-stage model of fairness interventions:
1. **Stage 1 (Critical)**: Achieve baseline retrieval quality through architecturally appropriate methods (embeddings over BM25, query expansion, cross-lingual models)
2. **Stage 2 (Conditional)**: Apply budget allocation strategies when retrieval quality is insufficient but additional documents contain relevant information

The null result thus reveals that **fairness-by-design** (choosing multilingual embeddings) proved more effective than **fairness-by-allocation** (dynamically adjusting top_k). Budget reallocation serves as crisis management; architectural quality is preventive care.

### 3.3 Persistent Challenges: The Three Failing Queries

Three Māori queries failed across all conditions, revealing limits of current approaches:
- mi_kaka_q1: "He aha te ingoa pūtaiao o te Kākā?" (What is the scientific name of the kākā?)
- mi_taupo_q1: "He aha te ingoa o te moana wai māori?" (What is the name of the freshwater lake?)
- mi_reo_maori_q1: "Nā wai i tae mai ki Aotearoa i te tau 1769?" (Who arrived in Aotearoa in 1769?)

These failures suggest deeper issues: either retrieval fails to surface the correct document, or Claude struggles with Māori-language grounding despite correct retrieval. Further diagnosis is needed to distinguish between retrieval errors and generation errors—a distinction with different remediation pathways.

---

## 4. Ethical, Cultural, and Societal Implications

### 4.1 Māori Data Sovereignty and Te Mana Raraunga

This project operates within the framework of Māori data sovereignty, articulated by Te Mana Raraunga (the Māori Data Sovereignty Network). Their principles assert Māori rights to control data about Māori communities, culture, and language (Te Mana Raraunga, 2018). Three principles particularly inform this work:

**Rangatiratanga (Authority/Self-Determination)**: Māori should govern how te reo Māori is represented in AI systems. This project's design choices—privileging bilingual performance, evaluating on culturally relevant queries about Aotearoa—represent a modest step toward centering Māori epistemologies. However, true rangatiratanga requires Māori communities not just as evaluation subjects but as co-designers, determining system priorities and acceptable trade-offs.

**Whakapapa (Relationships/Contextualization)**: Data must be understood in relational context. The corpus intentionally pairs English and Māori passages about shared referents (e.g., Tongariro/Mount Tongariro), acknowledging that these are not parallel but relational—the Māori text reflects mātauranga Māori (Māori knowledge systems), while English text reflects Pākehā epistemologies. RAG systems that treat these as interchangeable retrieval targets risk flattening important distinctions about knowledge authority.

**Kaitiakitanga (Guardianship)**: As discussed, the project's titular principle. Yet genuine kaitiakitanga extends beyond technical adequacy to asking: Who benefits from this system? Who is harmed? Current implementation improves Māori query performance from 46.7% to 80%, but still underperforms English. Is 80% acceptable? Under what theory of justice is systematic underperformance tolerable?

### 4.2 Ethical Philosophy: Beyond Equality to Capability

The three experimental conditions operationalize distinct theories of distributive justice:

**Utilitarian Equality (Uniform)**: Maximize average performance, treating all queries identically. This approach is efficient and simple but ignores structural inequalities. By distributing resources equally, it perpetuates existing advantage—those already well-served (English users) continue excelling, while disadvantaged groups (Māori users) receive insufficient support.

**Rawlsian Equity (Language-Aware)**: Prioritize improvements for the least advantaged group (Rawls, 1971). Allocating additional budget to Māori queries embodies the "maximin" principle—optimize for the worst-off position. This resonates with social justice frameworks but requires accepting tradeoffs: resources directed toward equity may reduce aggregate efficiency.

**Capabilities Approach (Fairness-Aware)**: Drawing from Amartya Sen and Martha Nussbaum, the capabilities approach asks not whether people have equal resources, but whether they have equal capability to achieve valuable outcomes (Sen, 1999). The fairness-aware condition operationalizes this by identifying multiple barriers (language, complexity) that constrain capability, then allocating resources to equalize effective capability rather than formal inputs.

The null result complicates this philosophical landscape: when all three approaches yield identical outcomes, the choice between them becomes moot in practice. This reveals a deeper insight—**fairness interventions are context-dependent**. Sophisticated allocation mechanisms matter when resources are scarce and quality is uneven; they become irrelevant when baseline quality meets user needs. The ethical imperative shifts from "how do we allocate resources fairly?" to "how do we ensure sufficient quality that allocation becomes unnecessary?"

### 4.3 Machine Consciousness and the Ethics of Linguistic Representation

While debates about machine consciousness typically focus on whether AI systems possess subjective experience or moral status (Chalmers, 1995), this project highlights a different dimension: the **ethics of representation** in systems that process language. Even if Claude lacks consciousness, it serves as an epistemological intermediary—shaping how users access knowledge about the world.

When a RAG system fails to retrieve correct Māori-language passages, it doesn't merely fail technically; it performs a kind of epistemic violence, reinforcing the false belief that information is less accessible in te reo Māori than English. This matters particularly for younger generations of Māori speakers: if digital systems consistently deliver better results in English, users face pragmatic pressure to abandon te reo, accelerating language loss.

The concept of **linguistic agency**—users' ability to interact with technology in their language of choice without penalty—thus becomes crucial. Fairness isn't just about percentage points on evaluation metrics; it's about whether computational infrastructure supports or undermines the vitality of indigenous languages. From this perspective, the 20-point gap between English and Māori represents not a technical deficiency but a social harm requiring urgent redress.

### 4.4 Cross-Cultural Implications and Lessons for Other Low-Resource Languages

While grounded in Aotearoa's specific context, this work has broader implications for the estimated 7,000 languages worldwide, most of which are "low-resource" from NLP's perspective (Joshi et al., 2020). Three lessons emerge:

**1. Architecture Matters More Than Tweaking**: Replacing BM25 with semantic embeddings yielded a 71% improvement for Māori retrieval. This architectural change—choosing models trained on broader linguistic data—proved far more effective than any allocation strategy. For language communities considering AI interventions, this suggests prioritizing investment in foundational models that respect linguistic diversity over downstream optimizations within systems already biased toward dominant languages.

**2. Measurement Shapes Outcomes**: The Grounded Correctness metric deliberately evaluates citation accuracy, not fluency. This choice reflects an understanding that retrieval quality precedes generation quality—a system that retrieves the wrong passage will generate a wrong answer, no matter how fluently. Other evaluation contexts might prioritize different values: some communities might emphasize cultural appropriateness of phrasing, others might focus on preservation of traditional metaphors. Fairness metrics must be co-designed with communities to reflect their priorities.

**3. The "Good Enough" Trap**: An 80% success rate for Māori queries might seem acceptable—good enough to deploy, certainly better than 46.7%. But "good enough" thinking entrenches inequality when one group (English users) receives 100% performance. The remaining 20-point gap represents structural unfairness: some users enjoy perfect service while others experience regular failures. Acceptability cannot be judged in isolation but only in comparison to the best performance achieved for any group.

### 4.5 Power, Participation, and the Limits of Technical Solutions

Finally, this project illuminates the limits of technical interventions divorced from structural change. Even a perfectly fair RAG system operates within broader inequalities: most training data comes from English sources; most AI research papers are published in English; most companies hiring NLP engineers expect English-language education credentials. Improving one system's Māori performance, while valuable, cannot substitute for systemic efforts:

- **Data Justice**: Investing in Māori-language corpus development, oral history digitization, and indigenous-language web content
- **Educational Justice**: Funding Māori-language immersion education, te reo Māori teacher training, and bilingual STEM resources
- **Economic Justice**: Supporting Māori-led tech companies, indigenous language technology grants, and procurement policies favoring bilingual AI systems

The principle of whanaungatanga (kinship, relationship) reminds us that technology exists within social webs. A RAG system cannot save a language; only communities can, through intergenerational transmission, cultural revitalization, and political power. Technology can support or hinder these efforts, making the question of algorithmic fairness not merely technical but deeply political.

---

## 5. Conclusion: Toward Language Justice in AI Systems

This capstone demonstrates that meaningful fairness improvements in bilingual RAG systems require architectural investment in retrieval quality, not merely clever resource allocation. By improving Māori query performance from 46.7% to 80% through semantic embeddings and keyword boosting, the system substantially reduced but did not eliminate the performance gap with English. The null result—all three budget allocation strategies performing identically—reveals that when retrieval quality reaches a threshold, additional budget provides no marginal benefit.

From a kaitiakitanga perspective, this suggests a clear mandate: those building AI systems for multilingual contexts must prioritize foundational model quality that respects linguistic diversity. Budget allocation strategies serve as contingency measures, useful when quality falls short but insufficient as primary fairness mechanisms.

The project also highlights unresolved challenges. The persistent 20-point gap between English and Māori performance, though statistically marginal (p=0.0719), remains practically significant. Three Māori queries failed consistently across all conditions, pointing to either retrieval failures or limitations in Claude's ability to ground answers in Māori-language passages. Future work must diagnose these failures and develop targeted interventions—potentially including:

- **Query expansion**: Automatically augmenting Māori queries with related terms to improve retrieval recall
- **Cross-lingual retrieval**: Allowing Māori queries to retrieve English passages when Māori passages are unavailable, then prompting bilingual generation
- **Community evaluation**: Replacing automated metrics with human evaluation by native speakers to assess cultural appropriateness, not just factual correctness

Ultimately, this work contributes to an emerging paradigm of **indigenous AI ethics**—frameworks that center indigenous epistemologies, data sovereignty, and language vitality rather than treating these as afterthoughts to dominant AI paradigms. As AI systems become increasingly ubiquitous, shaping how billions of people access information, the question of whose languages are supported becomes a question of whose knowledge endures. In Aotearoa and globally, the answer must be: all languages, equitably. Anything less perpetuates digital colonialism, using new technologies to repeat old injustices.

The mauri (life force) of te reo Māori depends not on algorithms but on speakers—children learning, elders teaching, communities thriving. Yet in a digitally mediated world, computational kaitiaki have a responsibility: to ensure that technology supports rather than suffocates the vitality of indigenous languages. This capstone represents one small effort in that direction, demonstrating both the possibilities and the profound work remaining.

---

## References

Chalmers, D. J. (1995). Facing up to the problem of consciousness. *Journal of Consciousness Studies*, 2(3), 200-219.

Couldry, N., & Mejias, U. A. (2019). *The costs of connection: How data is colonizing human life and appropriating it for capitalism*. Stanford University Press.

Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The state and fate of linguistic diversity and inclusion in the NLP world. *Proceedings of ACL 2020*, 6282-6293.

Rawls, J. (1971). *A theory of justice*. Harvard University Press.

Sen, A. (1999). *Development as freedom*. Oxford University Press.

Te Mana Raraunga. (2018). Principles of Māori data sovereignty. Retrieved from https://www.temanararaunga.maori.nz/

Waitangi Tribunal. (2011). *Ko Aotearoa Tēnei: A report into claims concerning New Zealand law and policy affecting Māori culture and identity* (Wai 262). Wellington: Legislation Direct.

---

**Word Count**: Approximately 3,800 words (4 pages at standard academic formatting)

**Author**: Alex
**Institution**: [Your University]
**Date**: October 2025
**Project**: Kaitiaki-Planner - Budget-Aware RAG for Te Reo Māori and English