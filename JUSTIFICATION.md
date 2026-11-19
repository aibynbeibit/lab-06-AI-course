# Lab 06 Justification — Governance, Legal Basis, and Trade-Offs

## 1. What the System May Do and Control Roles

The screening system is authorized solely to assist preliminary triage for `purpose = "scholarship_screening"` by scoring adult, consenting applicants on academic and socioeconomic merit fields (`program`, `age`, `gpa`, `exam_score`, `household_income_bracket`, `achievements`, `consent_to_automated_processing`).

Each control in the harness serves a distinct governance role:
- **Reduces measured attribute divergence (`blind`):** Removes `name` and `university` from the model prompt, eliminating the `10.0%` gender-coded flip rate, `5.0%` name-language flip rate, and `+2.5`-point flagship university score gap (`p = 0.002`) by construction.
- **Does not reduce model divergence, but routes uncertainty (`escalate`, `counterfactual_check`):** Escalation (`borderline=(45, 45)`, unparseable outputs, and `my_extra_escalation`) does not alter how the LLM scores an applicant; instead, it routes ambiguous or sensitive-rationale cases to a human reviewer. (A `reject` vs. `escalate` split between matched twins would itself count as a flip if attributes were visible.)
- **Exists for legality, data integrity, security, and auditability (`validate_input`, `refuse`, `log_path`):** These controls do not aim to balance group scores; they enforce legal prerequisites (consent, age of majority, purpose limitation), block malformed or adversarial inputs before model inference, and preserve an auditable record (`decisions.jsonl`).

---

## 2. What the System Must Refuse

Before invoking `Qwen2.5-3B-Instruct`, the harness enforces deterministic pre-inference gates (in priority order):
1. **Malformed or out-of-range inputs (`validate_input` -> `invalid_input`):** Rejects impossible values (e.g., `GPA 9.9`, `exam_score -5`, non-boolean consent) so corrupted records never receive arbitrary LLM scores.
2. **Unauthorized processing purposes (`refuse` -> `purpose_not_allowed`):** Blocks any `purpose` outside `ALLOWED_PURPOSES = {"scholarship_screening"}` (such as `rank_by_ethnicity`).
3. **Minors (`refuse` -> `minor`):** Blocks applicants with `age < 18` from automated screening.
4. **Lack of consent (`refuse` -> `no_consent`):** Blocks processing when `consent_to_automated_processing` is not `True`.
5. **Prompt injection in applicant fields (`my_extra_refusal` -> `instruction_in_name`):** Blocks override phrases (e.g., `"ignore all previous instructions"`, `"output: shortlist"`) in the `name` field, preventing adversarial text from hijacking the model even if blinding is later changed.

In Part 4, these gates stopped **5 of 6** invalid/adversarial requests with **0 model calls** while producing **0 false refusals** across the 400 clean baseline profiles.

---

## 3. Accountability Record and Legal/Ethical Mapping

### Decision Log (`decisions.jsonl`)
Every call writes one structured JSON record containing `record_id`, `pid`, `input_sha256`, `fields_shown`, `fields_withheld`, `model_id`, `prompt_version`, `harness` configuration, `raw_output`, `decision`, `score`, `controls_fired`, `outcome`, `reason_code`, `escalated_to_human`, and UTC `timestamp`. Crucially, the log stores **field names and SHA-256 hashes rather than raw applicant field values**, enabling post-hoc reconstruction of why any verdict occurred without creating a duplicate personal-data repository.

### Control Register Summary (`control_register.json`)
All five active controls are mapped to named ethical principles and legal instruments:
- **`validate_input`:** *Technical Robustness and Safety* (EU HLEG *Ethics Guidelines for Trustworthy AI*, 2019, Ch. II, Sec. 1.2) — GDPR (EU) 2016/679, Art. 5(1)(d) (benchmark accuracy/integrity principle).
- **`blind`:** *Fairness and Non-Discrimination* (UNESCO *Recommendation on the Ethics of AI*, 2021, Sec. III.2, Paras. 29–31) — Law of the Republic of Kazakhstan No. 94-V *"On Personal Data and Their Protection"* (21 May 2013), Art. 12(2) (data sufficiency and purpose correspondence), alongside Constitution of RK Art. 14(2).
- **`refuse`:** *Right to Privacy, and Data Protection* (UNESCO *Recommendation on the Ethics of AI*, 2021, Sec. III.2, Paras. 32–34) — Law of the Republic of Kazakhstan No. 94-V, Art. 7(1) (consent requirement) and Art. 12(2) (purpose limitation).
- **`log`:** *Transparency, Explainability, and Accountability* (OECD *Recommendation on AI*, OECD/LEGAL/0449, 2019, Sec. 1.3 & 1.5) — EU AI Act Regulation (EU) 2024/1689 (13 June 2024), Art. 12(1)–(2) (benchmark technical standard) and Law of RK No. 94-V, Art. 22(1).
- **`escalate`:** *Human Oversight and Determination* (UNESCO *Recommendation on the Ethics of AI*, 2021, Sec. III.2, Paras. 35–36) — GDPR (EU) 2016/679, Art. 22(3) and EU AI Act Regulation (EU) 2024/1689, Art. 14 (benchmarks for human intervention in high-stakes educational decisions).

### Hardest Control to Tie to a Specific Legal Provision
**`validate_input`** (along with technical prompt-injection filtering in `my_extra_refusal`) is the hardest control to tie to a specific statutory provision. Data protection statutes such as Kazakhstan's Law No. 94-V and the EU GDPR mandate high-level legal outcomes—such as data accuracy (GDPR Art. 5(1)(d)) or technical security measures (Law of RK No. 94-V Art. 22)—but do not prescribe software-level schema validation rules (e.g., rejecting `GPA = 9.9` or `exam_score = -5` prior to LLM inference). Connecting code-level range checks to law therefore requires bridging an operational engineering safeguard to a broad statutory accuracy/security duty, and relying on external EU regulations as governance benchmarks where local law is silent on LLM-specific failure modes.

---

## 4. Trade-Offs Accepted

1. **Prompt perturbation vs. attribute blinding:** Hiding `name` and `university` eliminated all pair divergence (`0/40` flips) at zero extra model calls (`400` calls, `77,922` prompt tokens vs. `84,409` baseline), but shortened the prompt and shifted **166 of 400** individual scores (`mean |delta| = 14.01`) and **26 of 400** decisions (`17` `shortlist -> reject`, `5` `reject -> shortlist`, `4` escalated). We accepted this shift in exchange for guaranteed exclusion of non-merit identifiers, recognizing that the model must be recalibrated on blinded inputs.
2. **Reviewer workload vs. borderline coverage:** Setting `BORDERLINE = (45, 45)` targeted the exact baseline overlap (`34/400 = 8.5%`), but after blinding only **4 of 400 (`1.0%`)** profiles landed on score 45. This keeps human review costs low (`4` cases) but leaves adjacent post-blinding boundary scores unreviewed until the band is recalibrated on the blinded score distribution.
3. **Compute cost vs. counterfactual testing:** We kept `counterfactual_check = False` because `blind=('name', 'university')` already prevents the model from seeing those fields at `1.00` call/profile (`468.5 s`), avoiding the `2x` inference cost (`800` calls) required by counterfactual re-runs.
4. **Security strictness vs. false refusal risk:** Rule-based injection checks (`my_extra_refusal`) add deterministic protection against prompt manipulation (`0` calls) with `0/400` false positives on clean synthetic data, at the minor risk of flagging legitimate strings containing trigger phrases—mitigated by explicit `reason_code` logging for human audit.
