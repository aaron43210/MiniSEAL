"""
Generates a professional capstone project proposal DOCX for MiniSEAL.
"""

from docx import Document
from docx.shared import Pt, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


DARK_BLUE = RGBColor(0x1A, 0x36, 0x5D)
MED_BLUE  = RGBColor(0x2C, 0x52, 0x82)
BLACK     = RGBColor(0x00, 0x00, 0x00)


#  Helpers 

def heading(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(16 if level == 1 else 10)
    p.paragraph_format.space_after  = Pt(4)
    r = p.add_run(text)
    r.bold = True
    r.font.color.rgb = DARK_BLUE if level == 1 else MED_BLUE
    r.font.size = Pt(15 if level == 1 else 12)
    return p


def body(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after  = Pt(7)
    p.paragraph_format.line_spacing = Pt(15)
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.font.color.rgb = BLACK
    return p


def bullet(doc, text, bold_lead=None):
    """bullet_lead: optional bold prefix text before the rest."""
    p = doc.add_paragraph(style="List Bullet")
    p.paragraph_format.space_after = Pt(4)
    if bold_lead:
        rb = p.add_run(bold_lead)
        rb.bold = True
        rb.font.size = Pt(11)
        rn = p.add_run(text)
        rn.font.size = Pt(11)
    else:
        r = p.add_run(text)
        r.font.size = Pt(11)


def numbered(doc, text, bold_lead=None):
    p = doc.add_paragraph(style="List Number")
    p.paragraph_format.space_after = Pt(5)
    if bold_lead:
        rb = p.add_run(bold_lead)
        rb.bold = True
        rb.font.size = Pt(11)
        rn = p.add_run(text)
        rn.font.size = Pt(11)
    else:
        r = p.add_run(text)
        r.font.size = Pt(11)


def rule(doc):
    p = doc.add_paragraph()
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bot = OxmlElement("w:bottom")
    bot.set(qn("w:val"),   "single")
    bot.set(qn("w:sz"),    "6")
    bot.set(qn("w:space"), "1")
    bot.set(qn("w:color"), "BBBBBB")
    pBdr.append(bot)
    pPr.append(pBdr)
    p.paragraph_format.space_after = Pt(8)


#  Document 

doc = Document()

for sec in doc.sections:
    sec.top_margin    = Inches(1.0)
    sec.bottom_margin = Inches(1.0)
    sec.left_margin   = Inches(1.25)
    sec.right_margin  = Inches(1.25)


#  COVER 
tp = doc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_before = Pt(20)
tp.paragraph_format.space_after  = Pt(6)
tr = tp.add_run("MiniSEAL")
tr.bold = True
tr.font.size = Pt(28)
tr.font.color.rgb = DARK_BLUE

sp = doc.add_paragraph()
sp.alignment = WD_ALIGN_PARAGRAPH.CENTER
sp.paragraph_format.space_after = Pt(4)
sr = sp.add_run(
    "A Human-Supervised, Self-Adapting Language Model with\n"
    "SEAL-Inspired Continuous Learning and LoRA-Based Adaptation"
)
sr.font.size = Pt(13)
sr.font.color.rgb = MED_BLUE

cat = doc.add_paragraph()
cat.alignment = WD_ALIGN_PARAGRAPH.CENTER
cr = cat.add_run("Capstone Project Proposal  |  Department of Computer Science")
cr.font.size = Pt(10)
cr.font.color.rgb = RGBColor(0x77, 0x77, 0x77)
cr.italic = True

doc.add_paragraph()
rule(doc)


#  1. TITLE 
heading(doc, "1.  Title")
body(doc,
    "MiniSEAL: A Human-Supervised, Self-Adapting Language Model with SEAL-Inspired Continuous Learning "
    "and LoRA-Based Adaptation.")
rule(doc)


#  2. INTRODUCTION 
heading(doc, "2.  Introduction")

body(doc,
    "Large Language Models have fundamentally transformed the landscape of artificial intelligence, "
    "demonstrating strong performance across natural language understanding, question answering, "
    "summarisation, and dialogue. Models such as GPT-4, Llama-3, and Claude have set new benchmarks "
    "for what is achievable with large-scale Transformer architectures trained on vast corpora. "
    "Despite these advances, a critical limitation persists: once a model is trained and deployed, "
    "its weights are frozen. It cannot incorporate new knowledge, adapt to evolving institutional "
    "contexts, or correct factual errors without undergoing an expensive and time-consuming full "
    "retraining cycle.")

body(doc,
    "This limitation is particularly pronounced in institutional deployments. A university assistant, "
    "for example, must remain current with changes to course curricula, departmental policies, "
    "research developments, and campus events — none of which a static model can accommodate. "
    "Similarly, in enterprise environments, proprietary knowledge evolves continuously, and relying "
    "on a commercially hosted model introduces significant data privacy risks. There is therefore a "
    "strong need for a locally owned, continuously learning language model that can adapt to new "
    "information safely and efficiently.")

body(doc,
    "This project, MiniSEAL, addresses this need by designing and implementing a complete, private "
    "language model pipeline from first principles. The system covers every layer of the modern LLM "
    "stack — from a custom Byte-Pair Encoding tokeniser and a decoder-only Transformer architecture "
    "to a four-stage training pipeline comprising Pretraining, Supervised Fine-Tuning, Direct "
    "Preference Optimisation, and SEAL-inspired Continuous Adaptation. The self-adaptation engine, "
    "which forms the central novelty of this project, is directly inspired by the Self-Adapting "
    "Language Models framework published by MIT CSAIL in 2025. Unlike the original SEAL framework, "
    "which employs an autonomous Reinforcement Learning feedback loop, this project replaces that "
    "loop with a Human-in-the-Loop safety quarantine system, ensuring that every proposed weight "
    "update is reviewed and approved by a human administrator before being applied — making "
    "continuous learning safe, auditable, and suitable for real-world institutional deployment.")

rule(doc)


#  3. LITERATURE SURVEY 
heading(doc, "3.  Literature Survey")

body(doc,
    "The theoretical and methodological foundation of MiniSEAL is drawn from four seminal research "
    "contributions that directly inform its architecture, training methodology, and adaptation "
    "strategy.")

# — Paper 1
heading(doc, "3.1  Attention Is All You Need  —  Vaswani et al. (2017)", level=2)
body(doc,
    "Vaswani et al. introduced the Transformer architecture, proposing that sequence-to-sequence "
    "tasks could be solved entirely through self-attention mechanisms, dispensing with recurrent "
    "or convolutional networks. The paper formalised scaled dot-product attention, multi-head "
    "attention, and the encoder-decoder structure that became the basis for all subsequent large "
    "language models. The decoder-only variant of this architecture, used in autoregressive models "
    "such as GPT and LLaMA, forms the core model structure of this project.")

# — Paper 2
heading(doc, "3.2  Self-Adapting Language Models (SEAL)  —  Zweiger et al., MIT CSAIL (2025)", level=2)
body(doc,
    "This paper represents the primary research motivation for this project. Zweiger et al. "
    "propose a framework in which a deployed language model autonomously generates structured "
    "self-edits — synthetic training examples derived from a raw passage of text — and applies "
    "supervised fine-tuning on its own weights to incorporate new factual knowledge persistently. "
    "The framework uses a Reinforcement Learning feedback loop in which the reward signal is the "
    "model's performance improvement on a downstream task following the self-edit. The authors "
    "demonstrate strong results in knowledge incorporation and few-shot generalisation tasks, while "
    "identifying catastrophic forgetting as a key open challenge. This project adopts the SEAL "
    "self-edit generation paradigm but replaces the autonomous RL loop with a human-supervised "
    "approval gate, mitigating the risk of adversarial or erroneous self-edits corrupting model weights.")

# — Paper 3
heading(doc, "3.3  LoRA: Low-Rank Adaptation  —  Hu et al., Microsoft Research (2022)", level=2)
body(doc,
    "Hu et al. proposed Low-Rank Adaptation as a parameter-efficient approach to fine-tuning large "
    "language models. Rather than updating all model weights during adaptation, LoRA freezes the "
    "pretrained weights and injects small, trainable low-rank matrix pairs into each targeted layer. "
    "The effective update is expressed as ΔW = BA, where the rank r is far smaller than the original "
    "weight dimensions, reducing trainable parameters by over 98% relative to full fine-tuning while "
    "achieving comparable performance. This technique is employed in this project as the exclusive "
    "mechanism for continuous adaptation, ensuring that the base model remains intact and that "
    "individual knowledge updates remain modular, versioned, and independently removable.")

# — Paper 4
heading(doc, "3.4  Direct Preference Optimisation  —  Rafailov et al., Stanford University (2023)", level=2)
body(doc,
    "Rafailov et al. introduced Direct Preference Optimisation as a stable, mathematically grounded "
    "alternative to Reinforcement Learning from Human Feedback for aligning language model behaviour "
    "with human preferences. DPO reformulates the preference alignment objective as a closed-form "
    "classification loss operating over pairs of preferred and rejected model responses, eliminating "
    "the need for a separately trained reward model. The resulting training procedure is significantly "
    "more stable and computationally efficient than RLHF. In this project, DPO is applied following "
    "Supervised Fine-Tuning to ensure that the model produces responses that are helpful, safe, and "
    "appropriately toned before continuous self-adaptation begins.")

rule(doc)


#  4. AIM 
heading(doc, "4.  Aim of the Project")
body(doc,
    "The overarching aim of this project is to design, implement, and train a complete, privately "
    "owned language model that is capable of continuous, human-supervised self-adaptation. "
    "The specific objectives are:")

numbered(doc,
    "To design and implement a decoder-only Transformer language model and a custom Byte-Pair "
    "Encoding tokeniser entirely from scratch, without relying on any pre-built foundational model "
    "or commercial API.")
numbered(doc,
    "To establish a complete, sequential four-stage training pipeline — Pretraining on raw corpora, "
    "Supervised Fine-Tuning on instruction-response pairs, Direct Preference Optimisation on "
    "preference data, and SEAL-inspired Continuous Adaptation — each implemented as a modular, "
    "independently executable service.")
numbered(doc,
    "To implement a SEAL-inspired self-adaptation engine in which the model generates its own "
    "structured training data from raw passages and updates its weights persistently and efficiently "
    "using Low-Rank Adaptation (LoRA).")
numbered(doc,
    "To engineer a Human-in-the-Loop safety system that intercepts every model-generated self-edit, "
    "automatically assesses its risk level, stages it in a quarantine database, and requires explicit "
    "human approval before any weight update is committed — preventing model poisoning and ensuring "
    "auditability.")
numbered(doc,
    "To train and validate the complete pipeline on an NVIDIA A100 GPU, demonstrating that the system "
    "is scalable to research-grade hardware and suitable for institutional deployment.")

rule(doc)


#  5. METHODOLOGY 
heading(doc, "5.  Methodology and Planned Process")

body(doc,
    "The project follows a structured, sequential development methodology. Each stage builds directly "
    "on the outputs of the previous stage, mirroring the training pipeline used by leading AI "
    "research organisations.")

# Step 1
heading(doc, "Step 1 — Architecture Design and Tokeniser Training", level=2)
body(doc,
    "The first step involves designing the core model architecture: a decoder-only Transformer with "
    "pre-norm RMSNorm, Rotary Positional Embedding (RoPE), and a SwiGLU feed-forward network. "
    "In parallel, a custom Byte-Pair Encoding tokeniser is trained on the project corpus, producing "
    "a vocabulary tailored to the target domain. This step establishes the foundational components "
    "on which all subsequent training stages depend.")

# Step 2
heading(doc, "Step 2 — Pretraining", level=2)
body(doc,
    "The model is pretrained using next-token prediction on a curated raw text corpus. This stage "
    "teaches the model the statistical structure of language, building a general-purpose language "
    "understanding capability. Training is performed on an NVIDIA A100 GPU using the causal "
    "cross-entropy loss. The output is a base model checkpoint that serves as the starting point "
    "for all subsequent alignment and adaptation stages.")

# Step 3
heading(doc, "Step 3 — Supervised Fine-Tuning and Preference Alignment", level=2)
body(doc,
    "The pretrained base model undergoes two sequential alignment stages. First, Supervised "
    "Fine-Tuning trains the model on instruction-response pairs, transforming it from a raw "
    "next-token predictor into an instruction-following assistant. Second, Direct Preference "
    "Optimisation aligns the model's tone and behaviour with human preferences by training it "
    "to prefer high-quality responses over low-quality ones. After this stage, the model is "
    "safe, helpful, and appropriately calibrated.")

# Step 4
heading(doc, "Step 4 — SEAL-Inspired Continuous Self-Adaptation", level=2)
body(doc,
    "This is the core novelty of the project. The aligned model is given a raw passage of text "
    "and generates its own structured question-and-answer pairs — self-edits — describing the "
    "key facts contained in the passage. These self-edits are automatically scored for risk and "
    "staged in a quarantine database. A human administrator reviews each self-edit on a web "
    "dashboard and approves or rejects it. Approved self-edits are used to train a LoRA adapter "
    "on the aligned model's Q and V projection layers, permanently incorporating the new knowledge "
    "without modifying the base model weights.")

# Step 5
heading(doc, "Step 5 — Validation and Evaluation", level=2)
body(doc,
    "The complete system is evaluated end-to-end on the NVIDIA A100 cluster. Evaluation metrics "
    "include training loss curves across all stages, perplexity on held-out validation sets, "
    "factual accuracy on knowledge incorporation tasks, and a qualitative assessment of the "
    "model's responses before and after LoRA adaptation.")

rule(doc)


#  6. EXPECTED OUTCOMES 
heading(doc, "6.  Expected Outcomes")

body(doc,
    "Upon successful completion, the project is expected to produce the following outcomes:")

bullet(doc,
    "A working, trained language model capable of generating coherent, factually grounded responses "
    "to natural language queries.",
    bold_lead="Trained Language Model:  ")
bullet(doc,
    "A demonstrated self-adaptation pipeline in which the model successfully reads a new passage, "
    "generates self-edits, and incorporates the new knowledge through LoRA fine-tuning.",
    bold_lead="Demonstrated Continuous Learning:  ")
bullet(doc,
    "Measurable improvement in factual accuracy on domain-specific questions following the "
    "application of approved LoRA adapters.",
    bold_lead="Factual Accuracy Improvement:  ")
bullet(doc,
    "A validated Human-in-the-Loop safety gate that successfully intercepts, scores, and flags "
    "high-risk self-edits before any model weights are updated.",
    bold_lead="Safety Gate Validation:  ")
bullet(doc,
    "A scalable, enterprise-ready MLOps architecture demonstrating that the entire pipeline can "
    "be reproducibly executed on GPU cluster hardware.",
    bold_lead="Reproducible MLOps Pipeline:  ")

rule(doc)


#  7. DELIVERABLES 
heading(doc, "7.  Deliverables")

numbered(doc,
    "Complete source code for the end-to-end pipeline, including the Transformer model, "
    "BPE tokeniser, training services, and self-adaptation engine.",
    bold_lead="Source Code Repository:  ")
numbered(doc,
    "Trained model checkpoints from each stage of the training pipeline (Pretrained, SFT-aligned, "
    "DPO-aligned, and LoRA-adapted), stored as versioned artefacts.",
    bold_lead="Model Checkpoints:  ")
numbered(doc,
    "A web-accessible Human-in-the-Loop administration dashboard for reviewing, approving, and "
    "rejecting model-generated self-edits.",
    bold_lead="Admin Dashboard:  ")
numbered(doc,
    "A comprehensive technical report documenting the system architecture, training methodology, "
    "evaluation results, and a comparison between the SEAL RL approach and the Human-in-the-Loop "
    "quarantine approach implemented in this project.",
    bold_lead="Technical Report:  ")
numbered(doc,
    "A project presentation and live demonstration of the self-adaptation pipeline, including a "
    "real-time demonstration of a new passage being incorporated into the model's knowledge base "
    "through the approval dashboard.",
    bold_lead="Project Presentation and Demo:  ")

rule(doc)


#  REFERENCES 
heading(doc, "References")

refs = [
    "[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS). arXiv:1706.03762.",
    "[2] Zweiger, A., Pari, J., Guo, H., Akyürek, E., Kim, Y., & Agrawal, P. (2025). Self-Adapting Language Models. MIT CSAIL. arXiv:2506.10943.",
    "[3] Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2022). LoRA: Low-Rank Adaptation of Large Language Models. International Conference on Learning Representations (ICLR). arXiv:2106.09685.",
    "[4] Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. Advances in Neural Information Processing Systems (NeurIPS). arXiv:2305.18290.",
]

for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(5)
    r = p.add_run(ref)
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)


#  Save 
output_path = "/Users/aaronr/Desktop/DUK/GENAI/self_owned_llm/Capstone_Project_Description.docx"
doc.save(output_path)
print(f"  Saved → {output_path}")
