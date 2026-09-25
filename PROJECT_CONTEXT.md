# Intelligent Person Identification System
## Project Context

> This document is the single source of truth for the project's architecture,
> scope, technology decisions, implementation plan, and development rules.
>
> All contributors, GitHub Copilot, and Codex should read this document before
> making architectural or multi-module changes.

---

# 1. Project Title

## AI-Based Intelligent Multi-Camera Person Identification and Surveillance System

---

# 2. Project Type

B.Tech Final-Year Major Project.

The project focuses on computer-vision-based person detection, tracking,
face recognition, person re-identification, visual attributes, and
multi-modal identity verification.

---

# 3. Final Project Objective

Develop an intelligent surveillance system capable of:

1. Detecting people in CCTV/video streams.
2. Tracking people across video frames.
3. Extracting facial identity information when a usable face is available.
4. Performing person re-identification using body appearance.
5. Extracting supporting visual attributes.
6. Combining multiple sources of evidence to verify a person's identity.
7. Maintaining persistent person identities separately from temporary track IDs.
8. Storing detection and identity evidence in a persistent database.
9. Providing a web dashboard for monitoring, searching, alerts, and reports.
10. Evaluating the proposed hybrid identity verification method against
   individual/baseline approaches.

---

# 4. IMPORTANT SCOPE RESTRICTIONS

## Do NOT add:

- RAG
- LLM
- Chatbot
- Generative AI investigation assistant
- LLM-based report generation

The project is intentionally focused on the Computer Vision and
Multi-Modal Identity Verification pipeline.

Do not introduce major technologies that are outside this scope without
team approval.

---

# 5. Current Repository Structure

The current application is located in the inner project directory.

```text
Intelligent-person-identification/
│
├── .venv/
├── frontend/
│   ├── index.html
│   ├── main.js
│   └── style.css
│
├── modules/
│   ├── attribute_module.py
│   ├── tracking_module.py
│   └── __init__.py
│
├── app.py
├── README.md
├── requirements.txt
└── PROJECT_CONTEXT.md