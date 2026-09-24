# AI Resume Review Agent

A beginner-friendly AI Resume Review Agent built with:

- Python
- Streamlit
- CrewAI
- Groq
- OpenAI GPT-OSS 120B
- pypdf

## Features

- Paste resume text
- Upload PDF resume
- Paste target job description
- Compare resume against job requirements
- Identify demonstrated skills
- Identify missing/not-demonstrated requirements
- Identify experience gaps
- Identify education/qualification gaps
- Suggest resume improvements
- Suggest relevant keywords
- Generate a priority action plan

## AI Architecture

The application uses a single CrewAI agent:

Resume + Job Description
        ↓
Resume Analyst Agent
        ↓
Groq GPT-OSS 120B
        ↓
Structured Resume Review

## Local Setup

Install Python 3.11.

Create a virtual environment:

```bash
python -m venv .venv
