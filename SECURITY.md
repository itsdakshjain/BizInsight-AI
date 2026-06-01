# Security Policy

## Overview

The **BizInsight AI** team takes security seriously. This document outlines
how to responsibly report security vulnerabilities and what you can expect
after submitting a report.

Since BizInsight AI handles sensitive data including customer reviews,
API keys, SQLite databases, ChromaDB vector storage, and a FastAPI backend,
responsible disclosure is critical.

## Supported Versions

Only the latest version of the `main` branch is actively maintained and
receives security fixes.

| Version / Branch | Supported |
| ---------------- | --------- |
| `main` (latest)  | Yes       |
| Older branches   | No        |

## Reporting a Vulnerability

> **Please do NOT report security vulnerabilities via public GitHub Issues.**
> Public disclosure before a fix is in place puts all users at risk.

### How to Report

1. **Open a GitHub Security Advisory** (preferred):
   - Go to the repository, click the **Security** tab, then **Advisories**, then **Report a vulnerability**
   - Describe the vulnerability in detail

2. **Or contact the maintainer directly via email:**
   <prateekiiitg56@gmail.com>

### What to Include in Your Report

Please provide as much of the following as possible:

- Type of vulnerability (e.g., API key exposure, SQL injection, IDOR)
- Location — file name, function, or endpoint affected
- Step-by-step reproduction instructions
- Potential impact — what an attacker could achieve
- Suggested fix (optional but appreciated)

## What to Expect After Reporting

| Timeline        | Action                                                  |
| --------------- | ------------------------------------------------------- |
| Within 48 hours | Acknowledgement of your report                          |
| Within 7 days   | Initial assessment and severity classification          |
| Within 14 days  | Fix in progress or workaround communicated              |
| After fix       | Credit given to reporter (if desired) in release notes  |

We will keep you informed at each step. If you do not receive a response
within 48 hours, please follow up.

## Scope

### In Scope

- Exposed or hardcoded API keys (OpenRouter, etc.)
- Authentication or authorization bypass in the FastAPI backend
- Insecure data storage (SQLite, ChromaDB)
- Injection attacks (SQL injection, prompt injection via LLM pipeline)
- Sensitive data leakage through API responses
- Vulnerabilities in environment variable or secrets management
- Dependencies with a known CVE and an exploitable path

### Out of Scope

- Vulnerabilities in third-party services (OpenRouter, Google Gemini, etc.)
- Issues requiring physical access to the machine
- Social engineering attacks
- Theoretical vulnerabilities with no practical exploit path
- Issues already tracked in public GitHub Issues

## Security Best Practices for Contributors

- Never commit `.env` files or API keys to the repository
- Use `.env.example` for environment variable templates
- Always use `python-dotenv` to load secrets from environment variables
- Validate and sanitize all user-uploaded CSV inputs in `app.py`
- Keep dependencies up to date by running `pip install -r requirements.txt` regularly
- Report any accidental secret exposure immediately

## Preferred Languages

We accept security reports in **English**.

## Attribution

We are grateful to security researchers and contributors who help keep
BizInsight AI safe. Responsible disclosures will be credited in our
release notes with your permission.

This security policy follows recommended practices from
[GitHub's security advisory guidelines](https://docs.github.com/en/code-security/security-advisories).
