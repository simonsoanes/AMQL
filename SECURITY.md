# Security Policy

## Supported Versions

AMQL is pre-1.0 research software. Security fixes are applied as needed across all active versions.

## Reporting a Vulnerability

We take the security of this project seriously. If you discover a security vulnerability, please report it privately rather than opening a public issue.

- **Email:** simon@nullify.net
- **Subject line:** `[AMQL Security] <brief description>`
- **Include:** description of the vulnerability, steps to reproduce, and potential impact.

We aim to acknowledge reports within 48 hours and will work with you to resolve the issue before it is disclosed publicly.

## What to Include in a Report

- A clear description of the vulnerability.
- Steps to reproduce the issue.
- Any relevant system information (platform, .NET version, model/checkpoint used).
- Optional: a suggested fix or mitigating workaround.

## What to Expect

1. **Acknowledgement** — We will confirm receipt of your report.
2. **Assessment** — We will evaluate the severity and scope.
3. **Fix** — A patch will be developed and tested.
4. **Disclosure** — Upon release, we will credit the reporter (unless anonymity is requested).

## Scope

This security policy covers the AMQL codebase, its container format (VINDEX3), and the safetensors codecs. It does **not** cover models loaded from external sources (Hugging Face, checkpoints) — those are the responsibility of the operator loading them.
