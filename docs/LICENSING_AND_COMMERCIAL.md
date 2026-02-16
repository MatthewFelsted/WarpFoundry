# Licensing and Commercial Planning

This guide explains the built-in licensing/commercial packaging support added to New Project creation.

## What It Generates

When enabled in **New Project -> Licensing & Commercial Packaging**, WarpFoundry creates:

- `.codex_manager/business/licensing_profile.json`
- `.codex_manager/business/legal_review.json`
- `docs/LICENSING_STRATEGY.md`
- `docs/COMMERCIAL_OFFERING.md`
- `docs/PRICING_TIERS.md` (optional)

These files are drafts intended to accelerate planning, not legal advice.

## Strategy Options

- `oss_only`: Open-source only
- `open_core`: Open core (OSS + paid add-ons)
- `dual_license`: Dual license (OSS + commercial)
- `hosted_service`: Hosted service (SaaS/API)

## Policy and Safety Defaults

- Generated plans explicitly avoid absolute legal/compliance guarantees.
- Monetization/marketing review now includes governance warnings for:
  - overconfident claims (for example, "risk-free", "guaranteed")
  - manipulative engagement language
  - weak or missing research citations

## Recommended Workflow

1. Generate initial licensing/commercial docs during project creation.
2. Run `Monetization Mode` to draft options and decision cards.
3. Use the Owner Decision Board (`approve` / `hold` / `deny`) to track choices.
4. Ask legal counsel to review before public release or paid launch.
5. Record legal sign-off in the legal-review checkpoint before publishing pricing/licensing docs.

## Legal Review Checkpoint API

- `GET /api/project/legal-review/status?repo_path=...`
- `POST /api/project/legal-review/signoff`
  - body fields: `repo_path`, `approved`, `reviewer`, `notes`, optional `required`

## Governance Policy Controls

You can now manage source-domain allow/deny policies from the GUI under:

- `Pipeline -> Advanced Settings -> Deep Research -> Source Governance Policy`

The same policy remains available through environment variables and API:

- `GET /api/governance/source-policy`
- `POST /api/governance/source-policy`

## Environment Variable Equivalents

You can tighten source policy checks with environment variables:

- `CODEX_MANAGER_RESEARCH_ALLOWED_DOMAINS`
  - Comma-separated allowlist for source domains used in governance checks.
- `CODEX_MANAGER_RESEARCH_BLOCKED_DOMAINS`
  - Comma-separated denylist for source domains used in governance checks.
- `DEEP_RESEARCH_ALLOWED_SOURCE_DOMAINS`
  - Allowlist used when filtering native deep-research citations.
- `DEEP_RESEARCH_BLOCKED_SOURCE_DOMAINS`
  - Denylist used when filtering native deep-research citations.

