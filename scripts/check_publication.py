"""Conservative tracked-file checks; findings never print matched secret values."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    "private-key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github-token": r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})",
    "aws-access-key": r"(?:AKIA|ASIA)[A-Z0-9]{16}",
    "service-key": r"(?:sk-(?:proj-|ant-)?[A-Za-z0-9_-]{24,}|hf_[A-Za-z0-9]{25,}|rpa_[A-Za-z0-9]{25,})",
    "credential-url": r"https?://[^\s/:]+:[^\s/@]+@",
    "user-home": r"(?:[A-Za-z]:[/\\]+Users[/\\]+|/(?:home|Users)/)[^\s\"']+",
    "jwt": r"eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}",
    "assigned-secret": r'''(?i)["']?(?:api_key|apikey|access_token|auth_token|password|client_secret)["']?\s*[:=]\s*["'][^"'\s]{12,}["']''',
}


def findings(name: str, data: bytes) -> list[str]:
    path = Path(name)
    issues = []
    if path.suffix not in {".py", ".md", ".json", ".txt", ".yml", ".toml"} and name != ".gitignore":
        issues.append("unexpected-file-type")
    if any(part.lower() in {"cloud", "data", ".aws", ".ssh", ".env", "__pycache__"} for part in path.parts):
        issues.append("excluded-directory")
    if len(data) > 2_000_000:
        issues.append("oversized-file")
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        return issues + ["non-text-file"]
    for label, pattern in PATTERNS.items():
        if re.search(pattern, content):
            issues.append(label)
    return issues


def main() -> None:
    # Inspect the index: exactly the blobs a commit will contain, not only disk copies.
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    paths = [name for name in paths if name]
    if not paths:
        raise SystemExit("No tracked files to audit; stage the allowlisted release first.")
    failures = []
    for name in paths:
        data = subprocess.check_output(["git", "show", f":{name}"], cwd=ROOT)
        for label in findings(name, data):
            failures.append(f"{name}: {label}")
    if failures:
        raise SystemExit("Publication check failed:\n" + "\n".join(failures))
    print(f"Publication check passed: {len(paths)} indexed text files; no configured secret/privacy patterns detected.")


if __name__ == "__main__":
    main()
