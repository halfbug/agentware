from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AGENTS_DIR = ROOT / "src" / "agentware" / "agents"
EXAMPLES_DIR = ROOT / "examples"
AGENT_TESTS_DIR = ROOT / "tests" / "agents"

REQUIRED_AGENT_FILES = ("README.md", "__init__.py", "agent.py")


def _agent_dirs() -> list[Path]:
    return sorted(
        [
            p
            for p in AGENTS_DIR.iterdir()
            if p.is_dir() and not p.name.startswith("__")
        ]
    )


def test_each_agent_has_required_files():
    missing: list[str] = []
    for agent_dir in _agent_dirs():
        for rel in REQUIRED_AGENT_FILES:
            if not (agent_dir / rel).exists():
                missing.append(f"{agent_dir.name}/{rel}")
    assert not missing, f"Missing required agent files: {missing}"


def test_each_agent_has_examples_and_tests():
    missing_examples: list[str] = []
    missing_tests: list[str] = []

    for agent_dir in _agent_dirs():
        name = agent_dir.name
        example_dir = EXAMPLES_DIR / name
        tests_dir = AGENT_TESTS_DIR / name

        if not example_dir.exists():
            missing_examples.append(name)
        elif not list(example_dir.glob("*.py")):
            missing_examples.append(f"{name} (no .py files)")

        if not tests_dir.exists():
            missing_tests.append(name)
        elif not list(tests_dir.glob("test_*.py")):
            missing_tests.append(f"{name} (no test_*.py files)")

    assert not missing_examples, f"Agent examples missing: {missing_examples}"
    assert not missing_tests, f"Agent tests missing: {missing_tests}"


def test_agent_exports_are_registered():
    agents_init = (AGENTS_DIR / "__init__.py").read_text(encoding="utf-8")
    package_init = (ROOT / "src" / "agentware" / "__init__.py").read_text(
        encoding="utf-8"
    )
    missing_from_agents_init: list[str] = []
    missing_from_package_init: list[str] = []

    for agent_dir in _agent_dirs():
        marker = f"agentware.agents.{agent_dir.name}"
        if marker not in agents_init:
            missing_from_agents_init.append(agent_dir.name)
        if marker not in package_init:
            missing_from_package_init.append(agent_dir.name)

    assert not missing_from_agents_init, (
        "Missing exports in src/agentware/agents/__init__.py for: "
        f"{missing_from_agents_init}"
    )
    assert not missing_from_package_init, (
        "Missing exports in src/agentware/__init__.py for: "
        f"{missing_from_package_init}"
    )
