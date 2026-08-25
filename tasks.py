"""
Task definitions for the Dev Team Crew.
Each task is assigned to a specific agent and has expected outputs.
"""
from crewai import Task


def create_research_task(topic, agents):
    return Task(
        description=(
            f"Research the following topic thoroughly: {topic}\n\n"
            "1. Explore current best practices and technologies\n"
            "2. Find relevant documentation and examples\n"
            "3. Compare at least 2-3 approaches\n"
            "4. Summarize findings with pros/cons\n"
            "5. Recommend the best approach with justification"
        ),
        expected_output=(
            "A detailed research report with:\n"
            "- Summary of findings\n"
            "- Comparison table of approaches\n"
            "- Recommended approach with justification\n"
            "- Key links and references"
        ),
        agent=agents["researcher"],
    )


def create_architecture_task(agents):
    return Task(
        description=(
            "Based on the research findings, design the system architecture:\n"
            "1. Define the overall architecture and tech stack\n"
            "2. Break the implementation into clear, ordered steps\n"
            "3. Define file structure and module boundaries\n"
            "4. Identify potential risks and mitigations\n"
            "5. Create a step-by-step implementation plan"
        ),
        expected_output=(
            "An architecture document with:\n"
            "- Tech stack and justification\n"
            "- File/module structure\n"
            "- Ordered implementation steps\n"
            "- Risk assessment\n"
            "- Data flow diagram (text-based)"
        ),
        agent=agents["architect"],
    )


def create_implementation_task(agents):
    return Task(
        description=(
            "Based on the architecture plan, implement the solution:\n"
            "1. Write clean, well-documented code following the architecture plan\n"
            "2. Follow best practices and design patterns\n"
            "3. Handle edge cases and error states\n"
            "4. Include meaningful comments\n"
            "5. Ensure code is modular and maintainable"
        ),
        expected_output=(
            "Complete, production-ready code with:\n"
            "- All source files\n"
            "- Configuration files\n"
            "- Inline documentation\n"
            "- Error handling\n"
            "- A summary of what was implemented"
        ),
        agent=agents["implementer"],
    )


def create_review_task(agents):
    return Task(
        description=(
            "Critically review the implementation:\n"
            "1. Check for bugs and logic errors\n"
            "2. Review security vulnerabilities\n"
            "3. Evaluate code quality and maintainability\n"
            "4. Identify edge cases not handled\n"
            "5. Rate the code 1-10 and list specific improvements\n"
            "6. If score is below 7, flag specific items for the Implementer to fix"
        ),
        expected_output=(
            "A detailed code review with:\n"
            "- Bug list (if any)\n"
            "- Security concerns (if any)\n"
            "- Quality rating (1-10)\n"
            "- Specific improvement suggestions\n"
            "- Verdict: APPROVED or NEEDS_REVISION with specific fixes"
        ),
        agent=agents["critic"],
    )


def create_testing_task(agents):
    return Task(
        description=(
            "Create comprehensive tests for the implementation:\n"
            "1. Write unit tests for all core functions\n"
            "2. Write integration tests for key workflows\n"
            "3. Test edge cases and error handling\n"
            "4. Report any failures with clear reproduction steps\n"
            "5. Provide test coverage summary"
        ),
        expected_output=(
            "A test suite with:\n"
            "- Unit test files\n"
            "- Integration test files\n"
            "- Test results summary (pass/fail counts)\n"
            "- Any failures with reproduction steps\n"
            "- Coverage estimate"
        ),
        agent=agents["tester"],
    )


def create_devops_task(agents):
    return Task(
        description=(
            "Set up deployment and infrastructure:\n"
            "1. Create Dockerfile for the application\n"
            "2. Define CI/CD pipeline (GitHub Actions)\n"
            "3. Set up environment variable templates\n"
            "4. Create deployment configuration\n"
            "5. Document the deployment process"
        ),
        expected_output=(
            "DevOps configuration with:\n"
            "- Dockerfile\n"
            "- CI/CD pipeline (.github/workflows/)\n"
            "- .env.example\n"
            "- Deployment config\n"
            "- Step-by-step deployment guide"
        ),
        agent=agents["devops"],
    )


def create_all_tasks(topic, agents):
    """Create and return all tasks in execution order."""
    return [
        create_research_task(topic, agents),
        create_architecture_task(agents),
        create_implementation_task(agents),
        create_review_task(agents),
        create_testing_task(agents),
        create_devops_task(agents),
    ]
