- Use pytest
- Name tests, test_that\_..... e.g. test_that_function_returns_true, test_that_function_with_input_raises_errortype
- Every \*.py file in src should have an equivalent test \*.py file or test folder if there are many tests distinct types of tests needed

# Core Functions

*Make sure .venv is activated before running tests*

- run tests with pytest: `python -m pytest rl_uplift/unit_tests/`
- run tests with pytest and generate html report: `python -m pytest rl_uplift/unit_tests/ --html=report.html`
- run tests with pytest and coverage report: `python -m pytest rl_uplift/unit_tests/ --cov=rl_uplift/src/`
