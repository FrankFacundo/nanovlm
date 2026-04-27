from nanovlm.train.verifiers import (
    exact_match_reward,
    extract_boxed,
    extract_last_number,
    instruction_reward,
    math_equivalence_reward,
    numeric_reward,
    python_unit_test_reward,
    reward_record,
)


def test_extract_last_number_handles_commas():
    assert extract_last_number("answer is 1,234.5") == "1234.5"
    assert extract_last_number("no numbers") is None


def test_extract_boxed_picks_last():
    assert extract_boxed(r"first \boxed{1} then \boxed{42}") == "42"


def test_numeric_reward_strict_match():
    assert numeric_reward("the answer is 42", "42") == 1.0
    assert numeric_reward("the answer is 41", "42") == 0.0


def test_numeric_reward_uses_boxed_when_present():
    assert numeric_reward(r"work...\boxed{42}", "42") == 1.0


def test_math_equivalence_reward_basic_string():
    # No sympy needed for the trivial case
    assert math_equivalence_reward("x+1", "x+1") == 1.0


def test_instruction_reward_must_contain_and_max_words():
    assert instruction_reward("hello world", {"must_contain": "world", "max_words": 3}) == 1.0
    assert instruction_reward("hello world", {"must_contain": "missing"}) == 0.0
    assert instruction_reward("a b c d e f g", {"max_words": 5}) == 0.0


def test_instruction_reward_regex_and_json():
    assert instruction_reward("[1, 2]", {"json_parsable": True}) == 1.0
    assert instruction_reward("not json", {"json_parsable": True}) == 0.0
    assert instruction_reward("abc123", {"regex": r"\d+"}) == 1.0


def test_python_unit_test_reward_passes_simple():
    code = "def add(a, b):\n    return a + b\n"
    tests = "from main import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    r = python_unit_test_reward(code, tests, timeout_s=10.0)
    assert r == 1.0


def test_python_unit_test_reward_fails_on_bug():
    code = "def add(a, b):\n    return a - b\n"
    tests = "from main import add\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    r = python_unit_test_reward(code, tests, timeout_s=10.0)
    assert r == 0.0


def test_reward_record_dispatches():
    r, breakdown = reward_record("the answer is 42", {"answer": "42"})
    assert r == 1.0
    assert "numeric" in breakdown

    r2, _ = reward_record("ignore", {"constraints": {"must_contain": "yes"}})
    assert r2 == 0.0
