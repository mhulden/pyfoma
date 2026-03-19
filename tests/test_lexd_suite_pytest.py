import pytest

from tests.test_lexd_suite import CPP_TESTS, HAND_TESTS, run_test


@pytest.mark.parametrize("test_case", HAND_TESTS + CPP_TESTS, ids=lambda t: t.name)
def test_lexd_suite_case(test_case):
    run_test(test_case)
