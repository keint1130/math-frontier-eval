import pytest

from grading import is_correct, verify


@pytest.mark.parametrize(
    ("gold", "response"),
    [
        (r"$\frac{9\sqrt{3}}{16}$", r"\boxed{\frac{9\sqrt3}{16}}"),
        (r"$\frac{7}{2}$", r"\boxed{YZ=\frac72}"),
        ("1.75", r"\boxed{\frac{S_{XYZ}}{S_{YZD}}=\frac74}"),
        ("567", r"\boxed{\sqrt2 A=567}"),
        (r"$\frac{4}{3}$", r"\boxed{\frac{QS}{QN}=\frac43}"),
    ],
)
def test_labelled_equivalent_answers_are_accepted(gold, response):
    assert verify(response, gold) == 1.0


def test_genuinely_different_answer_is_rejected():
    assert not is_correct("90", r"\text{Undefined as stated}")
