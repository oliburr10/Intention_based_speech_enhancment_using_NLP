import pytest

from shop.invoice import Invoice
from shop.money import format_money, to_pence


@pytest.mark.parametrize("pounds, pence", [(0.29, 29), (19.99, 1999), (4.10, 410), (1.05, 105), (0.0, 0), (2.5, 250)])
def test_to_pence_rounds_to_nearest_penny(pounds, pence):
    assert to_pence(pounds) == pence


def test_the_customer_example():
    invoice = Invoice()
    invoice.add("sticker", 0.35, 3)
    assert invoice.subtotal_pence() == 105
    assert invoice.vat_pence() == 21
    assert invoice.total() == "£1.26"


def test_several_lines():
    invoice = Invoice()
    invoice.add("pen", 0.29, 1)
    invoice.add("pad", 19.99, 1)
    assert invoice.subtotal_pence() == 2028
    assert invoice.total() == "£24.34"


def test_unchanged_behaviour():
    invoice = Invoice()
    invoice.add("notebook", 2.50, 2)
    assert invoice.total() == "£6.00"
    assert format_money(1205) == "£12.05"
