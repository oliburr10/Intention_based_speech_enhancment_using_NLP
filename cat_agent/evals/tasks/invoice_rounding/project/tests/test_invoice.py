from shop.invoice import Invoice
from shop.money import format_money


def test_simple_invoice():
    invoice = Invoice()
    invoice.add("notebook", 2.50, 2)
    assert invoice.subtotal_pence() == 500
    assert invoice.total() == "£6.00"


def test_format_money():
    assert format_money(1205) == "£12.05"
