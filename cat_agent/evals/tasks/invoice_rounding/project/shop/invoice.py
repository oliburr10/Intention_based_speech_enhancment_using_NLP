"""Invoices: a list of lines, with VAT added on top."""

from .money import format_money, to_pence

VAT_RATE = 0.20


class Invoice:
    def __init__(self):
        self.lines = []

    def add(self, name, unit_price, quantity=1):
        self.lines.append((name, unit_price, quantity))

    def subtotal_pence(self):
        return sum(to_pence(price * quantity) for _, price, quantity in self.lines)

    def vat_pence(self):
        return to_pence(self.subtotal_pence() * VAT_RATE / 100)

    def total_pence(self):
        return self.subtotal_pence() + self.vat_pence()

    def total(self):
        return format_money(self.total_pence())
