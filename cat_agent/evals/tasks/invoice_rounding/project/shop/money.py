"""Money is stored as whole pence to avoid floating-point surprises."""


def to_pence(amount):
    """Convert an amount in pounds (e.g. 1.05) to whole pence (105), rounding to the nearest penny."""
    return int(amount * 100)


def format_money(pence):
    """Format whole pence as pounds, e.g. 105 -> '£1.05'."""
    return f"£{pence // 100}.{pence % 100:02d}"
