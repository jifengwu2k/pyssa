# A warehouse/order-processing module used for lowering/interpreter testing.
#
# The goal is to look like ordinary application code while still exercising a broad slice of
# Python syntax and semantics: closures, classes, comprehensions, generators, pattern
# matching, with-statements, exceptions, async helpers, decorators, imports, and structured
# data processing.

GLOBAL_METRICS = {
    "orders_seen": 0,
    "shipments_built": 0,
    "validation_failures": 0,
}


def tagged(name):
    def decorate(obj):
        obj.tag = name
        return obj

    return decorate


def identity(obj):
    return obj


class AuditLog:
    def __init__(self, label):
        self.label = label
        self.events = []

    def __enter__(self):
        self.events.append(self.label + ":open")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.events.append(self.label + ":close")
        return False


class AsyncAuditLog:
    def __init__(self, label):
        self.label = label

    async def __aenter__(self):
        return self.label + ":async-open"

    async def __aexit__(self, exc_type, exc, tb):
        return False


@tagged("catalog-item")
class CatalogItem:
    __match_args__ = ("sku", "price", "stock")

    def __init__(self, sku, price, stock, tags):
        self.sku = sku
        self.price = price
        self.stock = stock
        self.tags = list(tags)

    def reserve(self, quantity):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if quantity > self.stock:
            raise ValueError("insufficient stock for " + self.sku)
        self.stock -= quantity
        line_total = round(quantity * self.price, 2)
        return {
            "sku": self.sku,
            "quantity": quantity,
            "unit_price": self.price,
            "line_total": line_total,
            "tags": tuple(self.tags),
        }

    def restock(self, amount):
        self.stock += amount
        return self.stock



def make_order_number(prefix):
    current = 1000

    def allocate():
        nonlocal current
        global GLOBAL_METRICS
        current = current + 1
        GLOBAL_METRICS = {
            "orders_seen": GLOBAL_METRICS["orders_seen"] + 1,
            "shipments_built": GLOBAL_METRICS["shipments_built"],
            "validation_failures": GLOBAL_METRICS["validation_failures"],
        }
        return prefix + "-" + str(current)

    return allocate



def parse_catalog_rows(rows):
    items = []
    for row in rows:
        match row:
            case {"sku": sku, "price": price, "stock": stock, "tags": tags}:
                items.append(CatalogItem(sku, float(price), int(stock), list(tags)))
            case _:
                raise ValueError("invalid catalog row")
    return items



def build_notice(order_id, customer, total=0, **metadata):
    return {
        "order_id": order_id,
        "customer": customer,
        "total": total,
        "metadata": metadata,
    }



def iter_line_skus(shipments):
    for shipment in shipments:
        for line in shipment["lines"]:
            yield line["sku"]


@identity
class OrderProcessor:
    tax_rate = 0.07

    def __init__(self, catalog_items):
        self.items_by_sku = {item.sku: item for item in catalog_items}
        self.audit_events = []
        self.processed_orders = 0

    def reserve_order(self, order):
        lines = []
        for request in order["lines"]:
            sku = request["sku"]
            quantity = request["quantity"]
            line = self.items_by_sku[sku].reserve(quantity)
            lines.append(line)

        subtotal = round(sum(line["line_total"] for line in lines), 2)
        tax = round(subtotal * self.tax_rate, 2)
        total = round(subtotal + tax, 2)
        notice = build_notice(order["order_id"], order["customer"], total=total, **{"line_count": len(lines)})
        self.processed_orders += 1
        return {
            "order_id": order["order_id"],
            "customer": order["customer"],
            "lines": lines,
            "subtotal": subtotal,
            "tax": tax,
            "total": total,
            "notice": notice,
        }

    def reserve_orders(self, orders):
        shipments = []
        self.current_batch = [order["order_id"] for order in orders]
        with AuditLog("reserve-orders") as audit:
            for order in orders:
                try:
                    shipment = self.reserve_order(order)
                except ValueError as exc:
                    audit.events.append("failed:" + order["order_id"] + ":" + str(exc))
                else:
                    shipments.append(shipment)
                    audit.events.append("ok:" + shipment["order_id"])
        self.audit_events = [*audit.events]
        self.last_batch_size = len(shipments)
        del self.current_batch
        return shipments

    def inventory_snapshot(self):
        return {sku: {"stock": item.stock, "tags": [*item.tags]} for sku, item in self.items_by_sku.items()}

    def restock_low_items(self, floor, amount):
        updated = []
        for item in self.items_by_sku.values():
            if item.stock < floor:
                updated.append((item.sku, item.restock(amount)))
        return updated



def update_shipment_metrics(shipments):
    global GLOBAL_METRICS
    GLOBAL_METRICS = {
        "orders_seen": GLOBAL_METRICS["orders_seen"],
        "shipments_built": GLOBAL_METRICS["shipments_built"] + len(shipments),
        "validation_failures": GLOBAL_METRICS["validation_failures"],
    }



def build_daily_report(processor, shipments):
    totals = [shipment["total"] for shipment in shipments]
    sorted_shipments = sorted(shipments, key=lambda shipment: shipment["total"], reverse=True)
    packed_ids = [] if not sorted_shipments else [sorted_shipments[0]["order_id"], *[shipment["order_id"] for shipment in sorted_shipments[1:]]]
    top_two = packed_ids[:2]
    high_value = [shipment["order_id"] for shipment in shipments if shipment["total"] >= 20]
    sku_names = sorted({line["sku"] for shipment in shipments for line in shipment["lines"]})
    sku_counts = {
        sku: sum(line["quantity"] for shipment in shipments for line in shipment["lines"] if line["sku"] == sku)
        for sku in sku_names
    }
    all_tags = {tag for item in processor.items_by_sku.values() for tag in item.tags}
    highlighted_tags = {"daily", *all_tags}
    report_flags = {"shipment_count": len(shipments), **{"high_value_count": len(high_value)}}
    summary_ids = ("report", *top_two)

    temporary_notes = [shipment["customer"] for shipment in shipments]
    note_count = len(temporary_notes)
    del temporary_notes, note_count

    if shipments and (first_total := shipments[0]["total"]) > 0:
        discount_fn = lambda amount, *, vip=False: amount * (0.85 if vip else 0.95)
        preview_discount = round(discount_fn(first_total, vip=True), 2)
    else:
        preview_discount = None

    return {
        "totals": tuple(totals),
        "window": totals[0:2],
        "top_two": top_two,
        "summary_ids": summary_ids,
        "high_value": high_value,
        "sku_counts": sku_counts,
        "highlighted_tags": highlighted_tags,
        "report_flags": report_flags,
        "preview_discount": preview_discount,
        "inventory": processor.inventory_snapshot(),
        "audit_tail": processor.audit_events[-3:],
    }



def import_examples(shipments):
    import math
    from statistics import mean as average

    totals = [shipment["total"] for shipment in shipments]
    floored = [math.floor(total) for total in totals]
    average_total = 0 if not totals else round(average(totals), 2)
    return floored, average_total



def classify_record(record):
    match record:
        case {"kind": "shipment", "order_id": order_id, "lines": [first, *rest]}:
            return ("shipment", order_id, first["sku"], len(rest))
        case CatalogItem(sku=sku, stock=stock):
            return ("item", sku, stock)
        case _:
            return ("unknown", None)



def validate_orders(orders):
    errors = []
    for order in orders:
        try:
            if not isinstance(order.get("customer"), str):
                raise TypeError("customer must be a string")
            if not order.get("lines"):
                raise ValueError("order must contain at least one line")
        except (TypeError, ValueError) as exc:
            errors.append(exc)
    if errors:
        raise ExceptionGroup("order validation failed", errors)
    return "ok"



def validation_summary(orders):
    global GLOBAL_METRICS
    summary = {"value": 0, "type": 0}
    had_failure = False
    try:
        validate_orders(orders)
    except* ValueError as errors:
        had_failure = True
        summary["value"] = len(errors.exceptions)
    except* TypeError as errors:
        had_failure = True
        summary["type"] = len(errors.exceptions)
    finally:
        if had_failure:
            GLOBAL_METRICS = {
                "orders_seen": GLOBAL_METRICS["orders_seen"],
                "shipments_built": GLOBAL_METRICS["shipments_built"],
                "validation_failures": GLOBAL_METRICS["validation_failures"] + 1,
            }
    if had_failure:
        return summary
    return "ok"


async def plus_one(value):
    return value + 1


async def supplier_price_stream(rows):
    for row in rows:
        yield row


async def refresh_remote_prices(rows):
    seen = [row async for row in supplier_price_stream(rows) if row["price"] > 0]
    sku_set = {row["sku"] async for row in supplier_price_stream(rows) if row["price"] > 0}
    price_map = {row["sku"]: row["price"] async for row in supplier_price_stream(rows) if row["price"] > 0}

    total_prices = 0
    async for row in supplier_price_stream(seen):
        total_prices = total_prices + row["price"]

    async with AsyncAuditLog("supplier-sync") as marker:
        session_marker = marker

    awaited = await plus_one(len(sku_set))

    try:
        raise ValueError("stale supplier record")
    except ValueError as exc:
        try:
            raise RuntimeError("supplier refresh failed") from exc
        except RuntimeError:
            try:
                raise
            except RuntimeError as reraised:
                reraised_name = type(reraised).__name__

    return (seen, sku_set, price_map, session_marker, total_prices, awaited, reraised_name)


NEXT_ORDER_NUMBER = make_order_number("ORD")

RAW_CATALOG = [
    {"sku": "tea", "price": 4.5, "stock": 12, "tags": ["drink", "pantry"]},
    {"sku": "mug", "price": 8.0, "stock": 4, "tags": ["home", "ceramic"]},
    {"sku": "beans", "price": 11.5, "stock": 6, "tags": ["drink", "coffee"]},
]

CATALOG = parse_catalog_rows(RAW_CATALOG)
PROCESSOR = OrderProcessor(CATALOG)

RAW_ORDERS = [
    {
        "order_id": NEXT_ORDER_NUMBER(),
        "customer": "Ada",
        "lines": [
            {"sku": "tea", "quantity": 2},
            {"sku": "mug", "quantity": 1},
        ],
    },
    {
        "order_id": NEXT_ORDER_NUMBER(),
        "customer": "Ben",
        "lines": [
            {"sku": "beans", "quantity": 1},
            {"sku": "tea", "quantity": 1},
        ],
    },
    {
        "order_id": NEXT_ORDER_NUMBER(),
        "customer": "Cara",
        "lines": [
            {"sku": "mug", "quantity": 10},
        ],
    },
]

SHIPMENTS = PROCESSOR.reserve_orders(RAW_ORDERS)
update_shipment_metrics(SHIPMENTS)
RESTOCKED = PROCESSOR.restock_low_items(4, 5)
DAILY_REPORT = build_daily_report(PROCESSOR, SHIPMENTS)
IMPORT_RESULT = import_examples(SHIPMENTS)
SKU_STREAM = list(iter_line_skus(SHIPMENTS))
SHIPMENT_TOTALS = tuple(shipment["total"] for shipment in SHIPMENTS)
CLASSIFIED_SHIPMENT = None if not SHIPMENTS else classify_record({
    "kind": "shipment",
    "order_id": SHIPMENTS[0]["order_id"],
    "lines": SHIPMENTS[0]["lines"],
})
CLASSIFIED_ITEM = classify_record(CATALOG[0])
VALIDATION_SAMPLE = [
    {"customer": "valid", "lines": [{"sku": "tea", "quantity": 1}]},
    {"customer": 42, "lines": [{"sku": "tea", "quantity": 1}]},
    {"customer": "missing-lines", "lines": []},
]
