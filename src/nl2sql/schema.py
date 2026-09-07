"""
Builds a small synthetic retail/business SQLite database the NL-to-SQL
translator operates over: customers, products, orders, order_items. All data
is generated (seeded, reproducible) -- no proprietary or real data involved.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

REGIONS = ["North America", "EMEA", "APAC", "LATAM"]
SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Consumer"]
CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Sporting Goods", "Office Supplies", "Grocery"]
ORDER_STATUSES = ["completed", "pending", "cancelled", "refunded"]

PRODUCT_NAMES = {
    "Electronics": ["Wireless Earbuds", "4K Monitor", "Bluetooth Speaker", "USB-C Hub", "Smart Watch", "Laptop Stand"],
    "Home & Kitchen": ["Stand Mixer", "Air Fryer", "Cast Iron Skillet", "French Press", "Knife Set", "Blender"],
    "Apparel": ["Running Jacket", "Denim Jeans", "Wool Sweater", "Rain Boots", "Baseball Cap", "Yoga Pants"],
    "Sporting Goods": ["Yoga Mat", "Dumbbell Set", "Tennis Racket", "Camping Tent", "Bike Helmet", "Hiking Backpack"],
    "Office Supplies": ["Ergonomic Chair", "Standing Desk", "Notebook Pack", "Wireless Mouse", "Desk Lamp", "Whiteboard"],
    "Grocery": ["Organic Coffee", "Trail Mix", "Olive Oil", "Pasta Bundle", "Herbal Tea", "Protein Bars"],
}

FIRST_NAMES = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Avery", "Quinn",
               "Priya", "Wei", "Fatima", "Noah", "Elena", "Marcus", "Hana", "Diego", "Kwame", "Grace"]
LAST_NAMES = ["Nguyen", "Garcia", "Smith", "Kim", "Patel", "Muller", "Rossi", "Silva", "Cohen", "Dubois",
              "Okafor", "Larsson", "Haddad", "Novak", "Reyes", "Suzuki", "Weiss", "Costa", "Mensah", "Kaur"]


def _random_date(rng: random.Random, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=rng.randint(0, delta))


def build_database(db_path: Path, n_customers: int = 800, n_products: int = 180,
                    n_orders: int = 6000, seed: int = 42) -> None:
    rng = random.Random(seed)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE customers (
        customer_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        segment TEXT NOT NULL,
        region TEXT NOT NULL,
        signup_date TEXT NOT NULL
    );
    CREATE TABLE products (
        product_id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        unit_price REAL NOT NULL
    );
    CREATE TABLE orders (
        order_id INTEGER PRIMARY KEY,
        customer_id INTEGER NOT NULL,
        order_date TEXT NOT NULL,
        region TEXT NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
    );
    CREATE TABLE order_items (
        order_item_id INTEGER PRIMARY KEY,
        order_id INTEGER NOT NULL,
        product_id INTEGER NOT NULL,
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL,
        FOREIGN KEY (order_id) REFERENCES orders(order_id),
        FOREIGN KEY (product_id) REFERENCES products(product_id)
    );
    """)

    start, end = date(2023, 1, 1), date(2026, 8, 31)

    customers = []
    for cid in range(1, n_customers + 1):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        segment = rng.choice(SEGMENTS)
        region = rng.choice(REGIONS)
        signup = _random_date(rng, start, end).isoformat()
        customers.append((cid, name, segment, region, signup))
    cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)

    products = []
    pid = 1
    for _ in range(n_products):
        category = rng.choice(CATEGORIES)
        base_name = rng.choice(PRODUCT_NAMES[category])
        name = f"{base_name} v{rng.randint(1,4)}" if rng.random() < 0.3 else base_name
        price = round(rng.uniform(8, 450), 2)
        products.append((pid, name, category, price))
        pid += 1
    cur.executemany("INSERT INTO products VALUES (?,?,?,?)", products)

    orders = []
    order_items = []
    oid, oiid = 1, 1
    for _ in range(n_orders):
        customer = rng.choice(customers)
        cid, _, _, cust_region, _ = customer
        order_date = _random_date(rng, start, end).isoformat()
        status = rng.choices(ORDER_STATUSES, weights=[0.78, 0.10, 0.07, 0.05])[0]
        orders.append((oid, cid, order_date, cust_region, status))

        n_items = rng.randint(1, 5)
        for _ in range(n_items):
            product = rng.choice(products)
            pid_, _, _, price = product
            qty = rng.randint(1, 6)
            order_items.append((oiid, oid, pid_, qty, price))
            oiid += 1
        oid += 1

    cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?)", orders)
    cur.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", order_items)

    conn.commit()
    conn.close()


SCHEMA_DESCRIPTION = """
customers(customer_id, name, segment, region, signup_date)
products(product_id, name, category, unit_price)
orders(order_id, customer_id, order_date, region, status)   -- status in (completed, pending, cancelled, refunded)
order_items(order_item_id, order_id, product_id, quantity, unit_price)
""".strip()


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "data" / "business.db"
    build_database(out)
    print(f"Built {out}")
