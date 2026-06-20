# Generates 500k polars examples

import random

OUTPUT_FILE = "train.txt"
NUM_EXAMPLES = 500_000

NUMERIC_COLUMNS = [
    "salary",
    "revenue",
    "sales",
    "price",
    "profit",
    "income",
    "balance",
    "score",
    "quantity",
    "age",
]

STRING_COLUMNS = [
    "department",
    "region",
    "city",
    "country",
    "product",
    "category",
    "company",
    "team",
    "employee",
    "customer",
]

DATE_COLUMNS = [
    "date",
    "timestamp",
    "order_date",
]

AGGREGATIONS = [
    ("average", "mean"),
    ("mean", "mean"),
    ("sum", "sum"),
    ("total", "sum"),
    ("maximum", "max"),
    ("minimum", "min"),
]

AGG_QUERIES = [
    "{agg} {value} by {group}",
    "{agg} {value} for each {group}",
    "{group} wise {agg} {value}",
    "calculate {agg} {value} grouped by {group}",
]

FILTER_GT = [
    "{column} greater than {value}",
    "{column} above {value}",
    "{column} more than {value}",
]

FILTER_LT = [
    "{column} less than {value}",
    "{column} below {value}",
    "{column} under {value}",
]

SORT_QUERIES = [
    "top 10 rows by {column}",
    "highest {column}",
    "largest {column} values",
]

NULL_QUERIES = [
    "fill missing {column} with zero",
]

DROP_NULL_QUERIES = [
    "remove rows with missing {column}",
]

STRING_UPPER = [
    "convert {column} to uppercase",
]

STRING_LOWER = [
    "convert {column} to lowercase",
]

WINDOW_QUERIES = [
    "rank {value} within {group}",
]

DATE_QUERIES = [
    "monthly total {value}",
    "monthly {value}",
]


def random_schema():

    schema = {}

    n_numeric = random.randint(1, 3)
    n_string = random.randint(1, 3)
    n_date = random.randint(0, 1)

    for c in random.sample(NUMERIC_COLUMNS, n_numeric):
        schema[c] = random.choice(["int", "float"])

    for c in random.sample(STRING_COLUMNS, n_string):
        schema[c] = "string"

    for c in random.sample(DATE_COLUMNS, n_date):
        schema[c] = "date"

    return schema


def schema_text(schema):

    return "\n".join(
        f"{k}: {v}"
        for k, v in schema.items()
    )


def aggregation(schema):

    numeric = [
        c for c, t in schema.items()
        if t in ["int", "float"]
    ]

    groups = [
        c for c, t in schema.items()
        if t == "string"
    ]

    if not numeric or not groups:
        return None

    value = random.choice(numeric)
    group = random.choice(groups)

    agg_text, agg_func = random.choice(AGGREGATIONS)

    query = random.choice(AGG_QUERIES).format(
        agg=agg_text,
        value=value,
        group=group,
    )

    code = f'''df.group_by("{group}").agg(
    pl.col("{value}").{agg_func}()
)'''

    return query, code


def filter_gt(schema):

    numeric = [
        c for c, t in schema.items()
        if t in ["int", "float"]
    ]

    if not numeric:
        return None

    column = random.choice(numeric)

    threshold = random.choice(
        [10, 25, 50, 100, 1000]
    )

    query = random.choice(FILTER_GT).format(
        column=column,
        value=threshold,
    )

    code = f'''df.filter(
    pl.col("{column}") > {threshold}
)'''

    return query, code


def filter_lt(schema):

    numeric = [
        c for c, t in schema.items()
        if t in ["int", "float"]
    ]

    if not numeric:
        return None

    column = random.choice(numeric)

    threshold = random.choice(
        [10, 25, 50, 100]
    )

    query = random.choice(FILTER_LT).format(
        column=column,
        value=threshold,
    )

    code = f'''df.filter(
    pl.col("{column}") < {threshold}
)'''

    return query, code


def sort_operation(schema):

    numeric = [
        c for c, t in schema.items()
        if t in ["int", "float"]
    ]

    if not numeric:
        return None

    column = random.choice(numeric)

    query = random.choice(SORT_QUERIES).format(
        column=column
    )

    code = f'''df.sort(
    "{column}",
    descending=True
).head(10)'''

    return query, code


def fill_null(schema):

    numeric = [
        c for c, t in schema.items()
        if t in ["int", "float"]
    ]

    if not numeric:
        return None

    column = random.choice(numeric)

    query = random.choice(NULL_QUERIES).format(
        column=column
    )

    code = f'''df.with_columns(
    pl.col("{column}").fill_null(0)
)'''

    return query, code


def drop_null(schema):

    column = random.choice(list(schema.keys()))

    query = random.choice(
        DROP_NULL_QUERIES
    ).format(column=column)

    code = f'''df.drop_nulls(["{column}"])'''

    return query, code


def uppercase(schema):

    strings = [
        c for c, t in schema.items()
        if t == "string"
    ]

    if not strings:
        return None

    column = random.choice(strings)

    query = random.choice(
        STRING_UPPER
    ).format(column=column)

    code = f'''df.with_columns(
    pl.col("{column}").str.to_uppercase()
)'''

    return query, code


def lowercase(schema):

    strings = [
        c for c, t in schema.items()
        if t == "string"
    ]

    if not strings:
        return None

    column = random.choice(strings)

    query = random.choice(
        STRING_LOWER
    ).format(column=column)

    code = f'''df.with_columns(
    pl.col("{column}").str.to_lowercase()
)'''

    return query, code


def window(schema):

    numeric = [
        c for c, t in schema.items()
        if t in ["int", "float"]
    ]

    strings = [
        c for c, t in schema.items()
        if t == "string"
    ]

    if not numeric or not strings:
        return None

    value = random.choice(numeric)
    group = random.choice(strings)

    query = random.choice(
        WINDOW_QUERIES
    ).format(
        value=value,
        group=group,
    )

    code = f'''df.with_columns(
    pl.col("{value}")
    .rank()
    .over("{group}")
)'''

    return query, code


def monthly(schema):

    dates = [
        c for c, t in schema.items()
        if t == "date"
    ]

    numeric = [
        c for c, t in schema.items()
        if t in ["int", "float"]
    ]

    if not dates or not numeric:
        return None

    date_col = random.choice(dates)
    value = random.choice(numeric)

    query = random.choice(
        DATE_QUERIES
    ).format(value=value)

    code = f'''df.group_by_dynamic(
    "{date_col}",
    every="1mo"
).agg(
    pl.col("{value}").sum()
)'''

    return query, code


GENERATORS = [
    aggregation,
    filter_gt,
    filter_lt,
    sort_operation,
    fill_null,
    drop_null,
    uppercase,
    lowercase,
    window,
    monthly,
]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    count = 0

    while count < NUM_EXAMPLES:

        schema = random_schema()

        generator = random.choice(
            GENERATORS
        )

        result = generator(schema)

        if result is None:
            continue

        query, response = result

        schema_block = schema_text(schema)

        example = f"""### Schema
{schema_block}

### Query
{query}

### Response
{response}

<eos>

"""

        f.write(example)

        count += 1

        if count % 10000 == 0:
            print(
                f"Generated {count:,} examples"
            )

print("Finished.")