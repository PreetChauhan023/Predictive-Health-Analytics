test_item_schema = {
    "type": "OBJECT",
    "properties": {
        "name":       {"type": "STRING"},
        "value":      {"type": "NUMBER"},
        "unit":       {"type": "STRING"},
        "normal_min": {"type": "NUMBER"},
        "normal_max": {"type": "NUMBER"},
        "status":     {"type": "STRING", "enum": ["Normal", "High", "Low", "Borderline"]},
        "meaning":    {"type": "STRING"}
    },
    "required": ["name", "value", "unit", "status", "meaning"]
}

LAB_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "counts": {
            "type": "OBJECT",
            "properties": {
                "normal": {"type": "INTEGER"},
                "high": {"type": "INTEGER"},
                "low": {"type": "INTEGER"},
                "borderline": {"type": "INTEGER"}
            },
            "required": ["normal", "high", "low", "borderline"]
        },
        "categories": {
            "type": "OBJECT",
            "properties": {
                "CBC": {"type": "ARRAY", "items": test_item_schema},
                "Lipid Profile": {"type": "ARRAY", "items": test_item_schema},
                "Liver Function": {"type": "ARRAY", "items": test_item_schema},
                "Kidney Function": {"type": "ARRAY", "items": test_item_schema},
                "Thyroid": {"type": "ARRAY", "items": test_item_schema},
                "Blood Sugar": {"type": "ARRAY", "items": test_item_schema},
                "Electrolytes": {"type": "ARRAY", "items": test_item_schema},
                "Vitamins": {"type": "ARRAY", "items": test_item_schema},
                "Other": {"type": "ARRAY", "items": test_item_schema}
            }
        },
        "overall_interpretation": {"type": "STRING"},
        "advice": {"type": "STRING"},
        "diet": {"type": "ARRAY", "items": {"type": "STRING"}},
        "lifestyle": {"type": "ARRAY", "items": {"type": "STRING"}},
        "doctors": {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["counts", "categories", "overall_interpretation", "advice"]
}