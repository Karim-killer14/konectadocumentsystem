# validator.py (final adaptive validator)

def validate_doc(document):
    """
    Adaptive validation based on doc_type.
    Only checks the required fields for that document class.
    """

    doc_type = document.get("doc_type", "unknown")
    fields = document.get("fields", {})
    confidence = document.get("confidence", {})
    issues = document.get("issues", [])

    # -----------------------------------------
    # REQUIRED FIELD SETS
    # -----------------------------------------
    REQUIRED = {
        "invoice": ["invoice_number", "vendor", "date", "amount", "currency"],
        "po": ["po_number", "vendor", "date", "total", "currency", "department"],
        "approval": ["request_id", "requested_by", "department", "purpose", "amount"],
        "unknown": []  # don't require anything
    }

    OPTIONAL = {
        "approval": ["approver", "status"],
        "invoice": ["line_items"],
        "po": ["line_items"],
        "unknown": []
    }

    required_fields = REQUIRED.get(doc_type, [])
    optional_fields = OPTIONAL.get(doc_type, [])

    field_results = {}
    suggestions = []

    # -----------------------------------------
    # Validate required fields
    # -----------------------------------------
    for field in required_fields:
        if field not in fields or fields[field] in [None, "", []]:
            field_results[field] = {
                "ok": False,
                "reason": "missing"
            }
            suggestions.append(f"Missing required field: {field}")
        else:
            field_results[field] = {
                "ok": True,
                "reason": "",
                "confidence": confidence.get(field, 0.5)
            }

    # -----------------------------------------
    # Optional fields: warn if missing
    # -----------------------------------------
    for field in optional_fields:
        if field not in fields or not fields[field]:
            suggestions.append(f"Optional field missing: {field}")

    # -----------------------------------------
    # Add internal extraction issues
    # -----------------------------------------
    for issue in issues:
        suggestions.append(f"Issue detected: {issue}")

    # Document is valid only if all required fields are present
    valid = all(v["ok"] for v in field_results.values())

    return {
        "valid": valid,
        "field_results": field_results,
        "suggestions": suggestions
    }
