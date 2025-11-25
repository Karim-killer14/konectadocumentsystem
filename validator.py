# validator.py

def validate_doc(document):
    """
    Adaptive validator per document type.
    Input:
      { "doc_type":..., "fields": {...}, "confidence": {...}, "issues": [...] }
    Returns:
      { "valid": bool, "field_results": {...}, "suggestions": [...] }
    """

    doc_type = document.get("doc_type", "unknown")
    fields = document.get("fields", {}) or {}
    confidence = document.get("confidence", {}) or {}
    issues = document.get("issues", []) or []

    REQUIRED = {
        "invoice": ["invoice_number", "vendor", "date", "total"],
        "po": ["po_number", "vendor", "date", "total"],
        "approval": ["request_id", "requested_by", "department", "purpose", "amount"],
        "unknown": []
    }

    OPTIONAL = {
        "invoice": ["line_items"],
        "po": ["delivery_date"],
        "approval": ["approver", "status"],
        "unknown": []
    }

    required = REQUIRED.get(doc_type, [])
    optional = OPTIONAL.get(doc_type, [])

    field_results = {}
    suggestions = []

    for f in required:
        if f not in fields or fields.get(f) in [None, "", []]:
            field_results[f] = {"ok": False, "reason": "missing"}
            suggestions.append(f"Missing required field: {f}")
        else:
            field_results[f] = {"ok": True, "reason": "", "confidence": confidence.get(f)}

    for f in optional:
        if f not in fields or fields.get(f) in [None, "", []]:
            suggestions.append(f"Optional field missing: {f}")

    for it in issues:
        suggestions.append(f"Issue detected: {it}")

    valid = all(v.get("ok", False) for v in field_results.values()) if field_results else True

    return {"valid": valid, "field_results": field_results, "suggestions": suggestions}
