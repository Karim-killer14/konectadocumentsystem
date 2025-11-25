# validator.py
"""
Validator returns pass/fail per-field and suggestions.
"""
from typing import Dict, Any
import re

def validate_fields(doc):
    """
    doc: {"doc_type":..., "fields": {...}, "confidence": {...}, "issues":[...]}
    Returns: {"valid": bool, "field_results": {k: {ok:bool, reason:str}}, "suggestions": [...]}
    """
    fields = doc.get("fields", {})
    conf = doc.get("confidence", {})
    issues = doc.get("issues", [])

    doc_type = doc.get("doc_type", "invoice")
    required = []
    if doc_type == "invoice" or doc_type == "purchase_order":
        required = ["document_id", "date", "amount", "currency"]
    elif doc_type == "approval":
        required = ["document_id", "date", "amount", "currency", "approver", "status"]

    field_results = {}
    suggestions = []

    # Check required fields present
    for f in required:
        if f not in fields:
            field_results[f] = {"ok": False, "reason": "missing"}
            suggestions.append(f"Missing required field: {f}")
        else:
            # lightweight checks
            val = fields[f]
            ok = True
            reason = ""
            if f == "amount":
                try:
                    v = float(val)
                    if v <= 0:
                        ok = False
                        reason = "non_positive"
                except:
                    ok = False
                    reason = "not_numeric"
            if f == "date":
                # basic ISO-like check
                if not isinstance(val, str) or not re.match(r'^\d{4}-\d{2}-\d{2}', val):
                    ok = False
                    reason = "invalid_format"
            if f == "currency":
                if not isinstance(val, str) or not re.match(r'^[A-Z]{3}$', val):
                    ok = False
                    reason = "invalid_currency"
            field_results[f] = {"ok": ok, "reason": reason, "confidence": conf.get(f)}

    # Check for suspicious issues flagged earlier
    for it in issues:
        suggestions.append(f"Issue detected: {it}")

    overall_ok = all(v.get("ok", False) for v in field_results.values())
    return {"valid": overall_ok, "field_results": field_results, "suggestions": suggestions}
