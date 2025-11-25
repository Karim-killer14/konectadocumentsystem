# validate.py
import re

def validate_doc(extracted):
    """
    extracted: { "doc_type":..., "fields":{...}, "confidence":{...}, "issues":[...] }
    Returns: { "valid": bool, "field_results": {...}, "suggestions":[...] }
    """
    doc_type = extracted.get("doc_type", "unknown")
    fields = extracted.get("fields", {})
    conf = extracted.get("confidence", {})
    issues = list(extracted.get("issues", []))

    required = []
    if doc_type == "invoice" or doc_type == "purchase_order":
        required = ["document_id", "date", "amount", "currency"]
    elif doc_type == "approval":
        required = ["document_id", "date", "amount", "currency", "approver", "status"]
    else:
        # conservative baseline
        required = ["amount", "date"]

    results = {}
    suggestions = []

    for f in required:
        if f not in fields:
            results[f] = {"ok": False, "reason": "missing"}
            suggestions.append(f"Missing required field: {f}")
        else:
            ok = True
            reason = ""
            val = fields[f]
            if f == "amount":
                try:
                    v = float(val)
                    if v <= 0:
                        ok = False; reason = "non_positive"
                except:
                    ok = False; reason = "not_numeric"
            if f == "date":
                if not isinstance(val, str) or not re.match(r'^\d{4}-\d{2}-\d{2}', val):
                    ok = False; reason = "invalid_format"
            if f == "currency":
                if not isinstance(val, str) or not re.match(r'^[A-Z]{3}$', val):
                    ok = False; reason = "invalid_currency"
            results[f] = {"ok": ok, "reason": reason, "confidence": conf.get(f)}
            if not ok:
                suggestions.append(f"Field {f} looks invalid: {reason}")

    # include earlier issues as suggestions
    for it in issues:
        suggestions.append(f"Issue detected: {it}")

    overall = all(v.get("ok", False) for v in results.values()) if results else False
    return {"valid": overall, "field_results": results, "suggestions": suggestions}
