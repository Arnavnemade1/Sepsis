"""Optional LLM layer: turns structured detector output into clinician-readable
narratives. Strictly an *explanation* layer — it never makes the detection
decision, never sees raw pixels, and is called sparingly + rate-limited."""
