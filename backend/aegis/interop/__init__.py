"""The integration plane: how a hospital's feeds become what AegisOS watches.

    hl7.py        HL7 v2 codec: parse/build/escape, ACKs with table-0357 codes, MLLP framing
    mllp.py       the TCP listener an on-premises interface engine connects to (AEGIS_MLLP_PORT)
    units.py      written units → tokens; dimension-level conversions
    vocab.py      the canonical concepts (LOINC, unit, physiological window, analyte
                  conversions) and their reference distributions (reference.json)
    evidence.py   does the data agree with a mapping? plausibility, location, tail mass,
                  physiological identities across fields, unit inference, live drift checks
    proposer.py   LOINC, IEEE 11073, lexical and language-model proposers (they only propose)
    mapping.py    one local code on one interface, its status and its history
    engine.py     the inbound interface engine: census, mappings earned on evidence,
                  delivery into rooms, live quarantine, the Sentinel's view
    sender.py     a simulated hospital (ADT, monitor gateway, two labs) that speaks only HL7

Reference distributions are rebuilt by ``make interop-reference``; ``make verify-interop``
checks the whole plane.
"""
