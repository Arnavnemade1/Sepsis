"""Clinical text as causal, tier-S features: the infrastructure for reading MIMIC-IV-Note.

What the dataset holds decides the design. MIMIC-IV-Note 2.2 has two kinds of note, and no
nursing or physician progress notes:

  * radiology reports, read during the stay. A report enters the grid at the hour it was
    STORED (``storetime``), never when the scan was taken: a read signed six hours after the
    film was not known for those six hours. A report with no ``storetime`` is dropped, not
    guessed;
  * discharge summaries, written at discharge. The current admission's is written after the
    outcome and never used. Those of admissions that ENDED before this ICU stay began are
    history, known on arrival.

Pieces:

  ``sources``   the note tables, from CSV or the PhysioNet lake (the same TableSpecs ingest them)
  ``timing``    which note may be read at which hour; the one place availability is decided
  ``readers``   text -> a fixed set of finding scores. ``KeywordReader`` (lexicon + negation +
                uncertainty) is the baseline; ``EncoderReader`` scores the same findings with a
                local text encoder against positive and negated prototypes. Same names, same
                scale, so the two can be compared head to head and the encoder kept only if it wins
  ``features``  finding scores -> ``(stay_id, hr, name, value)`` rows in the grid's binned format,
                plus static history from prior discharge summaries
  ``synthetic`` notes with planted traps (late storetimes, the current admission's summary,
                negated findings), for ``verify_notes``

Rules this module keeps, not just documents:

  * Text never leaves the machine and is never logged or printed: readers return numbers, and
    ``features`` refuses to emit a text column. PhysioNet's agreement forbids sending notes to a
    third-party model API; an encoder here loads from local files only.
  * Every feature is tier S (suspicion): ordering a film is already a clinician's suspicion, so
    note features never enter the physiology-only primary.
  * Not wired into the modelling run yet. ``features.note_rows`` returns frames shaped for the
    grid; joining them to ``extract`` is a separate, audited step.

Run ``make verify-notes``. The manual is docs/NOTES.md.
"""
