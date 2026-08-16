RHY_MAP = {
    "N": "N",
    "AFIB": "AFIB",
    "AFL": "AFL",
    "AFLUT": "AFL",      # if present
    "AFLUTTER": "AFL",   # if present
}

def normalize_rhythm_token(tok: str):
    tok = tok.strip().upper()
    tok = tok.replace("(", "").replace(")", "").strip()
    # e.g. "(N" -> "N", "(AFIB" -> "AFIB"
    return RHY_MAP.get(tok, None)

def build_rhythm_intervals_from_ann(ann, sig_len: int):
    """
    Parse WFDB rhythm annotations into non-overlapping labeled intervals.

    Behavior:
    - On "(LABEL": opens a segment at sample s.
      If a segment is already open, it is FORCE-CLOSED at s (new segment boundary).
    - On ")": closes the current segment at sample s (if open).
    - At end of annotation stream: closes any open segment at last annotation sample.

    Returns: list of (start, end, label) in samples, with end > start.
    """
    intervals = []
    cur_label = None
    cur_start = None

    def close_at(end_s: int):
        nonlocal cur_label, cur_start, intervals
        if cur_start is not None and cur_label is not None and end_s is not None and end_s > cur_start:
            intervals.append((int(cur_start), int(end_s), cur_label))
        cur_label = None
        cur_start = None

    for s, note in zip(ann.sample, ann.aux_note):
        if note is None:
            continue
        note = str(note).strip()
        if note == "" or note.upper() == "NONE":
            continue
        s = int(s)
        note = note.strip()

        if note.startswith("("):
            # Force-close prior segment at the boundary (even if the new token is unknown)
            if cur_start is not None:
                close_at(s)

            lab = normalize_rhythm_token(note)
            if lab is not None:
                cur_label = lab
                cur_start = s
            else:
                # Unknown rhythm start token: treat as boundary but do not open a labeled segment
                print("Warning: unknown rhythm token", note)
                cur_label = None
                cur_start = None

        elif note.startswith(")"):
            # Close current segment at s (if open)
            close_at(s)

    # Close any dangling open segment
    if cur_start is not None and cur_label is not None and len(ann.sample) > 0:
        close_at(sig_len)

    return intervals

def label_window_by_occupancy(intervals, w_start, w_end, thresh=0.10):
    """
    This function acts as metadata for SSL only.
    intervals: list of (start, end, label) in samples
    window: [w_start, w_end)
    """
    dur = w_end - w_start
    if dur <= 0:
        return None

    occ = {"AFIB": 0, "AFL": 0}  # N is default remainder
    for a, b, lab in intervals:
        if b <= w_start or a >= w_end:
            continue
        overlap = max(0, min(b, w_end) - max(a, w_start))
        if lab in occ:
            occ[lab] += overlap

    if occ["AFIB"] / dur > thresh:
        return "AFIB"
    if occ["AFL"] / dur > thresh:
        return "AFL"
    return "N"

# The one used in evaluation: only label if sufficiently covered by known rhythms
def label_window_strict(intervals, w_start, w_end, thresh=0.05, min_covered=0.95):
    """
    Strict window labeler:
    - Only labels windows that are sufficiently covered by known intervals (N/AFIB/AFL).
    - If coverage < min_covered => return None (treat as OTHER/UNLABELED and skip).
    - If covered, returns: "AFIB", "AFL", or "N" using the same occupancy logic.

    intervals must contain only labels from your RHY_MAP (N/AFIB/AFL).
    """
    dur = w_end - w_start
    if dur <= 0:
        return None

    # Coverage of known labels (N/AFIB/AFL) within this window
    covered = 0
    occ = {"AFIB": 0, "AFL": 0, "N": 0}

    for a, b, lab in intervals:
        if b <= w_start or a >= w_end:
            continue
        overlap = max(0, min(b, w_end) - max(a, w_start))
        if overlap <= 0:
            continue
        if lab in occ:
            occ[lab] += overlap
            covered += overlap

    # If too much of the window is not covered by known rhythms, treat it as OTHER/UNLABELED
    if covered / dur < min_covered:
        return None

    # Now decide label (AFIB/AFL/N)
    if occ["AFIB"] / dur > thresh:
        return "AFIB"
    if occ["AFL"] / dur > thresh:
        return "AFL"
    return "N"