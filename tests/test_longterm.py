"""Long-term (cross-episode) memory — offline, no model or network."""

from agent_hospital.graph import longterm as lt

_CATH = "A man with eosinophilia after cardiac catheterization and livedo reticularis renal failure"
_CISPLATIN = "A patient receiving cisplatin chemotherapy develops hearing loss ototoxicity"


def _store(tmp_path):
    return lt.open_store(str(tmp_path / "lessons.sqlite"))


def test_extract_lesson_reads_the_line():
    reply = "Keeping B.\nLesson: Eosinophilia after catheterization suggests cholesterol emboli.\nAnswer: B"
    assert lt.extract_lesson(reply) == "Eosinophilia after catheterization suggests cholesterol emboli."


def test_extract_lesson_absent_is_empty():
    assert lt.extract_lesson("No lesson at all.\nAnswer: C") == ""
    assert lt.extract_lesson("") == ""


def test_cold_store_recalls_nothing(tmp_path):
    store, close = _store(tmp_path)
    try:
        assert lt.recall(store, "train", _CATH) == ""
    finally:
        close()


def test_lesson_round_trips(tmp_path):
    store, close = _store(tmp_path)
    try:
        lt.remember(store, "train", "q1", _CATH, "Eosinophilia plus catheterization means cholesterol embolization.")
        assert "cholesterol embolization" in lt.recall(store, "train", _CATH)
    finally:
        close()


def test_recall_filters_by_topic(tmp_path):
    """An unrelated lesson must not be pasted into the prompt — a misleading precedent is
    worse than none, and `search(query=...)` alone does not rank without a vector index."""
    store, close = _store(tmp_path)
    try:
        lt.remember(store, "train", "q1", _CATH, "Cholesterol embolization lesson.")
        lt.remember(store, "train", "q2", _CISPLATIN, "Cisplatin ototoxicity lesson.")
        assert "Cisplatin" not in lt.recall(store, "train", _CATH)
        assert "Cholesterol" not in lt.recall(store, "train", _CISPLATIN)
        assert lt.recall(store, "train", "child with asthma wheezing albuterol") == ""
    finally:
        close()


def test_lesson_key_drops_the_split_prefix():
    """Lessons are named for what they are; the split already lives in the namespace."""
    assert lt.lesson_key("train-00000") == "lesson-00000"
    assert lt.lesson_key("test-01272") == "lesson-01272"
    assert lt.lesson_key("weird") == "lesson-weird"


def test_stored_under_lesson_key_with_source_item(tmp_path):
    store, close = _store(tmp_path)
    try:
        lt.remember(store, "train", "train-00007", _CATH, "Cholesterol embolization lesson.")
        item = store.get(lt.namespace("train"), "lesson-00007")
        assert item is not None and item.value["item_id"] == "train-00007"
    finally:
        close()


def test_read_only_blocks_writes(tmp_path):
    """The guard that keeps a scored split from leaking between its own items."""
    store, close = _store(tmp_path)
    try:
        lt.remember(store, "test", "test-00001", _CATH, "should not persist", read_only=True)
        assert store.get(lt.namespace("test"), lt.lesson_key("test-00001")) is None
    finally:
        close()


def test_warmup_item_is_not_stored(tmp_path):
    """The CLI's synthetic warm-up runs the real graph; its lesson must not be kept."""
    store, close = _store(tmp_path)
    try:
        lt.remember(store, "train", "warmup", _CATH, "throwaway lesson")
        assert store.get(lt.namespace("train"), lt.lesson_key("warmup")) is None
    finally:
        close()


def test_splits_are_isolated(tmp_path):
    """train-built lessons must be invisible when the namespace is test."""
    store, close = _store(tmp_path)
    try:
        lt.remember(store, "train", "q1", _CATH, "Cholesterol embolization lesson.")
        assert lt.recall(store, "test", _CATH) == ""
    finally:
        close()


def test_recall_survives_a_missing_store():
    """A cold/broken store degrades gracefully rather than breaking an eval run."""
    assert lt.recall(None, "train", _CATH) == ""
    lt.remember(None, "train", "q1", _CATH, "x")   # must not raise


# --- mistake bank (V5): separate namespace root, gold-conditioned writer -----------

def test_mistake_key_drops_the_split_prefix():
    assert lt.mistake_key("train-00000") == "mistake-00000"
    assert lt.mistake_key("test-01272") == "mistake-01272"


def test_cold_store_recalls_no_mistakes(tmp_path):
    store, close = _store(tmp_path)
    try:
        assert lt.recall_mistakes(store, "train", _CATH) == ""
    finally:
        close()


def test_mistake_round_trips(tmp_path):
    store, close = _store(tmp_path)
    try:
        lt.remember_mistake(store, "train", "q1", _CATH,
                             "Eosinophilia plus catheterization means cholesterol embolization.",
                             wrong="A", correct="B")
        assert "cholesterol embolization" in lt.recall_mistakes(store, "train", _CATH)
    finally:
        close()


def test_mistake_recall_filters_by_topic(tmp_path):
    store, close = _store(tmp_path)
    try:
        lt.remember_mistake(store, "train", "q1", _CATH, "Cholesterol embolization lesson.",
                             wrong="A", correct="B")
        lt.remember_mistake(store, "train", "q2", _CISPLATIN, "Cisplatin ototoxicity lesson.",
                             wrong="C", correct="D")
        assert "Cisplatin" not in lt.recall_mistakes(store, "train", _CATH)
        assert "Cholesterol" not in lt.recall_mistakes(store, "train", _CISPLATIN)
    finally:
        close()


def test_mistake_stored_under_mistake_key_with_wrong_and_correct(tmp_path):
    store, close = _store(tmp_path)
    try:
        lt.remember_mistake(store, "train", "train-00007", _CATH, "Cholesterol embolization lesson.",
                             wrong="A", correct="B")
        item = store.get(lt.mistake_namespace("train"), "mistake-00007")
        assert item is not None
        assert item.value["wrong"] == "A" and item.value["correct"] == "B"
    finally:
        close()


def test_mistake_read_only_blocks_writes(tmp_path):
    store, close = _store(tmp_path)
    try:
        lt.remember_mistake(store, "test", "test-00001", _CATH, "should not persist",
                             wrong="A", correct="B", read_only=True)
        assert store.get(lt.mistake_namespace("test"), lt.mistake_key("test-00001")) is None
    finally:
        close()


def test_mistake_warmup_item_is_not_stored(tmp_path):
    store, close = _store(tmp_path)
    try:
        lt.remember_mistake(store, "train", "warmup", _CATH, "throwaway", wrong="A", correct="B")
        assert store.get(lt.mistake_namespace("train"), lt.mistake_key("warmup")) is None
    finally:
        close()


def test_mistake_splits_are_isolated(tmp_path):
    store, close = _store(tmp_path)
    try:
        lt.remember_mistake(store, "train", "q1", _CATH, "Cholesterol embolization lesson.",
                             wrong="A", correct="B")
        assert lt.recall_mistakes(store, "test", _CATH) == ""
    finally:
        close()


def test_general_and_mistake_banks_do_not_leak_into_each_other(tmp_path):
    """Same item, same question, both banks — a lesson written to one must never surface
    when recalling from the other."""
    store, close = _store(tmp_path)
    try:
        lt.remember(store, "train", "q1", _CATH, "General lesson about cholesterol emboli.")
        lt.remember_mistake(store, "train", "q1", _CATH, "Mistake lesson about cholesterol emboli.",
                             wrong="A", correct="B")
        assert "Mistake lesson" not in lt.recall(store, "train", _CATH)
        assert "General lesson" not in lt.recall_mistakes(store, "train", _CATH)
    finally:
        close()
