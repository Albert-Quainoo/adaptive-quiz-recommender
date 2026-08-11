import inspect

from app.ui import content_gap, feedback, login, progress, question
from app.view_models import (
    ContentGapViewModel,
    ProgressViewModel,
    QuestionOptionViewModel,
    QuestionViewModel,
    SubmissionResultViewModel,
)


class FakeColumn:
    def __init__(self, calls):
        self.calls = calls

    def metric(self, *args, **kwargs):
        self.calls.append(("metric", args, kwargs))


class FakeStreamlit:
    def __init__(
        self, *, learner_id="", submitted=False, selected_option_id=None, next_clicked=False
    ):
        self.learner_id = learner_id
        self.submitted = submitted
        self.selected_option_id = selected_option_id
        self.next_clicked = next_clicked
        self.calls = []
        self.sidebar = self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record

    def form(self, key):
        self.calls.append(("form", (key,), {}))
        return self

    def text_input(self, *args, **kwargs):
        self.calls.append(("text_input", args, kwargs))
        return self.learner_id

    def form_submit_button(self, *args, **kwargs):
        self.calls.append(("form_submit_button", args, kwargs))
        return self.submitted

    def radio(self, *args, **kwargs):
        self.calls.append(("radio", args, kwargs))
        return self.selected_option_id

    def button(self, *args, **kwargs):
        self.calls.append(("button", args, kwargs))
        return self.next_clicked

    def columns(self, count):
        return [FakeColumn(self.calls) for _ in range(count)]

    def progress(self, *args, **kwargs):
        self.calls.append(("progress", args, kwargs))


def question_view():
    return QuestionViewModel(
        presentation_id="presentation-1",
        item_id="item-1",
        skill_id="AI-APP-01",
        topic="Integration",
        concept="Contracts",
        difficulty="introductory",
        question_text="Which option is correct?",
        options=[
            QuestionOptionViewModel(option_id="option-a", text="Answer A"),
            QuestionOptionViewModel(option_id="option-b", text="Answer B"),
        ],
        recommendation_reason="Start with a foundational skill.",
    )


def test_component_signatures_match_the_streamlit_shell():
    assert str(inspect.signature(login.render_login)) == (
        "(*, default_learner_id: str = '') -> str | None"
    )
    assert list(inspect.signature(question.render_question).parameters) == [
        "question",
        "selected_option_id",
        "disabled",
    ]
    assert list(inspect.signature(feedback.render_feedback).parameters) == [
        "result",
        "next_enabled",
    ]
    assert list(inspect.signature(progress.render_progress).parameters) == ["progress"]
    assert list(inspect.signature(content_gap.render_content_gap).parameters) == ["gap"]


def test_login_returns_only_a_submitted_normalized_learner_id(monkeypatch):
    rendered = FakeStreamlit(learner_id="  learner-1  ", submitted=True)
    monkeypatch.setattr(login, "st", rendered)

    assert login.render_login(default_learner_id="suggested") == "learner-1"
    text_input = next(call for call in rendered.calls if call[0] == "text_input")
    assert text_input[2]["value"] == "suggested"


def test_question_returns_selection_and_button_event(monkeypatch):
    rendered = FakeStreamlit(
        selected_option_id="option-b", submitted=True
    )
    monkeypatch.setattr(question, "st", rendered)

    assert question.render_question(
        question_view(), selected_option_id="option-b", disabled=False
    ) == ("option-b", True)
    form = next(call for call in rendered.calls if call[0] == "form")
    radio = next(call for call in rendered.calls if call[0] == "radio")
    assert form[1] == ("question_form_presentation-1",)
    assert radio[2]["options"] == ["option-a", "option-b"]
    assert radio[2]["index"] == 1
    assert radio[2]["format_func"]("option-a") == "Answer A"


def test_feedback_honors_next_enabled(monkeypatch):
    rendered = FakeStreamlit(next_clicked=False)
    monkeypatch.setattr(feedback, "st", rendered)
    result = SubmissionResultViewModel(
        attempt_id="attempt-1",
        correct=False,
        correct_option_id="option-a",
        correct_answer_text="Answer A",
        explanation="Answer A is correct.",
        previous_mastery=0.2,
        updated_mastery=0.4,
        attempt_count=1,
    )

    assert feedback.render_feedback(result, next_enabled=False) is False
    button = next(call for call in rendered.calls if call[0] == "button")
    assert button[2]["disabled"] is True


def test_progress_uses_probability_scale_and_readable_reason(monkeypatch):
    rendered = FakeStreamlit()
    monkeypatch.setattr(progress, "st", rendered)
    view = ProgressViewModel(
        learner_id="learner-1",
        skill_id="AI-APP-01",
        skill_name="Application contracts",
        mastery_probability=0.8,
        attempt_count=3,
        difficulty="advanced",
        readable_recommendation_reason="Ready for advanced practice.",
    )

    progress.render_progress(view)

    progress_call = next(call for call in rendered.calls if call[0] == "progress")
    caption = next(call for call in rendered.calls if call[0] == "caption")
    info = next(call for call in rendered.calls if call[0] == "info")
    assert progress_call[1] == (0.8,)
    assert progress_call[2]["text"] == "Mastery: 80%"
    assert "Attempts: 3" in caption[1][0]
    assert info[1] == ("Ready for advanced practice.",)


def test_content_gap_displays_progression_and_inventory_details(monkeypatch):
    rendered = FakeStreamlit()
    monkeypatch.setattr(content_gap, "st", rendered)
    gap = ContentGapViewModel(
        completed_skill_id="AI-FND-01",
        completed_skill_name="Artificial intelligence",
        newly_unlocked_skill_id="AI-AGT-01",
        newly_unlocked_skill_name="Agent and environment",
        missing_approved_content=True,
        current_mastery_probability=0.8,
        prerequisite_mastery_threshold=0.75,
    )

    content_gap.render_content_gap(gap)

    rendered_text = "\n".join(
        str(call[1][0]) for call in rendered.calls if call[0] in {"warning", "write"}
    )
    assert "Completed/mastered skill" in rendered_text
    assert "AI-FND-01" in rendered_text
    assert "Newly unlocked skill" in rendered_text
    assert "AI-AGT-01" in rendered_text
    assert "Missing approved content" in rendered_text
    assert "Current mastery: 80%" in rendered_text
    assert "Prerequisite threshold: 75%" in rendered_text
