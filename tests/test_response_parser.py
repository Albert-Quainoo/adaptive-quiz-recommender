import pytest
import json
from pydantic import ValidationError

from api.schemas import QuizResponse
from api.response_parser import parse_quiz_messages


def make_question(**overides):
    base = {
        "question": "What is the capital of Ghana?",
        "options": ["Accra", "Kumasi", "Tamale", "Cape Coast"],
        "correct_answer": "Accra",
        "explanation": "Accra is the capital and largest city of Ghana.",
        "concept": "West African geography",
        "difficulty": "introductory",
    }
    base.update(overides)
    return base

class TestParseResponseValidJSON:
    def test_valid_json_returns_quiz_response(self):
        raw = json.dumps({"questions": [make_question()]})

        result = parse_quiz_messages(raw)

        assert isinstance(result, QuizResponse)
        assert len(result.questions) == 1
        q = result.questions[0]
        assert q.question == "What is the capital of Ghana?"
        assert q.correct_answer == "Accra"
        assert len(q.options) == 4
        assert q.difficulty == "introductory"

    def test_valid_json_with_multiple_questions(self):
        raw = json.dumps({"questions": [make_question(), make_question(
            question="What is 2 + 2?",
            options=["3", "4", "5", "6"],
            correct_answer="4",
            concept="arithmetic",
        )]})

        result = parse_quiz_messages(raw)

        assert len(result.questions) == 2


class TestParseResponseMalformedJSON:
    @pytest.mark.parametrize("bad_raw", [
        '{"questions": [{"question": "Unterminated',
        '{"questions": [}',
        '',
        'not json at all',
        '{"questions": [{"question": "trailing comma",}]}',
    ])
    def test_malformed_json_raises_decode_error(self, bad_raw):
        with pytest.raises(json.JSONDecodeError):
            parse_quiz_messages(bad_raw)


class TestParseResponseSchemaViolations:
    def test_missing_required_field_raises_validation_error(self):
        q = make_question()
        del q["explanation"]
        raw = json.dumps({"questions": [q]})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)

    def test_invalid_field_type_raises_validation_error(self):
        raw = json.dumps({"questions": [make_question(options="Accra")]})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)

    def test_invalid_difficulty_literal_raises_validation_error(self):
        raw = json.dumps({"questions": [make_question(difficulty="expert")]})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)

    def test_options_below_min_length_raises_validation_error(self):
        raw = json.dumps({"questions": [make_question(
            options=["Accra", "Kumasi", "Tamale"]
        )]})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)

    def test_options_above_max_length_raises_validation_error(self):
        raw = json.dumps({"questions": [make_question(
            options=["Accra", "Kumasi", "Tamale", "Cape Coast", "Tema"]
        )]})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)

    def test_correct_answer_not_in_options_raises_validation_error(self):
        raw = json.dumps({"questions": [make_question(
            correct_answer="Lagos"
        )]})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)

    def test_duplicate_options_raises_validation_error(self):
        raw = json.dumps({"questions": [make_question(
            options=["Accra", "accra ", "Kumasi", "Tamale"]
        )]})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)

    def test_empty_questions_list_raises_validation_error(self):
        raw = json.dumps({"questions": []})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)

    def test_too_many_questions_raises_validation_error(self):
        raw = json.dumps({"questions": [make_question() for _ in range(11)]})

        with pytest.raises(ValidationError):
            parse_quiz_messages(raw)




