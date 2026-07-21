import json

from api.schemas import QuizResponse

def parse_quiz_messages(raw_response: str) -> QuizResponse:
    data = json.loads(raw_response)
    return QuizResponse.model_validate(data)    
