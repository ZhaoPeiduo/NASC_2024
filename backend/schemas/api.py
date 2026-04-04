from pydantic import BaseModel


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: int
    email: str


class SolveRequest(BaseModel):
    question: str
    options: list[str]  # ["A: ...", "B: ...", "C: ...", "D: ..."]


class AttemptRecord(BaseModel):
    question_text: str
    options: list[str]
    correct_answer: str
    llm_answer: str
    user_marked_correct: bool
    concepts: list[str]


class AttemptResponse(BaseModel):
    id: int
    question_text: str
    correct_answer: str
    llm_answer: str
    user_marked_correct: bool
    concepts: list[str]
    created_at: str


class StatsResponse(BaseModel):
    total_attempts: int
    correct_rate: float
    weak_concepts: list[str]
    study_days: int


class GenerateRequest(BaseModel):
    concept: str
    level: str = "N3"


class GeneratedQuestionResponse(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    explanation: str
    concepts: list[str]


class VideoRecommendation(BaseModel):
    title: str
    video_id: str
    thumbnail_url: str
    channel_title: str
