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
    explanation: str = ""


class AttemptResponse(BaseModel):
    id: int
    question_text: str
    correct_answer: str
    llm_answer: str
    user_marked_correct: bool
    concepts: list[str]
    created_at: str
    options: list[str] = []
    explanation: str = ""


class WeakConceptsResponse(BaseModel):
    concepts: list[str]


class ExplainResponse(BaseModel):
    explanation: str


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


class PracticeQuestion(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    from_history: bool = False


class UploadPracticeResponse(BaseModel):
    questions: list[PracticeQuestion]
    total: int


class WrongItem(BaseModel):
    question: str
    options: list[str]
    correct_answer: str
    user_answer: str


class AnalysisItem(BaseModel):
    question: str
    correct_answer: str
    user_answer: str
    explanation: str


class AnalyzeRequest(BaseModel):
    wrong_items: list[WrongItem]


class AnalyzeResponse(BaseModel):
    analyses: list[AnalysisItem]


class BatchAttemptItem(BaseModel):
    question_text: str
    options: list[str]
    correct_answer: str
    user_answer: str
    user_marked_correct: bool


class BatchRecordRequest(BaseModel):
    attempts: list[BatchAttemptItem]


class MediaRecommendRequest(BaseModel):
    concept: str


class SongRec(BaseModel):
    title: str
    artist: str


class AnimeRec(BaseModel):
    title: str
    scene: str


class ArticleRec(BaseModel):
    title: str
    keywords: str


class MediaRecommendResponse(BaseModel):
    songs: list[SongRec]
    anime: list[AnimeRec]
    articles: list[ArticleRec]
