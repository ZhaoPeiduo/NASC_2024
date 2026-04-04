from datetime import datetime, timezone
from sqlalchemy import String, Boolean, ForeignKey, DateTime, Text, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    attempts: Mapped[list["Attempt"]] = relationship(back_populates="user")


class Attempt(Base):
    __tablename__ = "attempts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[str] = mapped_column(Text, nullable=False)       # JSON array string
    correct_answer: Mapped[str] = mapped_column(String(1), nullable=False)
    llm_answer: Mapped[str] = mapped_column(String(1), nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="")
    concepts: Mapped[str] = mapped_column(Text, default="[]")         # JSON array string
    user_marked_correct: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    user: Mapped["User"] = relationship(back_populates="attempts")
