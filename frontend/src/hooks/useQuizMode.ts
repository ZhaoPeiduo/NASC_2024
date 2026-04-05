import { useState, useCallback, useRef, useEffect } from "react";
import { api } from "../api/client";
import type { QuizQuestion, AnalysisItem } from "../types";

export type QuizScreen = "setup" | "active" | "results";

interface QuizAnswer {
  questionIndex: number;
  userAnswer: string;
  correctAnswer: string;
  isCorrect: boolean;
  question: string;
  options: string[];
}

export function useQuizMode() {
  const [screen, setScreen] = useState<QuizScreen>("setup");
  const [questions, setQuestions] = useState<QuizQuestion[]>([]);
  const [current, setCurrent] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [answers, setAnswers] = useState<QuizAnswer[]>([]);
  const [timeLeft, setTimeLeft] = useState(0);
  const [timeLimitSec, setTimeLimitSec] = useState(0);
  const [analyses, setAnalyses] = useState<AnalysisItem[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = () => {
    if (timerRef.current) { clearInterval(timerRef.current); timerRef.current = null; }
  };

  const finishQuiz = useCallback(async (finalAnswers: QuizAnswer[]) => {
    stopTimer();
    setScreen("results");
    setAnalyzing(true);

    const wrong = finalAnswers
      .filter(a => !a.isCorrect)
      .map(a => ({
        question: a.question,
        options: a.options,
        correct_answer: a.correctAnswer,
        user_answer: a.userAnswer,
      }));

    // Record all attempts to history
    await api.recordBatch(
      finalAnswers.map(a => ({
        question_text: a.question,
        options: a.options,
        correct_answer: a.correctAnswer,
        user_answer: a.userAnswer,
        user_marked_correct: a.isCorrect,
      }))
    ).catch(() => {/* non-fatal */});

    // Get LLM analysis for wrong answers
    if (wrong.length > 0) {
      try {
        const res = await api.analyzePractice(wrong);
        setAnalyses(res.analyses);
      } catch { setAnalyses([]); }
    }
    setAnalyzing(false);
  }, []);

  const startQuiz = useCallback((qs: QuizQuestion[], timeSec: number) => {
    setQuestions(qs);
    setCurrent(0);
    setSelected(null);
    setAnswers([]);
    setAnalyses([]);
    setTimeLimitSec(timeSec);
    setTimeLeft(timeSec);
    setScreen("active");
  }, []);

  // Countdown timer
  useEffect(() => {
    if (screen !== "active" || timeLimitSec === 0) return;
    timerRef.current = setInterval(() => {
      setTimeLeft(t => {
        if (t <= 1) {
          // time's up — finish with current answers
          setAnswers(prev => {
            finishQuiz(prev);
            return prev;
          });
          return 0;
        }
        return t - 1;
      });
    }, 1000);
    return stopTimer;
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen, timeLimitSec]);

  const confirmAnswer = useCallback(() => {
    if (!selected || current >= questions.length) return;
    const q = questions[current];
    const isCorrect = selected === q.correct_answer;
    const newAnswer: QuizAnswer = {
      questionIndex: current,
      userAnswer: selected,
      correctAnswer: q.correct_answer,
      isCorrect,
      question: q.question,
      options: q.options,
    };
    const newAnswers = [...answers, newAnswer];
    setAnswers(newAnswers);
    setSelected(null);

    if (current + 1 >= questions.length) {
      finishQuiz(newAnswers);
    } else {
      setCurrent(c => c + 1);
    }
  }, [selected, current, questions, answers, finishQuiz]);

  const reset = () => {
    stopTimer();
    setScreen("setup");
    setQuestions([]);
    setCurrent(0);
    setSelected(null);
    setAnswers([]);
    setAnalyses([]);
  };

  const score = answers.filter(a => a.isCorrect).length;
  const timerWarning = timeLimitSec > 0 && timeLeft < 60;
  const timerCritical = timeLimitSec > 0 && timeLeft < 30;

  return {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset,
  };
}
