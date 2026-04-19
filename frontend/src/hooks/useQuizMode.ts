import { useState, useCallback, useRef, useEffect } from "react";
import { api } from "../api/client";
import type { QuizQuestion, AnalysisItem } from "../types";

export type QuizScreen = "setup" | "active" | "results";

const STORAGE_KEY = "quiz_last_results";

interface QuizAnswer {
  questionIndex: number;
  userAnswer: string;
  correctAnswer: string;
  isCorrect: boolean;
  question: string;
  options: string[];
}

export interface PersistedResults {
  answers: QuizAnswer[];
  analyses: AnalysisItem[];
  wrongAttemptIds: number[];
  timestamp: number;
}

export function loadPersistedResults(): PersistedResults | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    return JSON.parse(raw) as PersistedResults;
  } catch {
    return null;
  }
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
  const [wrongAttemptIds, setWrongAttemptIds] = useState<number[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const answersRef = useRef<QuizAnswer[]>([]);

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

    // Record all attempts, collect IDs
    let resolvedWrongIds: number[] = [];
    try {
      const batchResult = await api.recordBatch(
        finalAnswers.map(a => ({
          question_text: a.question,
          options: a.options,
          correct_answer: a.correctAnswer,
          user_answer: a.userAnswer,
          user_marked_correct: a.isCorrect,
        }))
      );
      const allIds = batchResult.ids ?? [];
      resolvedWrongIds = finalAnswers
        .map((a, i) => ({ isCorrect: a.isCorrect, id: allIds[i] ?? 0 }))
        .filter(x => !x.isCorrect)
        .map(x => x.id);
    } catch {/* non-fatal */}

    setWrongAttemptIds(resolvedWrongIds);

    // Get LLM analysis for wrong answers
    let resolvedAnalyses: AnalysisItem[] = [];
    if (wrong.length > 0) {
      try {
        const res = await api.analyzePractice(wrong);
        resolvedAnalyses = res.analyses;
        setAnalyses(resolvedAnalyses);
      } catch { setAnalyses([]); }
    }
    setAnalyzing(false);

    // Persist results to sessionStorage
    const toSave: PersistedResults = {
      answers: finalAnswers,
      analyses: resolvedAnalyses,
      wrongAttemptIds: resolvedWrongIds,
      timestamp: Date.now(),
    };
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(toSave)); } catch {/* ignore */}
  }, []);

  const startQuiz = useCallback((qs: QuizQuestion[], timeSec: number) => {
    setQuestions(qs);
    setCurrent(0);
    setSelected(null);
    setAnswers([]);
    setAnalyses([]);
    setWrongAttemptIds([]);
    setTimeLimitSec(timeSec);
    setTimeLeft(timeSec);
    setScreen("active");
  }, []);

  const restoreResults = useCallback((saved: PersistedResults) => {
    setAnswers(saved.answers);
    setAnalyses(saved.analyses);
    setWrongAttemptIds(saved.wrongAttemptIds);
    setScreen("results");
  }, []);

  const finishedRef = useRef(false);

  // Countdown timer
  useEffect(() => {
    if (screen !== "active" || timeLimitSec === 0) return;
    finishedRef.current = false;
    timerRef.current = setInterval(() => {
      setTimeLeft(t => {
        if (t <= 1) {
          if (!finishedRef.current) {
            finishedRef.current = true;
            setTimeout(() => finishQuiz(answersRef.current), 0);
          }
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
    answersRef.current = newAnswers;
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
    answersRef.current = [];
    setAnswers([]);
    setAnalyses([]);
    setWrongAttemptIds([]);
  };

  const score = answers.filter(a => a.isCorrect).length;
  const timerWarning = timeLimitSec > 0 && timeLeft < 60;
  const timerCritical = timeLimitSec > 0 && timeLeft < 30;

  const wrongAnswers = answers
    .filter(a => !a.isCorrect)
    .map(a => ({
      question: a.question,
      correct_answer: a.correctAnswer,
      user_answer: a.userAnswer,
      concepts: [] as string[],
    }));

  return {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    wrongAttemptIds,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset, restoreResults,
    wrongAnswers,
  };
}
