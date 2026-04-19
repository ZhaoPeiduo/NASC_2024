import { loadPersistedResults, useQuizMode } from "../hooks/useQuizMode";
import QuizSetup from "../components/QuizSetup";
import ActiveQuiz from "../components/ActiveQuiz";
import QuizResults from "../components/QuizResults";

export default function QuizPage() {
  const {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    wrongAttemptIds,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset, restoreResults,
    wrongAnswers,
  } = useQuizMode();

  const savedResults = loadPersistedResults();

  return (
    <div>
      {screen === "setup" && (
        <div className="space-y-3">
          <QuizSetup onStart={startQuiz} />
          {savedResults && (
            <button
              onClick={() => restoreResults(savedResults)}
              className="w-full border border-cream bg-sand hover:bg-sand/80 active:scale-[0.98] text-charcoal py-2 rounded-xl text-xs font-medium transition-all"
            >
              View Last Quiz Results
            </button>
          )}
        </div>
      )}

      {screen === "active" && questions[current] && (
        <ActiveQuiz
          question={questions[current]}
          questionNumber={current + 1}
          totalQuestions={questions.length}
          selected={selected}
          onSelect={setSelected}
          onConfirm={confirmAnswer}
          timeLeft={timeLeft}
          timeLimitSec={timeLimitSec}
          timerWarning={timerWarning}
          timerCritical={timerCritical}
        />
      )}

      {screen === "results" && (
        <QuizResults
          score={score}
          total={answers.length}
          analyses={analyses}
          analyzing={analyzing}
          wrongAttemptIds={wrongAttemptIds}
          wrongAnswers={wrongAnswers}
          onRetry={reset}
        />
      )}
    </div>
  );
}
