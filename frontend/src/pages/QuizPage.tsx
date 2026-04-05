import { useQuizMode } from "../hooks/useQuizMode";
import QuizSetup from "../components/QuizSetup";
import ActiveQuiz from "../components/ActiveQuiz";
import QuizResults from "../components/QuizResults";

export default function QuizPage() {
  const {
    screen, questions, current, selected, setSelected,
    answers, timeLeft, timeLimitSec, analyses, analyzing,
    score, timerWarning, timerCritical,
    startQuiz, confirmAnswer, reset,
  } = useQuizMode();

  return (
    <div>
      {screen === "setup" && (
        <QuizSetup onStart={startQuiz} />
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
          onRetry={reset}
        />
      )}
    </div>
  );
}
