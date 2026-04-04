import { useQuizSession } from "../hooks/useQuizSession";
import QuizForm from "../components/QuizForm";
import ExplanationCard from "../components/ExplanationCard";
import StreamingProgress from "../components/StreamingProgress";

export default function PracticePage() {
  const {
    fields, setFields, setOption,
    phase, streamText, result, error,
    submit, reset,
  } = useQuizSession();

  return (
    <div>
      <h1 className="text-xl font-bold text-slate-800 mb-6">Practice</h1>
      <QuizForm
        question={fields.question}
        options={fields.options}
        onQuestionChange={q => setFields(f => ({ ...f, question: q }))}
        onOptionChange={setOption}
        onSubmit={submit}
        onReset={reset}
        disabled={phase !== "idle" && phase !== "done"}
        error={error}
      />
      <StreamingProgress phase={phase} />
      <ExplanationCard streamText={streamText} result={result} phase={phase} />
    </div>
  );
}
