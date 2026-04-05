import { useQuizSession } from "../hooks/useQuizSession";
import QuizForm from "../components/QuizForm";
import ExplanationCard from "../components/ExplanationCard";
import StreamingProgress from "../components/StreamingProgress";
import ImageExtractor from "../components/ImageExtractor";
import RecommendationPanel from "../components/RecommendationPanel";

export default function PracticePage() {
  const {
    fields, setFields, setOption,
    phase, streamText, result, error,
    submit, reset,
  } = useQuizSession();

  const fillFromImage = (question: string, options: string[]) => {
    setFields({ question, options: [...options, "", "", "", ""].slice(0, 4) });
  };

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
      <ImageExtractor onExtract={fillFromImage} />
      <StreamingProgress phase={phase} />
      <ExplanationCard streamText={streamText} result={result} phase={phase} />
      {result && <RecommendationPanel concepts={result.concepts} />}
    </div>
  );
}
