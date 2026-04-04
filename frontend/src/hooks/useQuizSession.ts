import { useState, useCallback } from "react";
import { api } from "../api/client";
import type { Phase, SolveResult } from "../types";

interface Fields { question: string; options: string[] }

export function useQuizSession() {
  const [fields, setFields] = useState<Fields>({ question: "", options: ["", "", "", ""] });
  const [phase, setPhase] = useState<Phase>("idle");
  const [streamText, setStreamText] = useState("");
  const [result, setResult] = useState<SolveResult | null>(null);
  const [error, setError] = useState("");

  const setOption = (i: number, val: string) =>
    setFields(f => { const opts = [...f.options]; opts[i] = val; return { ...f, options: opts }; });

  const submit = useCallback(async () => {
    if (!fields.question.trim() || fields.options.filter(o => o.trim()).length < 2) {
      setError("Enter a question and at least 2 options.");
      return;
    }
    setError(""); setStreamText(""); setResult(null); setPhase("answering");

    try {
      const optLabels = fields.options
        .filter(o => o.trim())
        .map((o, i) => `${String.fromCharCode(65 + i)}: ${o}`);

      for await (const { event, data } of api.solveStreamFetch(fields.question, optLabels)) {
        if (event === "token") {
          setStreamText(t => t + data);
          if (data.includes("EXPLANATION")) setPhase("explaining");
        } else if (event === "result") {
          setResult(JSON.parse(data) as SolveResult);
          setPhase("done");
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong. Is the backend running?");
      setPhase("idle");
    }
  }, [fields]);

  const reset = () => {
    setFields({ question: "", options: ["", "", "", ""] });
    setPhase("idle"); setStreamText(""); setResult(null); setError("");
  };

  return { fields, setFields, setOption, phase, streamText, result, error, submit, reset };
}
