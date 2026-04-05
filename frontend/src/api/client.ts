const BASE = import.meta.env.VITE_API_BASE ?? "";

function authHeaders(): HeadersInit {
  const token = localStorage.getItem("token");
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function post<T>(path: string, body: unknown, auth = false): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(auth ? authHeaders() : {}),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { headers: authHeaders() });
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}

export const api = {
  register: (email: string, password: string) =>
    post<{ access_token: string }>("/auth/register", { email, password }),

  login: (email: string, password: string) =>
    post<{ access_token: string }>("/auth/login", { email, password }),

  me: () => get<{ id: number; email: string }>("/auth/me"),

  /** Stream-solve via fetch (supports POST body + SSE). */
  async *solveStreamFetch(
    question: string,
    options: string[]
  ): AsyncGenerator<{ event: string; data: string }> {
    const res = await fetch(`${BASE}/api/v1/quiz/solve`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders() },
      body: JSON.stringify({ question, options }),
    });
    if (!res.ok || !res.body) throw new Error("Stream failed");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      let event = "message";
      for (const line of lines) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) {
          yield { event, data: line.slice(6) };
          event = "message";
        }
      }
    }
  },

  recordAttempt: (body: {
    question_text: string;
    options: string[];
    correct_answer: string;
    llm_answer: string;
    user_marked_correct: boolean;
    concepts: string[];
  }) => post("/api/v1/history/record", body, true),

  getHistory: () =>
    get<import("../types").AttemptResponse[]>("/api/v1/history"),

  getStats: () => get<import("../types").StatsResponse>("/api/v1/stats"),

  getRecommendations: (concepts: string[]) =>
    get<import("../types").VideoRecommendation[]>(
      `/api/v1/recommendations?concepts=${concepts.join(",")}`
    ),

  generateQuestion: (concept: string, level: string) =>
    post<import("../types").GeneratedQuestion>(
      "/api/v1/generate",
      { concept, level },
      true
    ),

  uploadPracticeCSV: (
    file: File,
    includeHistory: boolean,
    historyCount: number
  ) => {
    const form = new FormData();
    form.append("file", file);
    form.append("include_history", String(includeHistory));
    form.append("history_count", String(historyCount));
    return fetch(`${BASE}/api/v1/practice/upload`, {
      method: "POST",
      headers: authHeaders(),
      body: form,
    }).then(async r => {
      if (!r.ok) { const e = await r.json().catch(() => ({ detail: "Upload failed" })); throw new Error(e.detail); }
      return r.json() as Promise<{ questions: import("../types").QuizQuestion[]; total: number }>;
    });
  },

  analyzePractice: (wrongItems: {
    question: string; options: string[]; correct_answer: string; user_answer: string;
  }[]) =>
    post<{ analyses: import("../types").AnalysisItem[] }>(
      "/api/v1/practice/analyze",
      { wrong_items: wrongItems },
      true
    ),

  recordBatch: (attempts: {
    question_text: string; options: string[]; correct_answer: string;
    user_answer: string; user_marked_correct: boolean;
  }[]) =>
    post("/api/v1/practice/record-batch", { attempts }, true),

  getMediaRecommendations: (concept: string) =>
    post<import("../types").MediaRecommendResponse>(
      "/api/v1/recommendations/media",
      { concept },
      true
    ),
};
