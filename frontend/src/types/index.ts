export type Phase = "idle" | "answering" | "explaining" | "done";

export interface SolveResult {
  answer: string;
  explanation: string;
  wrong_options: Record<string, string>;
  concepts: string[];
}

export interface AttemptResponse {
  id: number;
  question_text: string;
  correct_answer: string;
  llm_answer: string;
  user_marked_correct: boolean;
  concepts: string[];
  created_at: string;
  options: string[];
  explanation: string;
}

export interface StatsResponse {
  total_attempts: number;
  correct_rate: number;
  weak_concepts: string[];
  study_days: number;
}

export interface VideoRecommendation {
  title: string;
  video_id: string;
  thumbnail_url: string;
  channel_title: string;
}

export interface GeneratedQuestion {
  question: string;
  options: string[];
  correct_answer: string;
  explanation: string;
  concepts: string[];
}

export interface QuizQuestion {
  question: string;
  options: string[];
  correct_answer: string;
  from_history?: boolean;
}

export interface AnalysisItem {
  question: string;
  correct_answer: string;
  user_answer: string;
  explanation: string;
}

export interface MediaRecommendResponse {
  songs: { title: string; artist: string }[];
  anime: { title: string; scene: string }[];
  articles: { title: string; keywords: string }[];
}

export interface VideoTutorialSet {
  question_snippet: string;
  concepts: string[];
  search_query: string;
  videos: VideoRecommendation[];
}

export interface WrongAnswerRecsResponse {
  recommendations: VideoTutorialSet[];
}

export interface TikTokVideoMeta {
  video_id: string;
  title: string;
  author: string;
}
