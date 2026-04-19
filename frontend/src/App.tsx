import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import LoginPage from "./pages/LoginPage";
import Layout from "./components/Layout";
import AskPage from "./pages/AskPage";
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";
import QuizPage from "./pages/QuizPage";
import DiscoverPage from "./pages/DiscoverPage";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/ask" element={<Layout><AskPage /></Layout>} />
          <Route path="/practice" element={<Navigate to="/ask" replace />} />
          <Route path="/quiz" element={<Layout><QuizPage /></Layout>} />
          <Route path="/history" element={<Layout><HistoryPage /></Layout>} />
          <Route path="/stats" element={<Layout><StatsPage /></Layout>} />
          <Route path="/discover" element={<Layout><DiscoverPage /></Layout>} />
          <Route path="*" element={<Navigate to="/ask" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
