import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import LoginPage from "./pages/LoginPage";
import Layout from "./components/Layout";
import PracticePage from "./pages/PracticePage";
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";
import QuizPage from "./pages/QuizPage";

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/practice" element={<Layout><PracticePage /></Layout>} />
          <Route path="/quiz" element={<Layout><QuizPage /></Layout>} />
          <Route path="/history" element={<Layout><HistoryPage /></Layout>} />
          <Route path="/stats" element={<Layout><StatsPage /></Layout>} />
          <Route path="*" element={<Navigate to="/practice" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
