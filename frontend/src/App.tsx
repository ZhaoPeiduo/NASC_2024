import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import PracticePage from "./pages/PracticePage";
import HistoryPage from "./pages/HistoryPage";
import StatsPage from "./pages/StatsPage";
import LoginPage from "./pages/LoginPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/practice" element={<PracticePage />} />
        <Route path="/history" element={<HistoryPage />} />
        <Route path="/stats" element={<StatsPage />} />
        <Route path="*" element={<Navigate to="/practice" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
