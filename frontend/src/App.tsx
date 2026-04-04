import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider } from "./contexts/AuthContext";
import LoginPage from "./pages/LoginPage";
import Layout from "./components/Layout";
import PracticePage from "./pages/PracticePage";

function PlaceholderPage({ name }: { name: string }) {
  return <div className="text-slate-400 text-center py-20">{name} — coming soon</div>;
}

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/practice" element={<Layout><PracticePage /></Layout>} />
          <Route path="/history" element={<Layout><PlaceholderPage name="History" /></Layout>} />
          <Route path="/stats" element={<Layout><PlaceholderPage name="Stats" /></Layout>} />
          <Route path="*" element={<Navigate to="/practice" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  );
}
