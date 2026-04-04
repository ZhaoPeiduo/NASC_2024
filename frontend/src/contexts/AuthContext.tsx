import { createContext, useContext, useState, useEffect, type ReactNode } from "react";
import { api } from "../api/client";

interface AuthState {
  token: string | null;
  user: { id: number; email: string } | null;
  loading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    token: localStorage.getItem("token"),
    user: null,
    loading: true,
  });

  useEffect(() => {
    if (!state.token) { setState(s => ({ ...s, loading: false })); return; }
    api.me()
      .then(user => setState(s => ({ ...s, user, loading: false })))
      .catch(() => { localStorage.removeItem("token"); setState({ token: null, user: null, loading: false }); });
  }, [state.token]);

  const login = async (email: string, password: string) => {
    const { access_token } = await api.login(email, password);
    localStorage.setItem("token", access_token);
    setState(s => ({ ...s, token: access_token }));
  };

  const register = async (email: string, password: string) => {
    const { access_token } = await api.register(email, password);
    localStorage.setItem("token", access_token);
    setState(s => ({ ...s, token: access_token }));
  };

  const logout = () => {
    localStorage.removeItem("token");
    setState({ token: null, user: null, loading: false });
  };

  return (
    <AuthContext.Provider value={{ ...state, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuthContext() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuthContext must be inside AuthProvider");
  return ctx;
}
