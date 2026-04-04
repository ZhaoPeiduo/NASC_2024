import { createContext, useContext, useState, useEffect, useRef, type ReactNode } from "react";
import { api } from "../api/client";

interface User { id: number; email: string }

interface AuthState {
  token: string | null;
  user: User | null;
  loading: boolean;
}

interface AuthContextValue extends AuthState {
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function safeGetToken(): string | null {
  try { return localStorage.getItem("token"); } catch { return null; }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<AuthState>({
    token: safeGetToken(),
    user: null,
    loading: true,
  });
  const tokenRef = useRef(state.token);
  tokenRef.current = state.token;

  useEffect(() => {
    const token = tokenRef.current;
    if (!token) { setState(s => ({ ...s, loading: false })); return; }

    const controller = new AbortController();
    api.me()
      .then((user: User) => {
        if (!controller.signal.aborted) setState(s => ({ ...s, user, loading: false }));
      })
      .catch(() => {
        if (!controller.signal.aborted) {
          try { localStorage.removeItem("token"); } catch { /* ignore */ }
          setState({ token: null, user: null, loading: false });
        }
      });
    return () => controller.abort();
  }, [state.token]);

  const login = async (email: string, password: string) => {
    setState(s => ({ ...s, loading: true }));
    const { access_token } = await api.login(email, password);
    try { localStorage.setItem("token", access_token); } catch { /* ignore */ }
    setState(s => ({ ...s, token: access_token }));
  };

  const register = async (email: string, password: string) => {
    setState(s => ({ ...s, loading: true }));
    const { access_token } = await api.register(email, password);
    try { localStorage.setItem("token", access_token); } catch { /* ignore */ }
    setState(s => ({ ...s, token: access_token }));
  };

  const logout = () => {
    try { localStorage.removeItem("token"); } catch { /* ignore */ }
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
