import * as SecureStore from 'expo-secure-store';

const API_URL_KEY = 'trade.mobile.apiUrl';
const API_TOKEN_KEY = 'trade.mobile.apiToken';
const REQUEST_TIMEOUT_MS = 12_000;

export type MobileSettings = {
  apiUrl: string;
  apiToken: string;
};

export type EngineStatus = {
  state?: string;
  session_id?: string | null;
  account_mode?: string;
  symbol?: string | null;
  symbols?: string[];
  timeframe?: string | null;
  started_at?: string | null;
  last_heartbeat?: string | null;
  last_bar_time?: string | null;
  halt_reason?: string | null;
  error?: string | null;
  warnings?: string[];
};

export type Overview = {
  generated_at: string;
  stock: {
    status: EngineStatus | null;
    last_signal: Record<string, unknown> | null;
    kill_switch: Record<string, unknown> | null;
  };
  crypto: {
    status: EngineStatus | null;
    last_signals: Record<string, Record<string, unknown>> | null;
    kill_switch: Record<string, unknown> | null;
  };
  latest_session: {
    session_id?: string;
    symbol?: string;
    timeframe?: string;
    started_at?: string;
    stopped_at?: string | null;
    status?: string;
    stop_reason?: string | null;
    halted?: boolean;
    halt_reason?: string | null;
    starting_equity?: number | null;
  } | null;
  dashboard_url: string | null;
  capabilities: {
    monitoring: boolean;
    recent_activity: boolean;
    equity_history: boolean;
    trading_commands: boolean;
  };
};

export type TradeRow = {
  id?: number;
  session_id?: string | null;
  symbol?: string;
  action?: string;
  qty?: number;
  fill_price?: number | null;
  realized_pnl?: number | null;
  reason?: string | null;
  timestamp?: string;
};

export type LogRow = {
  id?: number;
  session_id?: string | null;
  timestamp?: string;
  severity?: string;
  message?: string;
};

export type EquityPoint = {
  timestamp: string;
  equity: number;
  buying_power?: number | null;
};

export type DashboardData = {
  overview: Overview;
  trades: TradeRow[];
  logs: LogRow[];
  equity: EquityPoint[];
};

function isPrivateIpv4(hostname: string): boolean {
  const pieces = hostname.split('.').map((part) => Number(part));
  if (
    pieces.length !== 4 ||
    pieces.some((part) => !Number.isInteger(part) || part < 0 || part > 255)
  ) {
    return false;
  }
  const first = pieces[0] ?? -1;
  const second = pieces[1] ?? -1;
  return (
    first === 10 ||
    first === 127 ||
    (first === 192 && second === 168) ||
    (first === 172 && second >= 16 && second <= 31)
  );
}

export function normalizeApiUrl(value: string): string {
  const trimmed = value.trim().replace(/\/+$/, '');
  if (!trimmed) {
    throw new Error('API URL is required.');
  }

  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    throw new Error('Enter a valid API URL, including https://.');
  }

  const localDevelopment =
    parsed.hostname === 'localhost' || isPrivateIpv4(parsed.hostname);
  if (parsed.protocol !== 'https:' && !(parsed.protocol === 'http:' && localDevelopment)) {
    throw new Error('Use HTTPS outside localhost or a private development network.');
  }
  return parsed.toString().replace(/\/+$/, '');
}

export function validateToken(value: string): string {
  const token = value.trim();
  if (token.length < 32) {
    throw new Error('Access token must contain at least 32 characters.');
  }
  return token;
}

export async function loadSettings(): Promise<MobileSettings | null> {
  const [apiUrl, apiToken] = await Promise.all([
    SecureStore.getItemAsync(API_URL_KEY),
    SecureStore.getItemAsync(API_TOKEN_KEY),
  ]);
  if (!apiUrl || !apiToken) {
    return null;
  }
  return { apiUrl, apiToken };
}

export async function saveSettings(settings: MobileSettings): Promise<MobileSettings> {
  const normalized: MobileSettings = {
    apiUrl: normalizeApiUrl(settings.apiUrl),
    apiToken: validateToken(settings.apiToken),
  };
  await Promise.all([
    SecureStore.setItemAsync(API_URL_KEY, normalized.apiUrl),
    SecureStore.setItemAsync(API_TOKEN_KEY, normalized.apiToken),
  ]);
  return normalized;
}

export async function clearSettings(): Promise<void> {
  await Promise.all([
    SecureStore.deleteItemAsync(API_URL_KEY),
    SecureStore.deleteItemAsync(API_TOKEN_KEY),
  ]);
}

async function requestJson<T>(
  settings: MobileSettings,
  path: string,
): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${settings.apiUrl}${path}`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
        Authorization: `Bearer ${settings.apiToken}`,
      },
      signal: controller.signal,
    });
    if (response.status === 401) {
      throw new Error('The saved mobile access token was rejected.');
    }
    if (response.status === 503) {
      throw new Error('The mobile API is not configured on the server.');
    }
    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}.`);
    }
    return (await response.json()) as T;
  } catch (error) {
    if (error instanceof Error && error.name === 'AbortError') {
      throw new Error('The mobile API did not respond before the timeout.');
    }
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

export async function checkHealth(apiUrl: string): Promise<boolean> {
  const normalized = normalizeApiUrl(apiUrl);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${normalized}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
    if (!response.ok) {
      return false;
    }
    const body = (await response.json()) as { ok?: boolean };
    return body.ok === true;
  } finally {
    clearTimeout(timer);
  }
}

export async function fetchDashboard(
  settings: MobileSettings,
): Promise<DashboardData> {
  const [overview, trades, logs, equity] = await Promise.all([
    requestJson<Overview>(settings, '/v1/overview'),
    requestJson<{ items: TradeRow[] }>(settings, '/v1/trades?limit=20'),
    requestJson<{ items: LogRow[] }>(settings, '/v1/logs?limit=30'),
    requestJson<{ items: EquityPoint[] }>(settings, '/v1/equity?limit=120'),
  ]);
  return {
    overview,
    trades: trades.items,
    logs: logs.items,
    equity: equity.items,
  };
}
