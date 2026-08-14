import { StatusBar } from 'expo-status-bar';
import {
  ActivityIndicator,
  Alert,
  Linking,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  useColorScheme,
  View,
} from 'react-native';
import { useCallback, useEffect, useMemo, useState } from 'react';

import {
  clearSettings,
  DashboardData,
  fetchDashboard,
  loadSettings,
  MobileSettings,
  saveSettings,
} from './src/api';

const DARK = {
  background: '#0A0A0A',
  card: '#171717',
  cardSoft: '#202020',
  text: '#F5F5F5',
  secondary: '#A3A3A3',
  border: '#2D2D2D',
  good: '#3DDC84',
  warn: '#FFB020',
  bad: '#FF5D5D',
  accent: '#5B8CFF',
};

const LIGHT = {
  background: '#F6F7F9',
  card: '#FFFFFF',
  cardSoft: '#F0F2F5',
  text: '#111318',
  secondary: '#667085',
  border: '#E2E5EA',
  good: '#14804A',
  warn: '#A15C00',
  bad: '#C93636',
  accent: '#285FE8',
};

type Palette = typeof DARK;

function asText(value: unknown, fallback = '—'): string {
  if (typeof value === 'string' && value.trim()) {
    return value;
  }
  if (typeof value === 'number' && Number.isFinite(value)) {
    return String(value);
  }
  return fallback;
}

function money(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return '—';
  }
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 2,
  }).format(value);
}

function timeLabel(value: string | null | undefined): string {
  if (!value) {
    return '—';
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString();
}

function statusTone(
  state: string | undefined,
  palette: Palette,
): string {
  const normalized = (state ?? '').toUpperCase();
  if (normalized === 'RUNNING') {
    return palette.good;
  }
  if (normalized === 'HALTED' || normalized === 'ERROR') {
    return palette.bad;
  }
  if (normalized === 'STARTING' || normalized === 'STOPPING') {
    return palette.warn;
  }
  return palette.secondary;
}

function Card({
  children,
  palette,
}: {
  children: React.ReactNode;
  palette: Palette;
}) {
  return (
    <View
      style={[
        styles.card,
        { backgroundColor: palette.card, borderColor: palette.border },
      ]}
    >
      {children}
    </View>
  );
}

function SectionTitle({
  title,
  subtitle,
  palette,
}: {
  title: string;
  subtitle?: string;
  palette: Palette;
}) {
  return (
    <View style={styles.sectionHeading}>
      <Text style={[styles.sectionTitle, { color: palette.text }]}>{title}</Text>
      {subtitle ? (
        <Text style={[styles.sectionSubtitle, { color: palette.secondary }]}>
          {subtitle}
        </Text>
      ) : null}
    </View>
  );
}

function EngineCard({
  title,
  status,
  signal,
  killSwitch,
  palette,
}: {
  title: string;
  status: Record<string, unknown> | null;
  signal: Record<string, unknown> | null;
  killSwitch: Record<string, unknown> | null;
  palette: Palette;
}) {
  const state = asText(status?.state, 'STOPPED');
  const mode = asText(status?.account_mode, title === 'Stocks' ? 'paper' : '—');
  const rawSymbols = status?.symbols;
  const symbol = asText(
    status?.symbol ?? (Array.isArray(rawSymbols) ? rawSymbols.join(', ') : undefined),
  );
  const timeframe = asText(status?.timeframe);
  const action = asText(signal?.action, 'HOLD');
  const reason = asText(signal?.reason, 'No recent signal');
  const killArmed =
    killSwitch !== null &&
    (killSwitch.armed === true || asText(killSwitch.reason, '') !== '');

  return (
    <Card palette={palette}>
      <View style={styles.engineHeader}>
        <View>
          <Text style={[styles.engineTitle, { color: palette.text }]}>{title}</Text>
          <Text style={[styles.engineMeta, { color: palette.secondary }]}>
            {mode.toUpperCase()} · {timeframe}
          </Text>
        </View>
        <View
          style={[
            styles.statusPill,
            { backgroundColor: statusTone(state, palette) + '20' },
          ]}
        >
          <View
            style={[
              styles.statusDot,
              { backgroundColor: statusTone(state, palette) },
            ]}
          />
          <Text
            style={[
              styles.statusText,
              { color: statusTone(state, palette) },
            ]}
          >
            {state}
          </Text>
        </View>
      </View>

      <Text style={[styles.symbol, { color: palette.text }]}>{symbol}</Text>

      <View style={[styles.signalBox, { backgroundColor: palette.cardSoft }]}>
        <Text style={[styles.signalAction, { color: palette.text }]}>
          {action}
        </Text>
        <Text
          numberOfLines={2}
          style={[styles.signalReason, { color: palette.secondary }]}
        >
          {reason}
        </Text>
      </View>

      {killArmed ? (
        <Text style={[styles.warningText, { color: palette.bad }]}>
          Safety lock is armed. Open the full dashboard to review it.
        </Text>
      ) : null}

      {status?.error ? (
        <Text style={[styles.warningText, { color: palette.bad }]}>
          {asText(status.error)}
        </Text>
      ) : null}
    </Card>
  );
}

function SetupScreen({
  initial,
  palette,
  onSaved,
}: {
  initial: MobileSettings | null;
  palette: Palette;
  onSaved: (settings: MobileSettings) => void;
}) {
  const [apiUrl, setApiUrl] = useState(initial?.apiUrl ?? '');
  const [apiToken, setApiToken] = useState(initial?.apiToken ?? '');
  const [saving, setSaving] = useState(false);

  const submit = async () => {
    setSaving(true);
    try {
      const settings = await saveSettings({ apiUrl, apiToken });
      onSaved(settings);
    } catch (error) {
      Alert.alert(
        'Could not save connection',
        error instanceof Error ? error.message : 'Unknown configuration error.',
      );
    } finally {
      setSaving(false);
    }
  };

  return (
    <ScrollView
      contentContainerStyle={[
        styles.setupContainer,
        { backgroundColor: palette.background },
      ]}
      keyboardShouldPersistTaps="handled"
    >
      <StatusBar style={palette === DARK ? 'light' : 'dark'} />
      <Text style={[styles.brand, { color: palette.text }]}>Trade</Text>
      <Text style={[styles.setupTitle, { color: palette.text }]}>
        Connect your mobile companion
      </Text>
      <Text style={[styles.setupBody, { color: palette.secondary }]}>
        The app reads the same persisted bot state as the Streamlit dashboard.
        Alpaca API keys never belong on this device.
      </Text>

      <Text style={[styles.label, { color: palette.text }]}>Mobile API URL</Text>
      <TextInput
        value={apiUrl}
        onChangeText={setApiUrl}
        autoCapitalize="none"
        autoCorrect={false}
        keyboardType="url"
        placeholder="https://trade-api.example.com"
        placeholderTextColor={palette.secondary}
        style={[
          styles.input,
          {
            backgroundColor: palette.card,
            borderColor: palette.border,
            color: palette.text,
          },
        ]}
      />

      <Text style={[styles.label, { color: palette.text }]}>Access token</Text>
      <TextInput
        value={apiToken}
        onChangeText={setApiToken}
        autoCapitalize="none"
        autoCorrect={false}
        secureTextEntry
        placeholder="32+ character server token"
        placeholderTextColor={palette.secondary}
        style={[
          styles.input,
          {
            backgroundColor: palette.card,
            borderColor: palette.border,
            color: palette.text,
          },
        ]}
      />
      <Text style={[styles.hint, { color: palette.secondary }]}>
        The token is stored in the device secure store, not in public Expo
        configuration.
      </Text>

      <Pressable
        disabled={saving}
        onPress={submit}
        style={({ pressed }) => [
          styles.primaryButton,
          {
            backgroundColor: palette.accent,
            opacity: pressed || saving ? 0.72 : 1,
          },
        ]}
      >
        {saving ? (
          <ActivityIndicator color="#FFFFFF" />
        ) : (
          <Text style={styles.primaryButtonText}>Connect</Text>
        )}
      </Pressable>
    </ScrollView>
  );
}

function DashboardScreen({
  settings,
  palette,
  onEditConnection,
}: {
  settings: MobileSettings;
  palette: Palette;
  onEditConnection: () => void;
}) {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    async (manual = false) => {
      manual ? setRefreshing(true) : setLoading(true);
      try {
        const result = await fetchDashboard(settings);
        setData(result);
        setError(null);
      } catch (requestError) {
        setError(
          requestError instanceof Error
            ? requestError.message
            : 'Could not load dashboard data.',
        );
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [settings],
  );

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => {
      void refresh(true);
    }, 30_000);
    return () => clearInterval(timer);
  }, [refresh]);

  const equitySummary = useMemo(() => {
    const points = data?.equity ?? [];
    if (points.length === 0) {
      return { current: null, change: null };
    }
    const first = points[0]?.equity;
    const last = points[points.length - 1]?.equity;
    if (
      first === undefined ||
      last === undefined ||
      !Number.isFinite(first) ||
      !Number.isFinite(last) ||
      first <= 0
    ) {
      return { current: last ?? null, change: null };
    }
    return {
      current: last,
      change: ((last / first) - 1) * 100,
    };
  }, [data]);

  const cryptoSignal = useMemo(() => {
    const signals = data?.overview.crypto.last_signals;
    if (!signals) {
      return null;
    }
    const entries = Object.values(signals);
    const actionable = entries.find(
      (item) => asText(item.action, 'HOLD').toUpperCase() !== 'HOLD',
    );
    return actionable ?? entries[0] ?? null;
  }, [data]);

  const openFullDashboard = async () => {
    const url = data?.overview.dashboard_url;
    if (!url) {
      Alert.alert(
        'Full dashboard URL not configured',
        'Set MOBILE_DASHBOARD_URL on the server to expose the existing Streamlit controls.',
      );
      return;
    }
    const supported = await Linking.canOpenURL(url);
    if (!supported) {
      Alert.alert('Cannot open dashboard', 'This device cannot open the configured URL.');
      return;
    }
    await Linking.openURL(url);
  };

  return (
    <ScrollView
      style={{ backgroundColor: palette.background }}
      contentContainerStyle={styles.dashboardContainer}
      refreshControl={
        <RefreshControl
          refreshing={refreshing}
          onRefresh={() => void refresh(true)}
          tintColor={palette.accent}
        />
      }
    >
      <StatusBar style={palette === DARK ? 'light' : 'dark'} />
      <View style={styles.topBar}>
        <View>
          <Text style={[styles.brand, { color: palette.text }]}>Trade</Text>
          <Text style={[styles.topSubtitle, { color: palette.secondary }]}>
            Mobile companion
          </Text>
        </View>
        <Pressable
          onPress={onEditConnection}
          style={[styles.secondaryButton, { borderColor: palette.border }]}
        >
          <Text style={[styles.secondaryButtonText, { color: palette.text }]}>
            Settings
          </Text>
        </Pressable>
      </View>

      {loading && !data ? (
        <View style={styles.loadingBox}>
          <ActivityIndicator size="large" color={palette.accent} />
          <Text style={[styles.hint, { color: palette.secondary }]}>
            Reading bot state…
          </Text>
        </View>
      ) : null}

      {error ? (
        <Card palette={palette}>
          <Text style={[styles.errorTitle, { color: palette.bad }]}>
            Connection issue
          </Text>
          <Text style={[styles.errorBody, { color: palette.secondary }]}>
            {error}
          </Text>
          <Pressable onPress={() => void refresh()} style={styles.inlineAction}>
            <Text style={{ color: palette.accent, fontWeight: '700' }}>Retry</Text>
          </Pressable>
        </Card>
      ) : null}

      {data ? (
        <>
          <View style={styles.summaryRow}>
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryLabel, { color: palette.secondary }]}>
                Latest equity
              </Text>
              <Text style={[styles.summaryValue, { color: palette.text }]}>
                {money(equitySummary.current)}
              </Text>
            </View>
            <View style={styles.summaryItem}>
              <Text style={[styles.summaryLabel, { color: palette.secondary }]}>
                Window change
              </Text>
              <Text
                style={[
                  styles.summaryValue,
                  {
                    color:
                      equitySummary.change === null
                        ? palette.secondary
                        : equitySummary.change >= 0
                          ? palette.good
                          : palette.bad,
                  },
                ]}
              >
                {equitySummary.change === null
                  ? '—'
                  : `${equitySummary.change >= 0 ? '+' : ''}${equitySummary.change.toFixed(2)}%`}
              </Text>
            </View>
          </View>

          <SectionTitle
            title="Systems"
            subtitle={`Updated ${timeLabel(data.overview.generated_at)}`}
            palette={palette}
          />
          <EngineCard
            title="Stocks"
            status={data.overview.stock.status as Record<string, unknown> | null}
            signal={data.overview.stock.last_signal}
            killSwitch={data.overview.stock.kill_switch}
            palette={palette}
          />
          <EngineCard
            title="Crypto"
            status={data.overview.crypto.status as Record<string, unknown> | null}
            signal={cryptoSignal}
            killSwitch={data.overview.crypto.kill_switch}
            palette={palette}
          />

          <Pressable
            onPress={() => void openFullDashboard()}
            style={({ pressed }) => [
              styles.primaryButton,
              {
                backgroundColor: palette.accent,
                opacity: pressed ? 0.75 : 1,
              },
            ]}
          >
            <Text style={styles.primaryButtonText}>Open full trading dashboard</Text>
          </Pressable>
          <Text style={[styles.safetyCaption, { color: palette.secondary }]}>
            Order entry and live-account unlock stay in the existing dashboard so
            the mobile companion cannot bypass its safety gates.
          </Text>

          <SectionTitle title="Recent trades" palette={palette} />
          <Card palette={palette}>
            {data.trades.length === 0 ? (
              <Text style={[styles.emptyText, { color: palette.secondary }]}>
                No persisted trades yet.
              </Text>
            ) : (
              data.trades.slice(0, 8).map((trade, index) => (
                <View
                  key={`${trade.id ?? index}-${trade.timestamp ?? index}`}
                  style={[
                    styles.row,
                    index > 0 && { borderTopColor: palette.border, borderTopWidth: 1 },
                  ]}
                >
                  <View style={styles.rowMain}>
                    <Text style={[styles.rowTitle, { color: palette.text }]}>
                      {asText(trade.action)} {asText(trade.symbol)}
                    </Text>
                    <Text
                      numberOfLines={1}
                      style={[styles.rowSubtitle, { color: palette.secondary }]}
                    >
                      {asText(trade.reason, 'Execution')} · {timeLabel(trade.timestamp)}
                    </Text>
                  </View>
                  <View style={styles.rowValue}>
                    <Text style={[styles.rowTitle, { color: palette.text }]}>
                      {trade.qty ?? '—'}
                    </Text>
                    <Text
                      style={[
                        styles.rowSubtitle,
                        {
                          color:
                            (trade.realized_pnl ?? 0) >= 0
                              ? palette.good
                              : palette.bad,
                        },
                      ]}
                    >
                      {money(trade.realized_pnl)}
                    </Text>
                  </View>
                </View>
              ))
            )}
          </Card>

          <SectionTitle title="Engine log" palette={palette} />
          <Card palette={palette}>
            {data.logs.length === 0 ? (
              <Text style={[styles.emptyText, { color: palette.secondary }]}>
                No persisted log events yet.
              </Text>
            ) : (
              data.logs.slice(0, 8).map((entry, index) => (
                <View
                  key={`${entry.id ?? index}-${entry.timestamp ?? index}`}
                  style={[
                    styles.logRow,
                    index > 0 && { borderTopColor: palette.border, borderTopWidth: 1 },
                  ]}
                >
                  <Text
                    style={[
                      styles.logSeverity,
                      {
                        color:
                          asText(entry.severity).toUpperCase() === 'ERROR' ||
                          asText(entry.severity).toUpperCase() === 'CRITICAL'
                            ? palette.bad
                            : asText(entry.severity).toUpperCase() === 'WARNING'
                              ? palette.warn
                              : palette.secondary,
                      },
                    ]}
                  >
                    {asText(entry.severity, 'INFO')}
                  </Text>
                  <View style={styles.logText}>
                    <Text
                      numberOfLines={2}
                      style={[styles.rowTitle, { color: palette.text }]}
                    >
                      {asText(entry.message)}
                    </Text>
                    <Text style={[styles.rowSubtitle, { color: palette.secondary }]}>
                      {timeLabel(entry.timestamp)}
                    </Text>
                  </View>
                </View>
              ))
            )}
          </Card>

          <Text style={[styles.footer, { color: palette.secondary }]}>
            Monitoring only · Alpaca credentials stay on the server
          </Text>
        </>
      ) : null}
    </ScrollView>
  );
}

export default function App() {
  const scheme = useColorScheme();
  const palette = scheme === 'light' ? LIGHT : DARK;
  const [settings, setSettings] = useState<MobileSettings | null>(null);
  const [checkingSettings, setCheckingSettings] = useState(true);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    loadSettings()
      .then((saved) => setSettings(saved))
      .catch(() => setSettings(null))
      .finally(() => setCheckingSettings(false));
  }, []);

  if (checkingSettings) {
    return (
      <View style={[styles.centered, { backgroundColor: palette.background }]}>
        <StatusBar style={palette === DARK ? 'light' : 'dark'} />
        <ActivityIndicator size="large" color={palette.accent} />
      </View>
    );
  }

  if (!settings || editing) {
    return (
      <SetupScreen
        initial={settings}
        palette={palette}
        onSaved={(next) => {
          setSettings(next);
          setEditing(false);
        }}
      />
    );
  }

  return (
    <DashboardScreen
      settings={settings}
      palette={palette}
      onEditConnection={() => {
        Alert.alert('Connection settings', 'Choose what you want to do.', [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Edit',
            onPress: () => setEditing(true),
          },
          {
            text: 'Forget connection',
            style: 'destructive',
            onPress: () => {
              void clearSettings().then(() => setSettings(null));
            },
          },
        ]);
      }}
    />
  );
}

const styles = StyleSheet.create({
  centered: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  setupContainer: {
    flexGrow: 1,
    paddingHorizontal: 24,
    paddingTop: 72,
    paddingBottom: 40,
  },
  dashboardContainer: {
    paddingHorizontal: 18,
    paddingTop: 58,
    paddingBottom: 48,
  },
  brand: {
    fontSize: 32,
    fontWeight: '800',
    letterSpacing: -1.2,
  },
  setupTitle: {
    marginTop: 34,
    fontSize: 28,
    fontWeight: '800',
    lineHeight: 34,
  },
  setupBody: {
    marginTop: 12,
    marginBottom: 30,
    fontSize: 16,
    lineHeight: 24,
  },
  label: {
    marginTop: 16,
    marginBottom: 8,
    fontSize: 14,
    fontWeight: '700',
  },
  input: {
    minHeight: 52,
    borderWidth: 1,
    borderRadius: 14,
    paddingHorizontal: 14,
    fontSize: 16,
  },
  hint: {
    marginTop: 10,
    fontSize: 13,
    lineHeight: 19,
  },
  primaryButton: {
    minHeight: 52,
    marginTop: 22,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 18,
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '800',
  },
  topBar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  topSubtitle: {
    marginTop: 2,
    fontSize: 13,
  },
  secondaryButton: {
    borderWidth: 1,
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  secondaryButtonText: {
    fontSize: 14,
    fontWeight: '700',
  },
  loadingBox: {
    minHeight: 180,
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    borderWidth: 1,
    borderRadius: 18,
    padding: 16,
    marginBottom: 12,
  },
  errorTitle: {
    fontSize: 16,
    fontWeight: '800',
  },
  errorBody: {
    marginTop: 6,
    fontSize: 14,
    lineHeight: 20,
  },
  inlineAction: {
    paddingTop: 14,
    paddingBottom: 2,
  },
  summaryRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 24,
  },
  summaryItem: {
    flex: 1,
  },
  summaryLabel: {
    fontSize: 12,
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.4,
  },
  summaryValue: {
    marginTop: 5,
    fontSize: 21,
    fontWeight: '800',
  },
  sectionHeading: {
    marginTop: 7,
    marginBottom: 10,
  },
  sectionTitle: {
    fontSize: 19,
    fontWeight: '800',
  },
  sectionSubtitle: {
    marginTop: 3,
    fontSize: 12,
  },
  engineHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
  },
  engineTitle: {
    fontSize: 17,
    fontWeight: '800',
  },
  engineMeta: {
    marginTop: 3,
    fontSize: 12,
    fontWeight: '600',
  },
  statusPill: {
    minHeight: 30,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 7,
    borderRadius: 999,
    paddingHorizontal: 10,
  },
  statusDot: {
    width: 7,
    height: 7,
    borderRadius: 99,
  },
  statusText: {
    fontSize: 11,
    fontWeight: '800',
  },
  symbol: {
    marginTop: 20,
    fontSize: 23,
    fontWeight: '800',
  },
  signalBox: {
    marginTop: 13,
    borderRadius: 13,
    padding: 13,
  },
  signalAction: {
    fontSize: 14,
    fontWeight: '800',
  },
  signalReason: {
    marginTop: 4,
    fontSize: 13,
    lineHeight: 18,
  },
  warningText: {
    marginTop: 11,
    fontSize: 12,
    lineHeight: 17,
    fontWeight: '600',
  },
  safetyCaption: {
    marginTop: 8,
    marginBottom: 24,
    fontSize: 12,
    lineHeight: 18,
    textAlign: 'center',
  },
  row: {
    minHeight: 58,
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
  },
  rowMain: {
    flex: 1,
    paddingRight: 10,
  },
  rowValue: {
    alignItems: 'flex-end',
  },
  rowTitle: {
    fontSize: 13,
    fontWeight: '700',
  },
  rowSubtitle: {
    marginTop: 3,
    fontSize: 11,
  },
  emptyText: {
    fontSize: 13,
    paddingVertical: 6,
  },
  logRow: {
    minHeight: 60,
    flexDirection: 'row',
    alignItems: 'flex-start',
    paddingVertical: 10,
  },
  logSeverity: {
    width: 72,
    fontSize: 10,
    fontWeight: '900',
    paddingTop: 2,
  },
  logText: {
    flex: 1,
  },
  footer: {
    marginTop: 18,
    textAlign: 'center',
    fontSize: 11,
  },
});
