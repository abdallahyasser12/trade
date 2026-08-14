import * as BackgroundTask from 'expo-background-task';
import Constants from 'expo-constants';
import * as Device from 'expo-device';
import * as Notifications from 'expo-notifications';
import * as TaskManager from 'expo-task-manager';

import { loadSettings, MobileSettings } from './api';

const BACKGROUND_TASK_NAME = 'trade-background-health-refresh';
const MINIMUM_BACKGROUND_INTERVAL_MINUTES = 15;
const BOOTSTRAP_RETRY_MS = 10_000;
const BOOTSTRAP_RETRY_LIMIT = 30;
const REQUEST_TIMEOUT_MS = 12_000;

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowBanner: true,
    shouldShowList: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

TaskManager.defineTask(BACKGROUND_TASK_NAME, async () => {
  const settings = await loadSettings();
  if (!settings) {
    return;
  }
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    await fetch(`${settings.apiUrl}/health`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: controller.signal,
    });
  } catch {
    // iOS background execution is opportunistic. A failed wake must never
    // crash the app or be treated as proof that the trading backend stopped.
  } finally {
    clearTimeout(timer);
  }
});

async function registerBackgroundHealthRefresh(): Promise<void> {
  try {
    const registered = await TaskManager.isTaskRegisteredAsync(BACKGROUND_TASK_NAME);
    if (!registered) {
      await BackgroundTask.registerTaskAsync(BACKGROUND_TASK_NAME, {
        minimumInterval: MINIMUM_BACKGROUND_INTERVAL_MINUTES,
      });
    }
  } catch {
    // Background scheduling is a convenience only. The server-side trading
    // runtime is authoritative and continues without this task.
  }
}

function easProjectId(): string | null {
  const direct = Constants.easConfig?.projectId;
  if (direct) {
    return direct;
  }
  const extra = Constants.expoConfig?.extra as
    | { eas?: { projectId?: string } }
    | undefined;
  return extra?.eas?.projectId ?? null;
}

function notificationsAllowed(
  permission: Notifications.NotificationPermissionsStatus,
): boolean {
  return (
    permission.granted ||
    permission.ios?.status === Notifications.IosAuthorizationStatus.PROVISIONAL
  );
}

async function registerPushToken(settings: MobileSettings): Promise<void> {
  if (!Device.isDevice) {
    return;
  }

  let permission = await Notifications.getPermissionsAsync();
  if (!notificationsAllowed(permission)) {
    permission = await Notifications.requestPermissionsAsync();
  }
  if (!notificationsAllowed(permission)) {
    return;
  }

  const projectId = easProjectId();
  if (!projectId) {
    return;
  }

  const token = (await Notifications.getExpoPushTokenAsync({ projectId })).data;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    const response = await fetch(`${settings.apiUrl}/v1/devices/push`, {
      method: 'POST',
      headers: {
        Accept: 'application/json',
        Authorization: `Bearer ${settings.apiToken}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ token, platform: 'ios' }),
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`Push registration failed with HTTP ${response.status}.`);
    }
  } finally {
    clearTimeout(timer);
  }
}

async function bootstrapWhenConfigured(attempt = 0): Promise<void> {
  const settings = await loadSettings();
  if (!settings) {
    if (attempt < BOOTSTRAP_RETRY_LIMIT) {
      setTimeout(() => {
        void bootstrapWhenConfigured(attempt + 1);
      }, BOOTSTRAP_RETRY_MS);
    }
    return;
  }
  try {
    await registerPushToken(settings);
  } catch {
    // Push registration is retried next launch. Monitoring and trading remain
    // available even if notification permission/network setup fails.
  }
}

export async function bootstrapMobileRuntime(): Promise<void> {
  await registerBackgroundHealthRefresh();
  await bootstrapWhenConfigured();
}
