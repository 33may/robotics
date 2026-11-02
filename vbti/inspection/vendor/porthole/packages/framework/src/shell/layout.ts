/**
 * Layout persistence in localStorage.
 *
 * Every read is defensive. A saved layout is untrusted input: it was written by
 * an older version of the app, possibly with panels that no longer exist, and a
 * dashboard that refuses to open because of a stale layout is worse than one
 * that quietly starts fresh.
 */

const PREFIX = 'porthole.layout.';

export function loadDashboardLayout(key: string): Record<string, unknown> | null {
  if (typeof localStorage === 'undefined') return null;
  const raw = localStorage.getItem(PREFIX + key);
  if (!raw) return null;
  try {
    const parsed: unknown = JSON.parse(raw);
    return typeof parsed === 'object' && parsed !== null
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
}

export function saveDashboardLayout(key: string, layout: unknown): void {
  if (typeof localStorage === 'undefined') return;
  try {
    localStorage.setItem(PREFIX + key, JSON.stringify(layout));
  } catch {
    // Quota exceeded or storage disabled. A dashboard that cannot remember its
    // layout still works; one that throws on every drag does not.
  }
}

export function clearDashboardLayout(key: string): void {
  if (typeof localStorage === 'undefined') return;
  localStorage.removeItem(PREFIX + key);
}
