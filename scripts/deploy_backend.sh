#!/usr/bin/env bash
# Bring up the always-on trading backend and print the two values the iOS app
# needs on first launch: the Mobile API URL and the access token.
#
# Usage:
#   scripts/deploy_backend.sh your-domain.example
#   TRADE_DOMAIN=your-domain.example scripts/deploy_backend.sh
#
# The script is idempotent. An existing .env is never overwritten; only empty
# generated values are filled in.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"
COMPOSE_FILE="${REPO_ROOT}/docker-compose.production.yml"
HEALTH_TIMEOUT_SECONDS="${HEALTH_TIMEOUT_SECONDS:-300}"

die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

note() {
    printf '==> %s\n' "$1"
}

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose "$@"
    else
        docker-compose "$@"
    fi
}

random_token() {
    if command -v python3 >/dev/null 2>&1; then
        python3 -c 'import secrets; print(secrets.token_urlsafe(48))'
    elif command -v openssl >/dev/null 2>&1; then
        openssl rand -base64 48 | tr -d '\n=+/' | cut -c1-64
    else
        die "need python3 or openssl to generate MOBILE_API_TOKEN"
    fi
}

# Read a KEY=value from .env without sourcing it, so quoting or stray shell
# metacharacters in secrets cannot execute.
env_value() {
    local key="$1"
    [ -f "${ENV_FILE}" ] || return 0
    sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1
}

set_env_value() {
    local key="$1" value="$2" tmp
    tmp="$(mktemp)"
    if [ -f "${ENV_FILE}" ] && grep -q "^${key}=" "${ENV_FILE}"; then
        # awk avoids sed -i portability differences between GNU and BSD.
        awk -v key="${key}" -v value="${value}" \
            'index($0, key "=") == 1 { print key "=" value; next } { print }' \
            "${ENV_FILE}" > "${tmp}"
    else
        [ -f "${ENV_FILE}" ] && cat "${ENV_FILE}" > "${tmp}"
        printf '%s=%s\n' "${key}" "${value}" >> "${tmp}"
    fi
    mv "${tmp}" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
}

command -v docker >/dev/null 2>&1 || die "docker is not installed on this host"
docker compose version >/dev/null 2>&1 || command -v docker-compose >/dev/null 2>&1 \
    || die "docker compose is not available"
[ -f "${COMPOSE_FILE}" ] || die "missing ${COMPOSE_FILE}"

domain="${1:-${TRADE_DOMAIN:-}}"
if [ -z "${domain}" ]; then
    domain="$(env_value TRADE_DOMAIN)"
fi
if [ -z "${domain}" ]; then
    die "no domain given. Run: scripts/deploy_backend.sh your-domain.example
The domain must already resolve publicly to this host so Caddy can issue HTTPS."
fi

if [ ! -f "${ENV_FILE}" ]; then
    note "creating .env from .env.example"
    cp "${REPO_ROOT}/.env.example" "${ENV_FILE}"
    chmod 600 "${ENV_FILE}"
fi

set_env_value TRADE_DOMAIN "${domain}"

token="$(env_value MOBILE_API_TOKEN)"
if [ -z "${token}" ]; then
    note "generating MOBILE_API_TOKEN"
    token="$(random_token)"
    set_env_value MOBILE_API_TOKEN "${token}"
fi

if [ -z "$(env_value ALPACA_PAPER_API_KEY)" ]; then
    printf 'warning: ALPACA_PAPER_API_KEY is empty in .env.\n' >&2
    printf '         The stack starts, but the engine cannot reach the broker\n' >&2
    printf '         until paper credentials are filled in.\n' >&2
fi

note "building and starting the production stack"
compose -f "${COMPOSE_FILE}" up -d --build

service_health() {
    local service="$1" container
    container="$(compose -f "${COMPOSE_FILE}" ps -q "${service}" 2>/dev/null | head -n 1)"
    [ -n "${container}" ] || { printf 'not-created'; return 0; }
    docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}{{.State.Status}}{{end}}' \
        "${container}" 2>/dev/null || printf 'unknown'
}

note "waiting for the dashboard and mobile API to report healthy"
deadline=$(( $(date +%s) + HEALTH_TIMEOUT_SECONDS ))
while :; do
    dashboard_state="$(service_health dashboard)"
    api_state="$(service_health mobile-api)"

    if [ "${dashboard_state}" = "healthy" ] && [ "${api_state}" = "healthy" ]; then
        break
    fi
    if [ "$(date +%s)" -ge "${deadline}" ]; then
        printf 'error: services did not become healthy within %ss\n' "${HEALTH_TIMEOUT_SECONDS}" >&2
        printf '       dashboard=%s mobile-api=%s\n' "${dashboard_state:-unknown}" "${api_state:-unknown}" >&2
        printf '       inspect with: docker compose -f %s logs\n' "${COMPOSE_FILE}" >&2
        exit 1
    fi
    sleep 5
done

note "stack is healthy"

api_url="https://${domain}/api"
if command -v curl >/dev/null 2>&1; then
    if curl -fsS --max-time 15 "${api_url}/health" >/dev/null 2>&1; then
        note "public HTTPS endpoint answered: ${api_url}/health"
    else
        printf 'warning: %s/health is not reachable from this host yet.\n' "${api_url}" >&2
        printf '         Certificate issuance needs public DNS for %s plus\n' "${domain}" >&2
        printf '         inbound ports 80 and 443. Re-check in a minute.\n' >&2
    fi
fi

cat <<EOF

------------------------------------------------------------------
Enter these two values on the iPhone app's first launch screen.

  Mobile API URL : ${api_url}
  Access token   : ${token}

  Dashboard      : https://${domain}/

The access token is a secret. It grants read access to the trading
state and push registration. Do not screenshot or paste it publicly.
Rotate it by clearing MOBILE_API_TOKEN in .env and re-running this
script; every device then has to be reconnected.
------------------------------------------------------------------
EOF
