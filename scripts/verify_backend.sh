#!/usr/bin/env bash
# Verify that a deployed backend is reachable and authenticating exactly the way
# the iOS app will call it. Run this before typing credentials into the phone.
#
# Usage:
#   scripts/verify_backend.sh https://your-domain.example/api <token>
#   scripts/verify_backend.sh          # reads TRADE_DOMAIN/MOBILE_API_TOKEN from .env

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/.env"

die() {
    printf 'error: %s\n' "$1" >&2
    exit 1
}

env_value() {
    local key="$1"
    [ -f "${ENV_FILE}" ] || return 0
    sed -n "s/^${key}=//p" "${ENV_FILE}" | tail -n 1
}

command -v curl >/dev/null 2>&1 || die "curl is required"

api_url="${1:-}"
token="${2:-${MOBILE_API_TOKEN:-}}"

if [ -z "${api_url}" ]; then
    domain="${TRADE_DOMAIN:-$(env_value TRADE_DOMAIN)}"
    [ -n "${domain}" ] || die "no API URL given and TRADE_DOMAIN is unset"
    api_url="https://${domain}/api"
fi
if [ -z "${token}" ]; then
    token="$(env_value MOBILE_API_TOKEN)"
fi
[ -n "${token}" ] || die "no access token given and MOBILE_API_TOKEN is unset"

api_url="${api_url%/}"
failures=0

check() {
    local label="$1" url="$2" expected="$3"
    shift 3
    local code
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "$@" "${url}" || echo "000")"
    if [ "${code}" = "${expected}" ]; then
        printf 'ok    %-28s %s\n' "${label}" "${code}"
    else
        printf 'FAIL  %-28s got %s, expected %s\n' "${label}" "${code}" "${expected}"
        failures=$((failures + 1))
    fi
}

printf 'checking %s\n\n' "${api_url}"

check "public health"        "${api_url}/health"   200
check "overview (authorized)" "${api_url}/v1/overview" 200 -H "Authorization: Bearer ${token}"
check "overview (no token)"   "${api_url}/v1/overview" 401
check "overview (bad token)"  "${api_url}/v1/overview" 401 -H "Authorization: Bearer definitely-not-the-token"

printf '\n'
if [ "${failures}" -eq 0 ]; then
    printf 'Backend is reachable and rejecting unauthenticated callers.\n'
    printf 'Enter this URL and token on the iPhone: %s\n' "${api_url}"
    exit 0
fi

printf '%s check(s) failed.\n' "${failures}" >&2
printf 'If the health check itself failed, the container, DNS, or the HTTPS\n' >&2
printf 'certificate is the problem: docker compose -f docker-compose.production.yml logs\n' >&2
exit 1
