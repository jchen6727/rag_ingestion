#!/usr/bin/env bash
#
# Preflight check for scripts/setup_vertex_search.py.
#
# Verifies, in order:
#   1. gcloud CLI is installed and an account is logged in
#   2. GCP_PROJECT_ID (from .env or the environment) exists and is the active project
#   3. Billing is enabled on the project
#   4. Required APIs (Discovery Engine, Cloud Storage, Vertex AI) are enabled
#   5. The active identity holds the IAM permissions setup_vertex_search.py needs
#
# On any failure, prints the exact gcloud command(s) to fix the issue and
# continues checking the rest (so a single run surfaces every problem, not
# just the first one). Exits non-zero if any check failed.
#
# Usage:
#   scripts/preflight_check.sh
#
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

FAILURES=0
CHECK_MARK="✓"
CROSS_MARK="✗"

pass() { echo "${CHECK_MARK} $1"; }
warn() { echo "! $1"; }   # advisory only — does not count as a failure
fail() {
    echo "${CROSS_MARK} $1"
    FAILURES=$((FAILURES + 1))
}
fix() {
    echo "  Fix:"
    while IFS= read -r line; do
        echo "    ${line}"
    done <<< "$1"
    echo
}

echo "=== RAG Guidance — Vertex AI Search preflight check ==="
echo

# ---------------------------------------------------------------------------
# 0. Load .env (if present) without clobbering already-exported vars
# ---------------------------------------------------------------------------
if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${REPO_ROOT}/.env"
    set +a
fi

GCP_PROJECT_ID="${GCP_PROJECT_ID:-}"
GCP_LOCATION="${GCP_LOCATION:-global}"

# ---------------------------------------------------------------------------
# 1. gcloud CLI present and authenticated
# ---------------------------------------------------------------------------
if ! command -v gcloud >/dev/null 2>&1; then
    fail "gcloud CLI not found on PATH"
    fix "Install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    echo
    echo "Cannot continue without gcloud. Aborting remaining checks."
    exit 1
fi
pass "gcloud CLI found ($(gcloud --version | head -n1))"

AUTH_LIST_OUTPUT="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>&1)"
AUTH_LIST_EXIT=$?
if [[ ${AUTH_LIST_EXIT} -ne 0 ]]; then
    fail "Unable to query gcloud auth state: ${AUTH_LIST_OUTPUT}"
    fix "gcloud auth login
gcloud auth application-default login"
    ACTIVE_ACCOUNT=""
else
    ACTIVE_ACCOUNT="${AUTH_LIST_OUTPUT}"
    if [[ -z "${ACTIVE_ACCOUNT}" ]]; then
        fail "No active gcloud account"
        fix "gcloud auth login
gcloud auth application-default login"
    else
        pass "Active gcloud account: ${ACTIVE_ACCOUNT}"
    fi
fi

# Application Default Credentials are what the Python client libraries in
# setup_vertex_search.py actually use — a plain `gcloud auth login` alone
# is not sufficient.
if ! gcloud auth application-default print-access-token >/dev/null 2>&1; then
    fail "Application Default Credentials (ADC) not set up"
    fix "gcloud auth application-default login"
else
    pass "Application Default Credentials available"
fi

# Standing policy (devlog.md#DONE(sh-to-rest)): programmatic checks below use
# curl/REST with this ADC token; only interactive auth (`gcloud auth ...`) and
# one-shot enablement (`gcloud services enable`) remain on gcloud. Acquired once
# here and reused by the project / billing / services / IAM checks.
ACCESS_TOKEN="$(gcloud auth application-default print-access-token 2>/dev/null)"

# ---------------------------------------------------------------------------
# 2. Project exists and matches GCP_PROJECT_ID / active gcloud config
# ---------------------------------------------------------------------------
if [[ -z "${GCP_PROJECT_ID}" ]]; then
    fail "GCP_PROJECT_ID is not set (checked .env and environment)"
    fix "cp .env.example .env   # then set GCP_PROJECT_ID=<your-project>"
    echo
    echo "Cannot continue project/billing/IAM checks without a project ID. Aborting."
    exit 1
fi

# REST: Cloud Resource Manager projects.get
PROJECT_HTTP_STATUS="$(curl -s -o /dev/null -w '%{http_code}' \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    "https://cloudresourcemanager.googleapis.com/v1/projects/${GCP_PROJECT_ID}" 2>/dev/null)"
if [[ "${PROJECT_HTTP_STATUS}" != "200" ]]; then
    fail "Project '${GCP_PROJECT_ID}' does not exist or is not accessible to ${ACTIVE_ACCOUNT:-<no account>} (HTTP ${PROJECT_HTTP_STATUS})"
    fix "gcloud projects list --filter=\"projectId:${GCP_PROJECT_ID}\"
# If it doesn't exist:
gcloud projects create ${GCP_PROJECT_ID}
# If it exists but isn't accessible, ask the project owner to run:
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \\
    --member=\"user:${ACTIVE_ACCOUNT:-YOUR_EMAIL}\" --role=\"roles/editor\""
else
    pass "Project '${GCP_PROJECT_ID}' exists and is accessible"
fi

CONFIG_GET_OUTPUT="$(gcloud config get-value project 2>&1)"
CONFIG_GET_EXIT=$?
if [[ ${CONFIG_GET_EXIT} -ne 0 ]]; then
    fail "Unable to read active gcloud config project: ${CONFIG_GET_OUTPUT}"
    fix "gcloud config set project ${GCP_PROJECT_ID}"
else
    ACTIVE_CONFIG_PROJECT="${CONFIG_GET_OUTPUT}"
    if [[ "${ACTIVE_CONFIG_PROJECT}" != "${GCP_PROJECT_ID}" ]]; then
        fail "Active gcloud config project ('${ACTIVE_CONFIG_PROJECT}') does not match GCP_PROJECT_ID ('${GCP_PROJECT_ID}')"
        fix "gcloud config set project ${GCP_PROJECT_ID}"
    else
        pass "Active gcloud config project matches GCP_PROJECT_ID"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Billing enabled
# ---------------------------------------------------------------------------
# REST: Cloud Billing projects.getBillingInfo
BILLING_FILE="$(mktemp)"
BILLING_HTTP_STATUS="$(curl -s -o "${BILLING_FILE}" -w '%{http_code}' \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    "https://cloudbilling.googleapis.com/v1/projects/${GCP_PROJECT_ID}/billingInfo" 2>/dev/null)"
if [[ "${BILLING_HTTP_STATUS}" != "200" ]]; then
    fail "Unable to check billing status for '${GCP_PROJECT_ID}' (HTTP ${BILLING_HTTP_STATUS}; caller likely lacks billing.resourceAssociations.get): $(cat "${BILLING_FILE}")"
    fix "gcloud billing accounts list
gcloud billing projects link ${GCP_PROJECT_ID} --billing-account=<BILLING_ACCOUNT_ID>"
else
    BILLING_ENABLED="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("billingEnabled", False))' < "${BILLING_FILE}" 2>/dev/null)"
    if [[ "${BILLING_ENABLED}" != "True" ]]; then
        fail "Billing is not enabled on project '${GCP_PROJECT_ID}'"
        fix "gcloud billing accounts list
gcloud billing projects link ${GCP_PROJECT_ID} --billing-account=<BILLING_ACCOUNT_ID>"
    else
        pass "Billing is enabled on '${GCP_PROJECT_ID}'"
    fi
fi
rm -f "${BILLING_FILE}"

# ---------------------------------------------------------------------------
# 4. Required APIs enabled
# ---------------------------------------------------------------------------
REQUIRED_APIS=(
    "discoveryengine.googleapis.com"   # Vertex AI Search DataStore + import
    "storage.googleapis.com"           # GCS staging of PDFs + chunk JSONL
    "aiplatform.googleapis.com"        # Vertex AI (future google-genai/Vertex path)
    "generativelanguage.googleapis.com" # Gemini Developer API (current metadata_gen path)
)
# REST: Service Usage services.list (state:ENABLED), paginated. `gcloud services
# enable` stays in the fix text per the .sh→REST policy (enablement, not a check).
ENABLED_APIS="$(ACCESS_TOKEN="${ACCESS_TOKEN}" GCP_PROJECT_ID="${GCP_PROJECT_ID}" python3 - <<'PY' 2>/dev/null
import json, os, sys, urllib.request, urllib.error
token = os.environ.get("ACCESS_TOKEN", "")
proj = os.environ.get("GCP_PROJECT_ID", "")
if not token:
    sys.exit(2)
base = (f"https://serviceusage.googleapis.com/v1/projects/{proj}/services"
        "?filter=state:ENABLED&pageSize=200")
names, page = [], ""
try:
    while True:
        url = base + (f"&pageToken={page}" if page else "")
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.load(resp)
        for svc in data.get("services", []):
            name = (svc.get("config", {}) or {}).get("name") or svc.get("name", "").split("/")[-1]
            if name:
                names.append(name)
        page = data.get("nextPageToken", "")
        if not page:
            break
except urllib.error.HTTPError as exc:
    sys.exit(3)
except Exception:
    sys.exit(4)
print("\n".join(names))
PY
)"
SERVICES_LIST_EXIT=$?

if [[ ${SERVICES_LIST_EXIT} -ne 0 ]]; then
    fail "Unable to list enabled APIs for '${GCP_PROJECT_ID}' via Service Usage REST (exit ${SERVICES_LIST_EXIT}; e.g. missing serviceusage.services.list, or no ADC token)"
    fix "gcloud services list --project=${GCP_PROJECT_ID}
# If this fails on permissions, ask an admin to grant roles/serviceusage.serviceUsageViewer,
# then re-run this preflight check."
else
    for api in "${REQUIRED_APIS[@]}"; do
        if grep -qx "${api}" <<< "${ENABLED_APIS}"; then
            pass "API enabled: ${api}"
        else
            fail "API not enabled: ${api}"
            fix "gcloud services enable ${api} --project=${GCP_PROJECT_ID}"
        fi
    done
fi

# ---------------------------------------------------------------------------
# 5. GCP_LOCATION / regional endpoint reachability (gRPC sinkhole check)
# ---------------------------------------------------------------------------
# setup_vertex_search.py targets the Discovery Engine endpoint matching
# GCP_LOCATION (global -> discoveryengine.googleapis.com, otherwise
# <location>-discoveryengine.googleapis.com). If that endpoint doesn't exist
# or isn't reachable, the Python client's LRO calls don't fail fast — they
# hang until the RPC/LRO timeout (up to 600s) with no useful error. Catch
# a bad/unreachable GCP_LOCATION here in seconds instead.
# Discovery Engine only serves 'global'/'us'/'eu'. The Python code DERIVES this
# from GCP_LOCATION (config/settings.py::discovery_engine_location): a compute
# region like 'us-central1' maps to 'us'. Mirror that derivation here so this
# check matches runtime behavior instead of rejecting valid compute regions.

case "$(echo $GCP_LOCATION | tr '[:upper:]' '[:lower:]')" in
    us*)            DE_REGION="us" ;;
    eu*|europe*)    DE_REGION="eu" ;;
    global|"")      DE_REGION="global" ;;
    *)              DE_REGION="global" ;;
esac

if [[ "${DE_REGION}" == "global" ]]; then
    DISCOVERYENGINE_ENDPOINT="discoveryengine.googleapis.com"
else
    DISCOVERYENGINE_ENDPOINT="${DE_REGION}-discoveryengine.googleapis.com"
fi

if [[ "${GCP_LOCATION}" != "${DE_REGION}" ]]; then
    warn "GCP_LOCATION='${GCP_LOCATION}' is a compute region; Discovery Engine calls are routed to the '${DE_REGION}' multi-region (derived). This is handled automatically by settings.discovery_engine_location."
fi

if ! getent hosts "${DISCOVERYENGINE_ENDPOINT}" >/dev/null 2>&1 && ! host "${DISCOVERYENGINE_ENDPOINT}" >/dev/null 2>&1 && ! nslookup "${DISCOVERYENGINE_ENDPOINT}" >/dev/null 2>&1; then
    fail "Cannot resolve Discovery Engine endpoint '${DISCOVERYENGINE_ENDPOINT}' (derived from GCP_LOCATION='${GCP_LOCATION}')"
    fix "Check DNS/network connectivity, and confirm GCP_LOCATION='${GCP_LOCATION}' maps to a valid region (us/eu/global).
The Python client will otherwise fail with INVALID_ARGUMENT or hang until its RPC/LRO timeout."
else
    pass "Discovery Engine region '${DE_REGION}' resolves to reachable endpoint: ${DISCOVERYENGINE_ENDPOINT}"
fi

# ---------------------------------------------------------------------------
# 6. IAM permissions required by setup_vertex_search.py
# ---------------------------------------------------------------------------
# Full end-to-end permission set: provisioning (setup_vertex_search.py),
# ingestion import + review (batch_ingest.py, review_datastore.py), purge
# (purge_datastore.py), and GCS staging (uploader.py). Grouped by the script
# that needs each. Gemini (generativelanguage) auth is validated separately by
# scripts/check_llm.py — it is not an IAM permission on this project.
REQUIRED_PERMISSIONS=(
    # project / service usage
    "resourcemanager.projects.get"
    "serviceusage.services.list"
    # provisioning — setup_vertex_search.py
    "discoveryengine.dataStores.create"
    "discoveryengine.dataStores.list"
    "discoveryengine.dataStores.get"
    "discoveryengine.schemas.create"
    "discoveryengine.schemas.update"
    "discoveryengine.schemas.get"
    "discoveryengine.engines.create"
    "discoveryengine.engines.get"
    # ingestion / review / purge — batch_ingest.py, review_datastore.py, purge_datastore.py
    "discoveryengine.documents.import"
    "discoveryengine.documents.list"
    "discoveryengine.documents.get"
    "discoveryengine.documents.delete"
    # GCS staging — uploader.py
    # "storage.objects.create"
    # "storage.objects.get"
    # "storage.objects.list"
    # "storage.buckets.get"
)

# testIamPermissions is exposed only via the Cloud Resource Manager REST API,
# called with curl using the shared ADC access token acquired in section 1.

if [[ -z "${ACCESS_TOKEN}" ]]; then
    fail "Cannot check IAM permissions: no Application Default Credentials access token available"
    fix "gcloud auth application-default login"
else
    PERM_JSON_ARRAY="$(printf '"%s",' "${REQUIRED_PERMISSIONS[@]}")"
    PERM_JSON_ARRAY="[${PERM_JSON_ARRAY%,}]"

    IAM_RESPONSE_FILE="$(mktemp)"
    trap 'rm -f "${IAM_RESPONSE_FILE}"' EXIT

    HTTP_STATUS="$(curl -s -o "${IAM_RESPONSE_FILE}" -w '%{http_code}' -X POST \
        -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        -H "Content-Type: application/json" \
        "https://cloudresourcemanager.googleapis.com/v1/projects/${GCP_PROJECT_ID}:testIamPermissions" \
        -d "{\"permissions\":${PERM_JSON_ARRAY}}")"
    CURL_EXIT=$?

    if [[ ${CURL_EXIT} -ne 0 ]]; then
        fail "Unable to reach cloudresourcemanager.googleapis.com to check IAM permissions (curl exit code ${CURL_EXIT})"
        fix "Check network/proxy connectivity, then re-run: scripts/preflight_check.sh"
    elif [[ "${HTTP_STATUS}" != "200" ]]; then
        fail "IAM permission check request failed (HTTP ${HTTP_STATUS}): $(cat "${IAM_RESPONSE_FILE}")"
        fix "Verify ${ACTIVE_ACCOUNT:-the active account} has resourcemanager.projects.getIamPolicy on '${GCP_PROJECT_ID}', then re-run: scripts/preflight_check.sh"
    else
        GRANTED_PERMISSIONS="$(python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except ValueError as exc:
    sys.exit(f"invalid JSON from testIamPermissions response: {exc}")
print("\n".join(data.get("permissions", [])))
' < "${IAM_RESPONSE_FILE}")"
        PARSE_EXIT=$?

        if [[ ${PARSE_EXIT} -ne 0 ]]; then
            fail "Could not parse testIamPermissions response for '${GCP_PROJECT_ID}'"
            fix "Inspect the raw response: cat ${IAM_RESPONSE_FILE}
(this preflight check will delete that file on exit — copy it first if you need to debug)"
        else
            MISSING_PERMISSIONS=()
            for perm in "${REQUIRED_PERMISSIONS[@]}"; do
                if ! grep -qx "${perm}" <<< "${GRANTED_PERMISSIONS}"; then
                    MISSING_PERMISSIONS+=("${perm}")
                fi
            done

            if [[ ${#MISSING_PERMISSIONS[@]} -eq 0 ]]; then
                pass "IAM permissions sufficient for ${ACTIVE_ACCOUNT:-active account} to run setup_vertex_search.py"
            else
                fail "Missing IAM permission(s) for ${ACTIVE_ACCOUNT:-active account}: ${MISSING_PERMISSIONS[*]}"
                fix "gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \\
    --member=\"user:${ACTIVE_ACCOUNT:-YOUR_EMAIL}\" --role=\"roles/discoveryengine.admin\"
gcloud projects add-iam-policy-binding ${GCP_PROJECT_ID} \\
    --member=\"user:${ACTIVE_ACCOUNT:-YOUR_EMAIL}\" --role=\"roles/serviceusage.serviceUsageViewer\""
            fi
        fi
    fi

    rm -f "${IAM_RESPONSE_FILE}"
    trap - EXIT
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================================="
if [[ ${FAILURES} -eq 0 ]]; then
    echo "${CHECK_MARK} All preflight checks passed. Safe to run:"
    echo "    PYTHONPATH=. python scripts/setup_vertex_search.py --dry-run"
    exit 0
else
    echo "${CROSS_MARK} ${FAILURES} preflight check(s) failed. Fix the issues above before running setup_vertex_search.py."
    exit 1
fi
