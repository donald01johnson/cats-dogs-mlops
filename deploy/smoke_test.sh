#!/bin/bash
# =============================================================
# deploy/smoke_test.sh
# Post-deployment smoke test for Cats vs Dogs API
#
# Tests:
#   1. GET  /health  — service is up, model is loaded
#   2. POST /predict — inference returns a valid label
#
# Exit codes:
#   0 = all tests passed
#   1 = one or more tests failed (CI/CD pipeline will fail)
#
# Usage:
#   chmod +x deploy/smoke_test.sh
#   ./deploy/smoke_test.sh
# =============================================================

set -e  # Exit immediately on any error

# ── Configuration ──────────────────────────────────────────────
API_BASE_URL="${API_URL:-http://localhost:8000}"
TEST_IMAGE="${TEST_IMAGE:-deploy/test_sample.jpg}"
MAX_RETRIES=10
RETRY_INTERVAL=5

# ── Color output ───────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No color

pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}[INFO]${NC} $1"; }

# ── Banner ─────────────────────────────────────────────────────
echo "=============================================="
echo "  Cats vs Dogs API — Post-Deploy Smoke Tests"
echo "  Target: $API_BASE_URL"
echo "=============================================="

# ── Wait for service to be ready ───────────────────────────────
info "Waiting for API to become ready (max ${MAX_RETRIES} retries)..."

for i in $(seq 1 $MAX_RETRIES); do
    if curl -sf "$API_BASE_URL/health" > /dev/null 2>&1; then
        info "API is responding after $i attempt(s)."
        break
    fi

    if [ $i -eq $MAX_RETRIES ]; then
        fail "API did not become ready after $((MAX_RETRIES * RETRY_INTERVAL)) seconds."
    fi

    info "Attempt $i/$MAX_RETRIES — not ready yet, retrying in ${RETRY_INTERVAL}s..."
    sleep $RETRY_INTERVAL
done

echo ""
echo "--- Running Smoke Tests ---"
echo ""

# ══════════════════════════════════════════════════════════════
# TEST 1: GET /health — Service is up
# ══════════════════════════════════════════════════════════════
info "Test 1: GET /health — checking service status..."

HEALTH_RESPONSE=$(curl -sf "$API_BASE_URL/health" 2>&1) || \
    fail "Test 1: /health endpoint returned non-200 status. Is the container running?"

echo "  Response: $HEALTH_RESPONSE"

# Validate 'status' field
echo "$HEALTH_RESPONSE" | grep -q '"status":"ok"' || \
    fail "Test 1: /health response does not contain '\"status\":\"ok\"'. Got: $HEALTH_RESPONSE"

# Validate 'model_loaded' is true
echo "$HEALTH_RESPONSE" | grep -q '"model_loaded":true' || \
    fail "Test 1: model_loaded is not true. Model may have failed to load. Got: $HEALTH_RESPONSE"

pass "Test 1: /health returned status=ok and model_loaded=true"
echo ""

# ══════════════════════════════════════════════════════════════
# TEST 2: POST /predict — Inference returns valid label
# ══════════════════════════════════════════════════════════════
info "Test 2: POST /predict — testing inference with sample image..."

# Check test image exists
if [ ! -f "$TEST_IMAGE" ]; then
    fail "Test 2: Test image not found at '$TEST_IMAGE'. Add deploy/test_sample.jpg to the repo."
fi

PREDICT_RESPONSE=$(curl -sf \
    -X POST "$API_BASE_URL/predict" \
    -F "file=@$TEST_IMAGE" \
    2>&1) || \
    fail "Test 2: /predict endpoint returned non-200 status. Response: $PREDICT_RESPONSE"

echo "  Response: $PREDICT_RESPONSE"

# Validate label is 'cat' or 'dog'
echo "$PREDICT_RESPONSE" | grep -qE '"label":"(cat|dog)"' || \
    fail "Test 2: /predict response does not contain a valid label (cat or dog). Got: $PREDICT_RESPONSE"

# Validate confidence field exists
echo "$PREDICT_RESPONSE" | grep -q '"confidence":' || \
    fail "Test 2: /predict response missing 'confidence' field. Got: $PREDICT_RESPONSE"

# Validate class_probabilities field exists
echo "$PREDICT_RESPONSE" | grep -q '"class_probabilities":' || \
    fail "Test 2: /predict response missing 'class_probabilities' field. Got: $PREDICT_RESPONSE"

pass "Test 2: /predict returned valid label with confidence and class_probabilities"
echo ""

# ══════════════════════════════════════════════════════════════
# TEST 3: GET /metrics — Prometheus metrics endpoint
# ══════════════════════════════════════════════════════════════
info "Test 3: GET /metrics — checking Prometheus metrics endpoint..."

METRICS_RESPONSE=$(curl -sf "$API_BASE_URL/metrics" 2>&1) || \
    fail "Test 3: /metrics endpoint returned non-200 status."

# Verify it contains Prometheus-style output
echo "$METRICS_RESPONSE" | grep -q "# HELP" || \
    fail "Test 3: /metrics does not contain Prometheus format (# HELP). Got: $METRICS_RESPONSE"

pass "Test 3: /metrics returned valid Prometheus exposition format"
echo ""

# ══════════════════════════════════════════════════════════════
# ALL TESTS PASSED
# ══════════════════════════════════════════════════════════════
echo "=============================================="
echo -e "${GREEN}  ALL SMOKE TESTS PASSED ✅${NC}"
echo "  Deployment is healthy and functional."
echo "=============================================="
exit 0
