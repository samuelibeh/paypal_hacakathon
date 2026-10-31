#!/usr/bin/env bash
# Runs on the VPS, started by .github/workflows/deploy.yml after it copies the new code to
# ~/scout-next. Tests that copy and, only if the tests pass, makes it the live app in ~/scout
# and restarts the service. The server's own .env, merchants.json and venv are never touched.
set -euo pipefail

APP="$HOME/scout"
NEXT="$HOME/scout-next"
cd "$NEXT"

# New or changed packages only (the venv already has the CPU build of PyTorch).
if ! cmp -s requirements.txt "$APP/requirements.txt"; then
  echo "requirements.txt changed: installing"
  "$APP/venv/bin/pip" install -q -r requirements.txt
fi

echo "Running the tests"
MONGODB_URI= "$APP/venv/bin/python" -m pytest -q --deselect tests/test_claude_live.py

echo "Tests passed: switching the live app over"
rsync -a --delete \
  --exclude '.env' --exclude 'merchants.json' --exclude 'venv/' \
  --exclude '__pycache__/' --exclude '.pytest_cache/' \
  "$NEXT/" "$APP/"
sudo -n systemctl restart scout

for _ in $(seq 1 45); do
  if curl -fs -o /dev/null http://127.0.0.1:5000/health; then  # quiet while it starts up
    echo "Scout is up"
    exit 0
  fi
  sleep 2
done
echo "Scout didn't come back after the restart" >&2
sudo -n journalctl -u scout -n 40 --no-pager >&2
exit 1
