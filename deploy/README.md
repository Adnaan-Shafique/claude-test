# Running on RHEL as systemd services

Two services, because there are two processes:

| Service | Script | Listens on |
|---|---|---|
| `ippms-mcp` | `instant_graph_mcp_server_v2_4_1.py` | `127.0.0.1:8056` (MCP), `127.0.0.1:8060` (ops console) |
| `ippms-app` | `talk_to_vi_ippms_updated_6_5_3.py` | `0.0.0.0:8079` (chat UI) |

`ippms-app` talks to `ippms-mcp` over `MCP_SERVER_URL`, and to your GPU
inference proxy over `GPU_PROXY_URL`. The GPU proxy is a separate service you
already run — nothing here manages it.

## 1. One-time setup

```bash
# Service account with no login and no home
sudo useradd --system --no-create-home --shell /sbin/nologin ippms

sudo mkdir -p /srv/ippms-assistant /etc/ippms-assistant
sudo chown -R ippms:ippms /srv/ippms-assistant

# Code + the files the app reads at startup
sudo install -o ippms -g ippms -m 644 \
    talk_to_vi_ippms_updated_6_5_3.py \
    instant_graph_mcp_server_v2_4_1.py \
    vi_ippms_tool_kb.md \
    vi_ippms_question_guide.xlsx \
    /srv/ippms-assistant/
# ...plus the Instant Graph self-signed cert
sudo install -o ippms -g ippms -m 644 ig_selfsigned.pem /srv/ippms-assistant/

# Virtualenv — don't use --break-system-packages on a server
sudo -u ippms python3 -m venv /srv/ippms-assistant/venv
sudo -u ippms /srv/ippms-assistant/venv/bin/pip install --upgrade pip
sudo -u ippms /srv/ippms-assistant/venv/bin/pip install \
    langgraph "mcp[cli]" dash plotly pandas requests psycopg2-binary openpyxl
```

## 2. Secrets

```bash
sudo cp deploy/ippms.env.example /etc/ippms-assistant/ippms.env
sudo vi /etc/ippms-assistant/ippms.env          # fill in the CHANGE_ME values
sudo chown root:ippms /etc/ippms-assistant/ippms.env
sudo chmod 640        /etc/ippms-assistant/ippms.env
```

Secrets go in this file, never in the unit files — unit files under
`/etc/systemd/system` are world-readable.

## 3. Install and start

```bash
sudo cp deploy/ippms-mcp.service deploy/ippms-app.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ippms-mcp ippms-app
```

`enable` is what makes them come back after a reboot; `Restart=always` is what
makes them come back after a crash.

## 4. Verify

```bash
systemctl status ippms-mcp ippms-app
journalctl -u ippms-app -f              # live logs
journalctl -u ippms-app -n 200 --no-pager

# Startup line confirming which subsystems came up:
journalctl -u ippms-app | grep '\[STARTUP\]'

# Prove the restart-on-crash actually works:
sudo systemctl kill -s SIGKILL ippms-app
sleep 8 && systemctl is-active ippms-app     # expect: active
```

## 5. Firewall (only the chat UI should be reachable)

```bash
sudo firewall-cmd --permanent --add-port=8079/tcp
sudo firewall-cmd --reload
```

8056 and 8060 stay on loopback (see `MCP_HOST`/`DASH_HOST` in the env file).
The MCP server exposes the Instant Graph tools with no authentication of its
own, so it should not be reachable from the network. Reach the ops console
with an SSH tunnel when you need it:

```bash
ssh -L 8060:127.0.0.1:8060 you@rhel-host    # then browse http://localhost:8060
```

### SELinux

Usually nothing to do — services started by systemd from `/srv` binding a high
port work under the default policy. If a bind is denied, check first and only
then add a rule:

```bash
sudo ausearch -m avc -ts recent | grep -i ippms
sudo semanage port -a -t http_port_t -p tcp 8079    # only if actually needed
```

## Restarts

- **Crash / OOM-kill / unhandled exception** → `Restart=always`, `RestartSec=5s`.
- **Reboot** → `systemctl enable`.
- **Repeated fast failures** → `StartLimitIntervalSec=0` disables systemd's
  default rate limit (5 starts in 10s then give up permanently). Without it,
  a service whose dependency is slow to come back after a reboot ends up dead
  and stays dead.
- **Hung but alive** (process up, not serving) → systemd cannot detect this on
  its own. If you want that covered, add the optional healthcheck below.

### Optional: restart if it stops answering

`Restart=always` only catches a process that exits. To also catch a wedged one,
add a timer that probes the UI and restarts on failure:

```ini
# /etc/systemd/system/ippms-app-healthcheck.service
[Unit]
Description=Restart ippms-app if it stops answering
[Service]
Type=oneshot
ExecStart=/bin/bash -c '/usr/bin/curl -fsS --max-time 10 -o /dev/null http://127.0.0.1:8079/ || /usr/bin/systemctl restart ippms-app'
```

```ini
# /etc/systemd/system/ippms-app-healthcheck.timer
[Unit]
Description=Probe ippms-app every 2 minutes
[Timer]
OnBootSec=5min
OnUnitActiveSec=2min
[Install]
WantedBy=timers.target
```

```bash
sudo systemctl enable --now ippms-app-healthcheck.timer
```

## Updating the code

```bash
sudo install -o ippms -g ippms -m 644 talk_to_vi_ippms_updated_6_5_3.py /srv/ippms-assistant/
sudo systemctl restart ippms-app
```

Restarting logs everyone out — browser sessions live in memory in the app
process (`_SESSIONS`), not in Postgres. Chat history is in Postgres and
survives.

---

## Three things to know before this goes live

### 1. Credentials are hardcoded in the source

Both files carry a real-looking fallback despite comments claiming otherwise:

```python
# talk_to_vi_ippms_updated_6_5_3.py:233  and  instant_graph_mcp_server_v2_4_1.py:147
DB_PASSWORD = os.environ.get("IG_DB_PASSWORD", "ig_app_user_Qwer!234")
# talk_to_vi_ippms_updated_6_5_3.py:172
GPU_API_KEY = os.environ.get("GPU_API_KEY", "secret-falcon2")
```

These are in git history. Setting the env vars above means the fallbacks are
never *used*, but the values are still *published*. Rotate them, and change
the fallbacks to `""` so a missing env var fails loudly instead of silently
connecting with a known password.

### 2. Run exactly one process — do not scale out

Browser sessions (`_SESSIONS`) and the answer cache are per-process, in memory.
Two workers means users randomly appear logged out as requests land on the
worker that never saw their login. So: no `--workers 2`, and no second host
behind a load balancer without sticky sessions.

(The *Instant Graph token* is fine across processes — it's persisted to
`ig_auth_sessions` in Postgres for exactly that reason. It's the app's own
session layer that isn't.)

### 3. If you later switch to gunicorn

The unit runs `python <script>` deliberately. Under a WSGI server,
`_warm_startup()` never runs, because it only executes under
`if __name__ == "__main__"` — so the **7-day chat-history retention sweeper
never starts** and old conversations are never purged. If you do move to
gunicorn, call `_warm_startup()` from a WSGI entrypoint module and keep it to
one worker:

```bash
gunicorn --workers 1 --threads 8 --timeout 300 --bind 0.0.0.0:8079 wsgi:server
```

```python
# wsgi.py
from talk_to_vi_ippms_updated_6_5_3 import server, _warm_startup
_warm_startup()
```
