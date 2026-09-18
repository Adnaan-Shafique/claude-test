# Running the demo as a service

`app/demo_dash_modes.py` on port 7872, started at boot and restarted if it
dies. Templates live in `deploy/`.

Everything below assumes the field-ops VM:

| | |
|---|---|
| Host | `AIOPS-POC01` (10.19.75.122) |
| Project | `/data/adnaan/field_ops/integrated_pipeline` |
| venv | `venv_yolox` (must be the one with `_ctypes` — see SETUP.md) |
| Route | proxy, via `llm_proxy_v3` on FALCONPRD — see `PROXY.md` |

Confirm the first three before you start; a unit file with a wrong path fails
with a bare `203/EXEC` that says nothing about which path was wrong:

```bash
cd /data/adnaan/field_ops/integrated_pipeline && pwd
id                       # the User= and Group= to use
./venv_yolox/bin/python -c "import _ctypes; print('venv OK')"
```

## 1. The secret, separately

The API key does **not** go in the unit file. Unit files are world-readable and
`systemctl cat` prints them to anyone who asks.

```bash
sudo install -m 600 /dev/null /etc/fieldops-demo.env
sudo vi /etc/fieldops-demo.env        # contents from deploy/fieldops-demo.env.example
sudo chown root:root /etc/fieldops-demo.env
ls -l /etc/fieldops-demo.env          # expect -rw-------
```

## 2. The unit

```bash
sudo cp deploy/fieldops-demo-modes.service /etc/systemd/system/
sudo vi /etc/systemd/system/fieldops-demo-modes.service   # User, Group, paths
sudo systemctl daemon-reload
sudo systemctl enable --now fieldops-demo-modes
```

## 3. Check it

```bash
systemctl status fieldops-demo-modes
journalctl -u fieldops-demo-modes -f          # Ctrl-C to stop following
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:7872/    # expect 200
```

Then open `http://10.19.75.122:7872/` and press **Load / check model**. It
should report the proxy route, not "answers will be MOCK". If it reports MOCK,
the environment file is not reaching the process — `systemctl show
fieldops-demo-modes -p Environment` shows what it actually got.

## If you have no root

A user service needs no sudo, but it stops when you log out unless lingering is
enabled — and enabling lingering itself needs root. Without either, the service
dies with your SSH session, which is not a service.

```bash
mkdir -p ~/.config/systemd/user
cp deploy/fieldops-demo-modes.service ~/.config/systemd/user/
# Remove the User=, Group= and EnvironmentFile= lines; put the variables in
# ~/.config/fieldops-demo.env (chmod 600) and point EnvironmentFile there.
sed -i 's|WantedBy=multi-user.target|WantedBy=default.target|' \
    ~/.config/systemd/user/fieldops-demo-modes.service
systemctl --user daemon-reload
systemctl --user enable --now fieldops-demo-modes
sudo loginctl enable-linger $USER      # needs root ONCE; without it, see above
```

## The firewall

Binding 0.0.0.0 is not enough — the host still has to allow the port:

```bash
sudo firewall-cmd --permanent --add-port=7872/tcp && sudo firewall-cmd --reload
sudo firewall-cmd --list-ports
```

If the page loads from the VM itself (`curl 127.0.0.1:7872`) but not from a
laptop, it is the firewall, not the app.

## Updating

```bash
cd /data/adnaan/field_ops/integrated_pipeline
git pull
for t in tests/test_*.py; do ./venv_yolox/bin/python "$t" >/dev/null || echo "FAILED $t"; done
sudo systemctl restart fieldops-demo-modes
```

Restart is required for a code change: the process imports everything once at
start. **Hard-reload the browser tab** afterwards (Ctrl-Shift-R) — a cached page
keeps the old callback graph and fails in confusing ways when the layout has
changed.

## What the service does NOT survive

`config/prompts.yaml` — mode 3's tuned prompts — lives in the project directory
and is read **once at import**. Editing prompts through the UI writes the file
and takes effect immediately in that process, but editing the file by hand
needs a restart.

Run outputs accumulate under `demo_runs/` and nothing prunes them. Each run
writes annotated copies of every photograph across three modes, so a long
rehearsal is measured in gigabytes. Before a demo:

```bash
du -sh demo_runs/
# keep the last few, delete the rest
```

The service user must own the project directory: the app writes `demo_runs/`,
`config/prompts.yaml` and the staged uploads folder. Running it as a user who
can only read the checkout fails at the first run, not at startup.

## Werkzeug, not gunicorn

This runs Dash's built-in server. That is a development server, and it says so
in the log — for a demo on a closed network in front of a known audience it is
the right trade: no extra dependency, no extra failure mode, and it is threaded,
so a slow pipeline run does not block the page.

If this ever outlives the demo, `app/demo_dash_modes.py` already exposes
`server = app.server`, so the move is one line:

```
ExecStart=.../venv_yolox/bin/gunicorn --chdir app --workers 1 --threads 4 \
          --timeout 600 --bind 0.0.0.0:7872 demo_dash_modes:server
```

Keep `--workers 1`. Results are held in a module-level dict, so a second worker
would answer half the requests from an empty set and the mode switcher would
show blanks at random.
