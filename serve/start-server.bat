@echo off
REM Starts the Bu Melati proxy and a Cloudflare quick tunnel.
REM Ollama must already be running with the bu-melati model.
REM Copy the https://....trycloudflare.com address it prints into GAME_CONFIG.serverUrl
REM in index.html, adding /v1/chat/completions at the end.

set PORT=8010
cd /d %~dp0

REM Proxy in its own window, using the project's venv (no need to activate it first)
start "Bu Melati proxy (port %PORT%)" cmd /k ""%~dp0..\.venv\Scripts\python.exe" -m uvicorn proxy:app --host 127.0.0.1 --port %PORT%"
timeout /t 3 >nul

REM Public tunnel to the proxy
"%~dp0cloudflared.exe" tunnel --url http://localhost:%PORT%
