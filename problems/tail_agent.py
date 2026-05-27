#!/usr/bin/env python3
"""
tail_agent.py — poll the hermes SQLite session DB and print new messages as they arrive.
Usage: python3 problems/tail_agent.py [session_id]
       If session_id omitted, watches the most recently started CLI session.
"""
import sqlite3, json, sys, time, textwrap, os

DB = os.path.expanduser("~/.hermes/state.db")
POLL_INTERVAL = 2.0

def get_latest_cli_session(conn):
    row = conn.execute(
        "SELECT id FROM sessions WHERE source='cli' ORDER BY started_at DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else None

def fmt_message(row):
    sid, role, tool_name, content, ts = row
    t = time.strftime("%H:%M:%S", time.localtime(ts))
    if role == "assistant":
        # content may be empty (streaming placeholder) or text
        text = content or ""
        if text.strip():
            return f"\n[{t}] ASSISTANT\n{textwrap.fill(text[:600], width=100)}"
        return None
    elif role == "tool":
        name = tool_name or "?"
        try:
            data = json.loads(content or "{}")
            out = data.get("output", data.get("error", content or ""))
        except Exception:
            out = content or ""
        out = str(out)[:300].replace("\n", " ")
        return f"[{t}] TOOL({name}) → {out}"
    elif role == "user":
        return f"[{t}] USER: {(content or '')[:200]}"
    else:
        return f"[{t}] {role.upper()}: {(content or '')[:200]}"

def main():
    session_id = sys.argv[1] if len(sys.argv) > 1 else None
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    if not session_id:
        session_id = get_latest_cli_session(conn)
        if not session_id:
            print("No CLI session found.")
            sys.exit(1)
        print(f"Watching session: {session_id}")

    seen_ids = set()
    print(f"Polling {DB} every {POLL_INTERVAL}s ... Ctrl+C to stop\n{'='*60}")

    try:
        while True:
            # Re-open connection each poll to bypass WAL cache
            conn.close()
            conn = sqlite3.connect(DB)
            rows = conn.execute(
                "SELECT session_id, role, tool_name, content, timestamp "
                "FROM messages WHERE session_id=? ORDER BY timestamp ASC",
                (session_id,)
            ).fetchall()

            new_rows = [r for r in rows if r[4] not in seen_ids]
            for r in new_rows:
                seen_ids.add(r[4])  # timestamp as unique key (good enough)
                line = fmt_message(r)
                if line:
                    print(line)

            # Check if session ended
            sess = conn.execute(
                "SELECT ended_at, message_count, input_tokens, output_tokens FROM sessions WHERE id=?",
                (session_id,)
            ).fetchone()
            if sess and sess[0]:  # ended_at is set
                print(f"\n{'='*60}")
                print(f"Session ended. messages={sess[1]} input={sess[2]} output={sess[3]}")
                break

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
