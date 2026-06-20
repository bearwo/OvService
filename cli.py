from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))

import config
config.setup_openvino()

from core.engine import ModelEngine
from core.conversation import Conversation
from adapters.chat import ChatAdapter
from features.memory import ConversationMemory
from features.image import load_image_as_tensor

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console = Console()
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    class _FakeConsole:
        def print(self, *a, **kw):
            end = kw.get("end", "\n")
            for arg in a:
                print(arg, end=end, flush=True)
        def input(self, prompt=""):
            return input(prompt)
    console = _FakeConsole()


BANNER = """
[bold cyan]OvService[/bold cyan] — OpenVINO GenAI Chat CLI
Type your message and press Enter. Commands start with /
Type /quit to exit.
""".strip()

HELP_TEXT = """
Commands:
  /clear              Clear current conversation
  /history            Show conversation history
  /config             Show current config
  /image <path>       Ask about an image (VLM)
  /think              Enable thinking mode
  /nothink            Disable thinking mode
  /sessions           List all saved sessions
  /session new        Start a new session
  /session load <id>  Load a saved session
  /session export     Export current session to JSON
  /model list         List all registered models
  /model load <name>  Load a model
  /model unload <name> Unload a model
  /help               Show this help
  /quit               Exit
""".strip()


def cmd_help():
    if HAS_RICH:
        console.print(Panel(HELP_TEXT, title="Commands", border_style="cyan"))
    else:
        print(HELP_TEXT)


def cmd_config(engine: ModelEngine):
    active = engine.active()
    if active:
        s = active.status()
        if HAS_RICH:
            table = Table(title="Current Config")
            table.add_column("Key", style="cyan")
            table.add_column("Value")
            table.add_row("Active Model", s.name)
            table.add_row("Model Path", s.model_path)
            table.add_row("Device", s.device)
            table.add_row("Load Time", f"{s.load_time_ms:.0f}ms")
            console.print(table)
        else:
            print(f"Active Model: {s.name}")
            print(f"Model Path: {s.model_path}")
            print(f"Device: {s.device}")
            print(f"Load Time: {s.load_time_ms:.0f}ms")
    else:
        print("No model loaded.")


def cmd_model_list(engine: ModelEngine):
    statuses = engine.list_models()
    if not statuses:
        print("No models registered.")
        return
    if HAS_RICH:
        table = Table(title="Registered Models")
        table.add_column("Name", style="cyan")
        table.add_column("Status")
        table.add_column("Device")
        table.add_column("Load Time")
        for s in statuses:
            status = "[green]Loaded[/green]" if s.loaded else "[dim]Unloaded[/dim]"
            table.add_row(s.name, status, s.device, f"{s.load_time_ms:.0f}ms")
        console.print(table)
    else:
        for s in statuses:
            status = "Loaded" if s.loaded else "Unloaded"
            print(f"{s.name}: {status} | {s.device} | {s.load_time_ms:.0f}ms")


def cmd_sessions(memory: ConversationMemory):
    sessions = memory.list_sessions(limit=20)
    if not sessions:
        print("No saved sessions.")
        return
    import datetime
    if HAS_RICH:
        table = Table(title="Saved Sessions")
        table.add_column("Session ID", style="cyan")
        table.add_column("Messages")
        table.add_column("Last Active")
        for s in sessions:
            sid = s["session_id"][:16] + "..."
            msgs = str(s["msg_count"])
            ts = datetime.datetime.fromtimestamp(s["updated_at"]).strftime("%m-%d %H:%M")
            table.add_row(sid, msgs, ts)
        console.print(table)
    else:
        for s in sessions:
            sid = s["session_id"][:16] + "..."
            msgs = str(s["msg_count"])
            ts = datetime.datetime.fromtimestamp(s["updated_at"]).strftime("%m-%d %H:%M")
            print(f"{sid} | {msgs} messages | {ts}")


def handle_command(cmd: str, engine: ModelEngine, conv: Conversation,
                   memory: ConversationMemory, session_id: str) -> tuple[bool, str | None]:
    parts = cmd.strip().split()
    if not parts:
        return True, session_id

    command = parts[0].lower()

    if command == "/quit" or command == "/exit":
        return False, session_id
    elif command == "/help":
        cmd_help()
    elif command == "/clear":
        conv.clear()
        console.print("[dim]Conversation cleared.[/dim]")
    elif command == "/history":
        console.print(conv.summary())
    elif command == "/config":
        cmd_config(engine)
    elif command == "/think":
        adapter = engine.active()
        if adapter and hasattr(adapter, "thinking"):
            adapter.thinking = True
            console.print("[dim]Thinking mode: ON[/dim]")
        else:
            console.print("[dim]No active model.[/dim]")
    elif command == "/nothink":
        adapter = engine.active()
        if adapter and hasattr(adapter, "thinking"):
            adapter.thinking = False
            console.print("[dim]Thinking mode: OFF[/dim]")
        else:
            console.print("[dim]No active model.[/dim]")
    elif command == "/sessions":
        cmd_sessions(memory)
    elif command == "/session":
        if len(parts) < 2:
            console.print("Usage: /session <new|load|export> [id]")
            return True, session_id
        sub = parts[1].lower()
        if sub == "new":
            old_id = session_id
            conv.clear()
            session_id = str(uuid.uuid4())[:8]
            ctx = memory.get_context_memory(old_id)
            if ctx:
                conv.set_memory_context(
                    f"Previous conversation summary:\n{ctx}\n\n"
                    "Continue the conversation based on the above context."
                )
            console.print(f"[dim]New session: {session_id}[/dim]")
            return True, session_id
        elif sub == "load" and len(parts) >= 3:
            target_id = parts[2]
            sessions = memory.list_sessions(limit=50)
            found = None
            for s in sessions:
                if s["session_id"].startswith(target_id):
                    found = s
                    break
            if found:
                sid = found["session_id"]
                messages = memory.get_messages(sid)
                conv.clear()
                for msg in messages:
                    if msg["role"] in ("user", "assistant"):
                        if msg["role"] == "user":
                            conv.add_user(msg["content"])
                        else:
                            conv.add_assistant(msg["content"])
                session_id = sid[:8]
                ctx = memory.get_context_memory(sid)
                if ctx:
                    conv.set_memory_context(
                        f"Previous conversation summary:\n{ctx}\n\n"
                        "Continue the conversation based on the above context."
                    )
                console.print(f"[dim]Loaded session {session_id} ({len(messages)} messages)[/dim]")
            else:
                console.print(f"[red]Session '{target_id}' not found.[/red]")
        elif sub == "export":
            exported = memory.export_session(session_id)
            export_path = config.DATA_DIR / f"session_{session_id}.json"
            export_path.write_text(exported, encoding="utf-8")
            console.print(f"[dim]Exported to {export_path}[/dim]")
        else:
            console.print("Usage: /session <new|load|export> [id]")
    elif command == "/image":
        if len(parts) < 2:
            console.print("Usage: /image <path> [question]")
            return True, session_id
        image_path = Path(parts[1])
        if not image_path.exists():
            console.print(f"[red]Image not found: {image_path}[/red]")
            return True, session_id
        question = " ".join(parts[2:]) if len(parts) > 2 else "Describe this image in detail."
        try:
            tensor = load_image_as_tensor(image_path)
        except Exception as e:
            console.print(f"[red]Failed to load image: {e}[/red]")
            return True, session_id
        memory.save_message(session_id, "user", f"[Image: {image_path.name}] {question}")
        if HAS_RICH:
            console.print("[bold green]AI:[/bold green] ", end="", highlight=False)
        else:
            print("AI: ", end="", flush=True)
        result = engine.generate(
            [{"role": "user", "content": question}],
            images=[tensor],
        )
        text = result.text if hasattr(result, "text") else str(result)
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()
        conv.add_user(f"[Image: {image_path.name}] {question}")
        conv.add_assistant(text)
        memory.save_message(session_id, "assistant", text)
    elif command == "/model":
        if len(parts) < 2:
            console.print("Usage: /model <list|load|unload> [name]")
            return True, session_id
        sub = parts[1].lower()
        if sub == "list":
            cmd_model_list(engine)
        elif sub == "load" and len(parts) >= 3:
            name = parts[2]
            console.print(f"Loading {name}...")
            if engine.load(name):
                engine.set_active(name)
                console.print(f"[green]{name} loaded.[/green]")
            else:
                console.print(f"[red]Model '{name}' not found.[/red]")
        elif sub == "unload" and len(parts) >= 3:
            name = parts[2]
            if engine.unload(name):
                console.print(f"[dim]{name} unloaded.[/dim]")
            else:
                console.print(f"[red]Model '{name}' not found.[/red]")
        else:
            console.print("Usage: /model <list|load|unload> [name]")
    else:
        console.print(f"[dim]Unknown command: {command}. Type /help for commands.[/dim]")

    return True, session_id


def _auto_summarize(memory: ConversationMemory, engine: ModelEngine,
                    session_id: str, conv: Conversation) -> None:
    old_msgs = conv.get_old_messages_for_summary(keep_recent=4)
    if len(old_msgs) < 4:
        return
    dialogue = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in old_msgs)
    prompt_text = (
        "Summarize the following conversation in 3-5 sentences in the same language "
        "as the conversation. Focus on key topics, decisions, and important facts.\n\n"
        f"{dialogue}"
    )
    from core.base import GenerateConfig
    result = engine.generate(
        [{"role": "user", "content": prompt_text}],
        GenerateConfig(max_length=512, temperature=0.3),
    )
    summary_text = result.text if hasattr(result, "text") else str(result)
    msg_count = len(old_msgs)
    memory.save_summary(session_id, summary_text, msg_count, level=0)
    conv.trim_after_summary(keep_recent=4)
    _maybe_compress_db(engine, memory, session_id)
    ctx = memory.get_context_memory(session_id)
    if ctx:
        conv.set_memory_context(
            f"Previous conversation summary:\n{ctx}\n\n"
            "Continue the conversation based on the above context."
        )
    conv.consume_compression()
    if HAS_RICH:
        console.print(f"[dim]Auto-summary saved ({msg_count} messages compressed).[/dim]")


def _maybe_compress_db(engine: ModelEngine, memory: ConversationMemory, session_id: str) -> None:
    summaries = memory.get_all_summaries(session_id, level=0)
    count_ok = len(summaries) >= config.DB_COMPRESS_MAX_COUNT
    total_chars = sum(len(s["summary"]) for s in summaries)
    max_tokens = config.get_model_context_length()
    size_ok = (total_chars // 2) >= int(max_tokens * config.DB_COMPRESS_MAX_RATIO)
    if not count_ok and not size_ok:
        return
    old_ids = [s["id"] for s in summaries[:-2]]
    texts = [s["summary"] for s in summaries]
    combined = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    from core.base import GenerateConfig
    prompt_text = (
        "Compress these summaries into one concise summary preserving all key information:\n\n"
        f"{combined}"
    )
    result = engine.generate(
        [{"role": "user", "content": prompt_text}],
        GenerateConfig(max_length=1024, temperature=0.2),
    )
    compressed = result.text if hasattr(result, "text") else str(result)
    memory.save_summary(session_id, compressed, sum(s["message_count"] for s in summaries), level=1)
    memory.delete_summaries(session_id, old_ids)
    if HAS_RICH:
        console.print(f"[dim]DB compressed ({len(summaries)} summaries merged).[/dim]")


def main():
    engine = ModelEngine()
    adapter = ChatAdapter(config.CHAT_MODEL, config.DEFAULT_DEVICE)
    engine.register(adapter)
    engine.set_active("chat")

    conv = Conversation()
    memory = ConversationMemory()
    session_id = str(uuid.uuid4())[:8]

    if HAS_RICH:
        console.print(Markdown(BANNER))
    else:
        print(BANNER)

    console.print(f"Session: [cyan]{session_id}[/cyan]")
    console.print(f"Loading {adapter.name} from {adapter.model_path}...")
    adapter.load()
    console.print(f"[green]Model loaded in {adapter._load_time_ms:.0f}ms[/green]\n")

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        user_input = user_input.strip()
        if not user_input:
            continue

        if user_input.startswith("/"):
            running, session_id = handle_command(
                user_input, engine, conv, memory, session_id
            )
            if not running:
                break
            continue

        conv.add_user(user_input)
        memory.save_message(session_id, "user", user_input)

        if HAS_RICH:
            console.print("[bold green]AI:[/bold green] ", end="", highlight=False)
        else:
            print("AI: ", end="", flush=True)

        try:
            full_response = engine.generate_stream(conv.to_messages())
            if not isinstance(full_response, str):
                full_response = str(full_response)
        except Exception as e:
            console.print(f"\n[red]Generation error: {e}[/red]")
            continue
        conv.add_assistant(full_response)
        memory.save_message(session_id, "assistant", full_response)

        if conv.needs_compression:
            _auto_summarize(memory, engine, session_id, conv)

    memory.close()


if __name__ == "__main__":
    main()
