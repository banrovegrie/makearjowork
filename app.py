from __future__ import annotations

import os
import sqlite3
import secrets
import smtplib
import json
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Protocol
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from flask import Flask, request, jsonify, render_template, redirect, session, url_for
from dotenv import load_dotenv

# Google GenAI imports for Gemini Chats API
from google import genai
from google.genai import types

# Google Calendar imports (optional - graceful fallback if not installed)
_google_calendar_available = False
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build  # pyright: ignore[reportMissingImports]
    _google_calendar_available = True
except ImportError:
    service_account = None  # type: ignore[assignment]
    build = None  # type: ignore[assignment]

GOOGLE_CALENDAR_AVAILABLE: bool = _google_calendar_available

_ = load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Session configuration - stay logged in for 30 days
app.config['SESSION_COOKIE_SECURE'] = True  # HTTPS only in production
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JS access
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)

# Configuration
USE_CLOUD_SQL = os.environ.get('USE_CLOUD_SQL', 'false').lower() == 'true'
CLOUD_SQL_CONNECTION = os.environ.get('CLOUD_SQL_CONNECTION', '')  # e.g., project:region:instance
DB_USER = os.environ.get('DB_USER', 'appuser')
DB_PASS = os.environ.get('DB_PASS', '')
DB_NAME = os.environ.get('DB_NAME', 'makearjowork')
DATABASE = 'tasks.db'  # SQLite fallback for local dev

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = 'gemini-3-pro-preview'
DOMAIN = os.environ.get('DOMAIN', 'http://localhost:5001')
SMTP_HOST = os.environ.get('SMTP_HOST', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', 587))
SMTP_USER = os.environ.get('SMTP_USER', '')
SMTP_PASS = os.environ.get('SMTP_PASS', '')
FROM_EMAIL = os.environ.get('FROM_EMAIL', SMTP_USER)

# Persona configuration
PERSONA_FILE = os.path.join(os.path.dirname(__file__), 'persona.json')
_persona_cache = None
_persona_load_time = None

# Initialize Gemini client for Chats API
gemini_client = genai.Client(api_key=GEMINI_API_KEY)


def build_system_prompt(tasks_context, reads_context=""):
    """Build full system prompt with persona, current tasks, and reads"""
    persona = load_persona()
    persona_context = build_persona_context(persona)
    name = persona.get('identity', {}).get('name', 'Arjo')

    reads_section = f"\n## Current Reads\n{reads_context}\n" if reads_context else ""

    return f"""You ARE {name}. First person always.

## Who I Am
{persona_context}

## Current Tasks
{tasks_context}
{reads_section}
## What I Can Do
### Tasks
- add_task: Accept new work
- update_task: Change title/description/status
- delete_task: Remove a task
- mark_task_done: Complete a task

### Reading List
- add_read: Add paper/book to reading list (can include url, author)
- update_read: Update a reading list item
- delete_read: Remove from reading list
- mark_read_done: Mark as read
- search_arxiv: Search arxiv for a paper URL (call this first, then use the result to call add_read)

### Other
- ask_clarification: Ask when something is unclear

## Rules
- Only act on the current user message
- Previous messages are context only - don't re-execute
- Be direct and concise
- When adding papers, use search_arxiv first to get the URL
"""

def load_persona():
    """Load persona from JSON file with caching (reloads if file changes)"""
    global _persona_cache, _persona_load_time
    if not os.path.exists(PERSONA_FILE):
        return get_default_persona()
    try:
        file_mtime = os.path.getmtime(PERSONA_FILE)
        if _persona_cache and _persona_load_time and file_mtime <= _persona_load_time:
            return _persona_cache
    except OSError:
        pass
    try:
        with open(PERSONA_FILE, 'r') as f:
            _persona_cache = json.load(f)
            _persona_load_time = os.path.getmtime(PERSONA_FILE)
            return _persona_cache
    except (json.JSONDecodeError, IOError):
        return get_default_persona()

def get_default_persona():
    return {
        "identity": {"name": "Arjo"},
        "voice": {"style": "friendly", "characteristics": ["Be helpful and direct"]}
    }

def build_persona_context(persona):
    """Build persona context for system prompt - essence over credentials"""
    parts = []

    essence = persona.get('essence', {})
    if who := essence.get('who_i_am'):
        parts.append(who)
    if drives := essence.get('what_drives_me'):
        parts.append(f"What drives me: {drives}")

    if mission := persona.get('company', {}).get('mission'):
        parts.append(f"FYDY's mission: {mission}")

    voice = persona.get('voice', {})
    if chars := voice.get('characteristics'):
        parts.append("How I communicate: " + "; ".join(chars))
    if when_ask := voice.get('when_to_ask_questions'):
        parts.append("I ask questions when: " + "; ".join(when_ask))

    if notes := persona.get('context_notes'):
        parts.append("Current context: " + "; ".join(notes))

    return "\n".join(parts)

# Gemini Function Calling Tools - using SDK types for Chats API
def get_task_tools():
    """Build tool declarations using SDK types"""
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="add_task",
            description="Add a new task to my list",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "title": types.Schema(type=types.Type.STRING, description="Task title"),
                    "description": types.Schema(type=types.Type.STRING, description="Optional details")
                },
                required=["title"]
            )
        ),
        types.FunctionDeclaration(
            name="update_task",
            description="Update a task's title, description, or status",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.INTEGER, description="Task ID"),
                    "title": types.Schema(type=types.Type.STRING),
                    "description": types.Schema(type=types.Type.STRING),
                    "status": types.Schema(type=types.Type.STRING, enum=["pending", "in_progress", "done"])
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="delete_task",
            description="Delete a task permanently",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.INTEGER, description="Task ID")
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="mark_task_done",
            description="Mark a task as completed",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.INTEGER, description="Task ID")
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="ask_clarification",
            description="Ask the user a clarifying question before proceeding. Use when request is ambiguous or you need more context. Don't overuse.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "question": types.Schema(type=types.Type.STRING, description="The clarifying question to ask"),
                    "context": types.Schema(type=types.Type.STRING, description="Brief context for why you're asking")
                },
                required=["question"]
            )
        ),
        # Reads functions
        types.FunctionDeclaration(
            name="add_read",
            description="Add a paper or book to the reading list",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "title": types.Schema(type=types.Type.STRING, description="Paper/book title"),
                    "url": types.Schema(type=types.Type.STRING, description="URL to the paper/book"),
                    "author": types.Schema(type=types.Type.STRING, description="Author(s)"),
                    "notes": types.Schema(type=types.Type.STRING, description="Optional notes")
                },
                required=["title"]
            )
        ),
        types.FunctionDeclaration(
            name="update_read",
            description="Update a reading list item",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.INTEGER, description="Read ID"),
                    "title": types.Schema(type=types.Type.STRING),
                    "url": types.Schema(type=types.Type.STRING),
                    "author": types.Schema(type=types.Type.STRING),
                    "notes": types.Schema(type=types.Type.STRING),
                    "status": types.Schema(type=types.Type.STRING, enum=["unread", "reading", "read"])
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="delete_read",
            description="Remove a paper/book from the reading list",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.INTEGER, description="Read ID")
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="mark_read_done",
            description="Mark a paper/book as read",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "id": types.Schema(type=types.Type.INTEGER, description="Read ID")
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="search_arxiv",
            description="Search arxiv for a paper and get its URL. Use this to find paper URLs before adding to reading list.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "query": types.Schema(type=types.Type.STRING, description="Search query (paper title or keywords)")
                },
                required=["query"]
            )
        )
    ])]



def task_to_dict(task: RowLike) -> dict[str, Any]:
    """Convert task row to dict, handling both SQLite and PostgreSQL rows"""
    if isinstance(task, PgRowWrapper):
        return dict(task._data)
    return {k: task[k] for k in task.keys()}

def execute_function_call(func_call: dict[str, Any], conn: Any, user_email: str) -> dict[str, Any]:
    """Execute a Gemini function call and return result"""
    try:
        name = func_call.get('name')
        args = func_call.get('args', {})

        if not name:
            return {'type': 'error', 'message': 'No function name provided'}

        if name == 'add_task':
            title = args.get('title', '').strip()
            if not title:
                return {'type': 'error', 'message': 'Task title is required'}

            if USE_CLOUD_SQL:
                cursor = execute_query(conn,
                    'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?) RETURNING id',
                    (title, args.get('description', ''), user_email))
                task_id = cursor.fetchone()[0]
            else:
                cursor = execute_query(conn,
                    'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?)',
                    (title, args.get('description', ''), user_email))
                task_id = cursor.lastrowid
            conn.commit()
            cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
            new_task = fetchone(cursor)
            if new_task is None:
                return {'type': 'error', 'message': 'Failed to retrieve created task'}
            return {'type': 'added', 'task': task_to_dict(new_task)}

        elif name == 'update_task':
            task_id = args.get('id')
            if not task_id:
                return {'type': 'error', 'message': 'Task ID is required'}

            cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
            task = fetchone(cursor)
            if not task:
                return {'type': 'error', 'message': f'Task #{task_id} not found'}

            execute_query(conn,
                'UPDATE tasks SET title = ?, description = ?, status = ?, updated_at = ? WHERE id = ?',
                (args.get('title', task['title']), args.get('description', task['description']),
                 args.get('status', task['status']), datetime.now(), task_id))
            conn.commit()
            cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
            updated_task = fetchone(cursor)
            if updated_task is None:
                return {'type': 'error', 'message': 'Failed to retrieve updated task'}
            return {'type': 'updated', 'task': task_to_dict(updated_task)}

        elif name == 'delete_task':
            task_id = args.get('id')
            if not task_id:
                return {'type': 'error', 'message': 'Task ID is required'}

            cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
            task = fetchone(cursor)
            if not task:
                return {'type': 'error', 'message': f'Task #{task_id} not found'}

            task_dict = task_to_dict(task)
            execute_query(conn, 'DELETE FROM tasks WHERE id = ?', (task_id,))
            conn.commit()
            return {'type': 'deleted', 'task': task_dict}

        elif name == 'mark_task_done':
            task_id = args.get('id')
            if not task_id:
                return {'type': 'error', 'message': 'Task ID is required'}

            cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
            task = fetchone(cursor)
            if not task:
                return {'type': 'error', 'message': f'Task #{task_id} not found'}

            execute_query(conn, 'UPDATE tasks SET status = ?, updated_at = ? WHERE id = ?',
                          ('done', datetime.now(), task_id))
            conn.commit()
            cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
            done_task = fetchone(cursor)
            if done_task is None:
                return {'type': 'error', 'message': 'Failed to retrieve completed task'}
            return {'type': 'done', 'task': task_to_dict(done_task)}

        elif name == 'ask_clarification':
            # No DB action - clarification is in the AI's text response
            return {'type': 'clarification'}

        elif name == 'add_read':
            title = args.get('title', '').strip()
            if not title:
                return {'type': 'error', 'message': 'Read title is required'}

            if USE_CLOUD_SQL:
                cursor = execute_query(conn,
                    'INSERT INTO reads (title, url, author, notes, added_by) VALUES (?, ?, ?, ?, ?) RETURNING id',
                    (title, args.get('url', ''), args.get('author', ''), args.get('notes', ''), user_email))
                read_id = cursor.fetchone()[0]
            else:
                cursor = execute_query(conn,
                    'INSERT INTO reads (title, url, author, notes, added_by) VALUES (?, ?, ?, ?, ?)',
                    (title, args.get('url', ''), args.get('author', ''), args.get('notes', ''), user_email))
                read_id = cursor.lastrowid
            conn.commit()
            cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
            new_read = fetchone(cursor)
            if new_read is None:
                return {'type': 'error', 'message': 'Failed to retrieve created read'}
            return {'type': 'read_added', 'read': task_to_dict(new_read)}

        elif name == 'update_read':
            read_id = args.get('id')
            if not read_id:
                return {'type': 'error', 'message': 'Read ID is required'}

            cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
            read = fetchone(cursor)
            if not read:
                return {'type': 'error', 'message': f'Read #{read_id} not found'}

            execute_query(conn,
                'UPDATE reads SET title = ?, url = ?, author = ?, notes = ?, status = ?, updated_at = ? WHERE id = ?',
                (args.get('title', read['title']), args.get('url', read['url']),
                 args.get('author', read['author']), args.get('notes', read['notes']),
                 args.get('status', read['status']), datetime.now(), read_id))
            conn.commit()
            cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
            updated_read = fetchone(cursor)
            if updated_read is None:
                return {'type': 'error', 'message': 'Failed to retrieve updated read'}
            return {'type': 'read_updated', 'read': task_to_dict(updated_read)}

        elif name == 'delete_read':
            read_id = args.get('id')
            if not read_id:
                return {'type': 'error', 'message': 'Read ID is required'}

            cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
            read = fetchone(cursor)
            if not read:
                return {'type': 'error', 'message': f'Read #{read_id} not found'}

            read_dict = task_to_dict(read)
            execute_query(conn, 'DELETE FROM reads WHERE id = ?', (read_id,))
            conn.commit()
            return {'type': 'read_deleted', 'read': read_dict}

        elif name == 'mark_read_done':
            read_id = args.get('id')
            if not read_id:
                return {'type': 'error', 'message': 'Read ID is required'}

            cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
            read = fetchone(cursor)
            if not read:
                return {'type': 'error', 'message': f'Read #{read_id} not found'}

            execute_query(conn, 'UPDATE reads SET status = ?, updated_at = ? WHERE id = ?',
                          ('read', datetime.now(), read_id))
            conn.commit()
            cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
            done_read = fetchone(cursor)
            if done_read is None:
                return {'type': 'error', 'message': 'Failed to retrieve completed read'}
            return {'type': 'read_done', 'read': task_to_dict(done_read)}

        elif name == 'search_arxiv':
            query = args.get('query', '').strip()
            if not query:
                return {'type': 'error', 'message': 'Search query is required'}

            result = search_arxiv(query)
            if 'error' in result:
                return {'type': 'arxiv_result', 'result': result}
            return {'type': 'arxiv_result', 'result': result}

        return {'type': 'error', 'message': f'Unknown function: {name}'}

    except Exception as e:
        return {'type': 'error', 'message': f'Function execution failed: {str(e)}'}


def get_calendar_events() -> list[dict[str, Any]]:
    """Fetch events from Google Calendar for next 5 days"""
    if not GOOGLE_CALENDAR_AVAILABLE or service_account is None or build is None:
        return []

    creds_b64 = os.environ.get('GOOGLE_CALENDAR_CREDENTIALS')
    if not creds_b64:
        return []

    try:
        creds_json = json.loads(base64.b64decode(creds_b64))
        creds = service_account.Credentials.from_service_account_info(
            creds_json, scopes=['https://www.googleapis.com/auth/calendar.readonly']
        )
        service = build('calendar', 'v3', credentials=creds)  # type: ignore[misc]

        now = datetime.utcnow()
        time_min = now.isoformat() + 'Z'
        time_max = (now + timedelta(days=5)).isoformat() + 'Z'

        events = service.events().list(
            calendarId=os.environ.get('GOOGLE_CALENDAR_ID', 'primary'),
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        return [{
            'id': f"cal_{e['id'][:8]}",
            'title': e.get('summary', 'Untitled Event'),
            'description': e.get('description', ''),
            'status': 'EVENT',
            'event_start': e['start'].get('dateTime', e['start'].get('date')),
            'event_end': e['end'].get('dateTime', e['end'].get('date')),
            'assigned_by': 'calendar',
            'created_at': e.get('created', now.isoformat())
        } for e in events.get('items', [])]
    except Exception:
        # Log minimal info to avoid leaking credentials
        print("Calendar fetch failed")
        return []


def search_arxiv(query: str) -> dict[str, Any]:
    """Search arxiv for paper and return URL, title, authors"""
    try:
        search_url = f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}&max_results=1"
        with urllib.request.urlopen(search_url, timeout=10) as response:
            xml_data = response.read().decode('utf-8')

        # Parse XML response
        root = ET.fromstring(xml_data)
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entry = root.find('atom:entry', ns)
        if entry is None:
            return {'error': 'No results found'}

        title = entry.find('atom:title', ns)
        title_text = title.text.strip().replace('\n', ' ') if title is not None and title.text else 'Unknown'

        # Get arxiv URL (prefer abs link)
        url = ''
        for link in entry.findall('atom:link', ns):
            if link.get('type') == 'text/html':
                url = link.get('href', '')
                break
            if link.get('rel') == 'alternate':
                url = link.get('href', '')

        # Get authors
        authors = []
        for author in entry.findall('atom:author', ns):
            name = author.find('atom:name', ns)
            if name is not None and name.text:
                authors.append(name.text)

        return {
            'url': url,
            'title': title_text,
            'authors': ', '.join(authors[:3]) + ('...' if len(authors) > 3 else '')
        }
    except Exception as e:
        return {'error': str(e)}


class RowLike(Protocol):
    """Protocol for dict-like row access"""
    def __getitem__(self, key: str) -> Any: ...
    def keys(self) -> Any: ...


class PgRowWrapper:
    """Wrapper to make psycopg2 rows dict-accessible like sqlite3.Row"""
    def __init__(self, row: tuple[Any, ...], columns: list[str]) -> None:
        self._data: dict[str, Any] = dict(zip(columns, row))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def keys(self) -> Any:
        return self._data.keys()


def get_db() -> Any:
    if USE_CLOUD_SQL:
        import psycopg2  # pyright: ignore[reportMissingModuleSource]
        # Cloud Run provides Unix socket at /cloudsql/CONNECTION_NAME
        conn = psycopg2.connect(
            host=f'/cloudsql/{CLOUD_SQL_CONNECTION}',
            user=DB_USER,
            password=DB_PASS,
            dbname=DB_NAME,
        )
        return conn
    else:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        return conn


def execute_query(conn: Any, query: str, params: tuple[Any, ...] | None = None) -> Any:
    """Execute query with proper placeholder handling for both SQLite and PostgreSQL"""
    if USE_CLOUD_SQL:
        # Convert ? placeholders to %s for PostgreSQL
        query = query.replace('?', '%s')
    cursor = conn.cursor()
    if params:
        cursor.execute(query, params)
    else:
        cursor.execute(query)
    return cursor


def fetchone(cursor: Any) -> RowLike | None:
    """Fetch one row with proper dict-like access"""
    row = cursor.fetchone()
    if row is None:
        return None
    if USE_CLOUD_SQL:
        columns: list[str] = [desc[0] for desc in cursor.description]
        return PgRowWrapper(row, columns)
    return row  # type: ignore[return-value]


def fetchall(cursor: Any) -> list[RowLike]:
    """Fetch all rows with proper dict-like access"""
    rows = cursor.fetchall()
    if USE_CLOUD_SQL:
        columns: list[str] = [desc[0] for desc in cursor.description]
        return [PgRowWrapper(row, columns) for row in rows]
    return rows  # type: ignore[return-value]


def init_db():
    conn = get_db()
    if USE_CLOUD_SQL:
        cursor = conn.cursor()

        # Create each table separately so one failure doesn't affect others
        tables = [
            ('users', '''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    is_admin INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''),
            ('magic_links', '''
                CREATE TABLE IF NOT EXISTS magic_links (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    used INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''),
            ('tasks', '''
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    assigned_by TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''),
            ('chat_history', '''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    user_email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''),
            ('reads', '''
                CREATE TABLE IF NOT EXISTS reads (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    url TEXT,
                    author TEXT,
                    notes TEXT,
                    status TEXT DEFAULT 'unread',
                    added_by TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''),
        ]

        for table_name, create_sql in tables:
            try:
                cursor.execute(create_sql)
                conn.commit()
            except Exception as e:
                print(f"Table {table_name}: {e}")
                conn.rollback()

        # Create index separately
        try:
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_history_user_email
                ON chat_history(user_email)
            ''')
            conn.commit()
        except Exception:
            conn.rollback()

        conn.close()
    else:
        # SQLite path - executescript is available on sqlite3.Connection
        conn.executescript(  # type: ignore[union-attr]
            '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                is_admin INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS magic_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                used INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                assigned_by TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                user_email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_chat_history_user_email
            ON chat_history(user_email);

            CREATE TABLE IF NOT EXISTS reads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                url TEXT,
                author TEXT,
                notes TEXT,
                status TEXT DEFAULT 'unread',
                added_by TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')
        conn.commit()
        conn.close()


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def send_magic_link(email, token):
    link = f"{DOMAIN}/auth/{token}"

    msg = MIMEMultipart()
    msg['From'] = FROM_EMAIL
    msg['To'] = email
    msg['Subject'] = 'Your login link for Make Arjo Work'

    body = f'''
    Click here to log in: {link}

    This link expires in 15 minutes.

    If you didn't request this, you can ignore this email.
    '''
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(FROM_EMAIL, email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"Email error: {e}")
        # In development, just print the link
        print(f"\n>>> MAGIC LINK: {link}\n", flush=True)
        return True  # Return True in dev mode so testing works


@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').lower().strip()
        if not email:
            return render_template('login.html', error='Email is required')

        if not email.endswith('@fydy.ai'):
            return render_template('login.html', error='Only @fydy.ai emails are allowed')

        # Create magic link
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(minutes=15)

        conn = get_db()
        execute_query(conn,
            'INSERT INTO magic_links (email, token, expires_at) VALUES (?, ?, ?)',
            (email, token, expires_at)
        )
        conn.commit()
        conn.close()

        send_magic_link(email, token)
        return render_template('login.html', success=True, email=email)

    return render_template('login.html')


@app.route('/auth/<token>')
def authenticate(token):
    conn = get_db()
    cursor = execute_query(conn,
        'SELECT * FROM magic_links WHERE token = ? AND used = 0 AND expires_at > ?',
        (token, datetime.now())
    )
    link = fetchone(cursor)

    if not link:
        conn.close()
        return render_template('login.html', error='Invalid or expired link')

    # Mark link as used
    execute_query(conn, 'UPDATE magic_links SET used = 1 WHERE id = ?', (link['id'],))

    # Get or create user
    cursor = execute_query(conn, 'SELECT * FROM users WHERE email = ?', (link['email'],))
    user = fetchone(cursor)
    if not user:
        if USE_CLOUD_SQL:
            cursor = execute_query(conn, 'INSERT INTO users (email) VALUES (?) RETURNING id', (link['email'],))
            row = cursor.fetchone()
            user_id = row[0]
        else:
            cursor = execute_query(conn, 'INSERT INTO users (email) VALUES (?)', (link['email'],))
            user_id = cursor.lastrowid
    else:
        user_id = user['id']

    conn.commit()
    conn.close()

    session.permanent = True  # Stay logged in for 30 days
    session['user_id'] = user_id
    session['email'] = link['email']

    return redirect(url_for('dashboard'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    cursor = execute_query(conn, 'SELECT * FROM users WHERE id = ?', (session['user_id'],))
    user = fetchone(cursor)
    conn.close()
    return render_template('dashboard.html', user=user)


@app.route('/api/tasks', methods=['GET'])
@login_required
def get_tasks():
    status = request.args.get('status')
    conn = get_db()

    if status and status != 'pending':
        cursor = execute_query(conn,
            'SELECT * FROM tasks WHERE status = ? ORDER BY created_at DESC',
            (status,))
    else:
        cursor = execute_query(conn, 'SELECT * FROM tasks ORDER BY created_at DESC')

    tasks = [task_to_dict(t) for t in fetchall(cursor)]
    conn.close()

    # Include calendar events for 'all' or 'pending' filter
    if not status or status == 'pending':
        events = get_calendar_events()
        tasks = events + tasks  # Events first

    return jsonify(tasks)


@app.route('/api/tasks', methods=['POST'])
@login_required
def create_task():
    data = request.json
    title = data.get('title')
    description = data.get('description', '')
    assigned_by = session.get('email', 'Unknown')

    if not title:
        return jsonify({'error': 'Title is required'}), 400

    conn = get_db()
    if USE_CLOUD_SQL:
        cursor = execute_query(conn,
            'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?) RETURNING id',
            (title, description, assigned_by)
        )
        task_id = cursor.fetchone()[0]
    else:
        cursor = execute_query(conn,
            'INSERT INTO tasks (title, description, assigned_by) VALUES (?, ?, ?)',
            (title, description, assigned_by)
        )
        task_id = cursor.lastrowid
    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
    task = fetchone(cursor)
    conn.commit()
    conn.close()

    if task is None:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task_to_dict(task)), 201


@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
@login_required
def update_task(task_id):
    data = request.json

    conn = get_db()
    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
    task = fetchone(cursor)

    if not task:
        conn.close()
        return jsonify({'error': 'Task not found'}), 404

    title = data.get('title', task['title'])
    description = data.get('description', task['description'])
    status = data.get('status', task['status'])

    execute_query(conn,
        'UPDATE tasks SET title = ?, description = ?, status = ?, updated_at = ? WHERE id = ?',
        (title, description, status, datetime.now(), task_id)
    )
    conn.commit()

    cursor = execute_query(conn, 'SELECT * FROM tasks WHERE id = ?', (task_id,))
    updated_task = fetchone(cursor)
    conn.close()

    if updated_task is None:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task_to_dict(updated_task))


@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@login_required
def delete_task(task_id):
    conn = get_db()
    execute_query(conn, 'DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    return '', 204


# Reads API endpoints
@app.route('/api/reads', methods=['GET'])
@login_required
def get_reads():
    status = request.args.get('status')
    conn = get_db()

    try:
        if status:
            cursor = execute_query(conn,
                'SELECT * FROM reads WHERE status = ? ORDER BY created_at DESC',
                (status,))
        else:
            cursor = execute_query(conn, 'SELECT * FROM reads ORDER BY created_at DESC')

        reads = [task_to_dict(r) for r in fetchall(cursor)]
    except Exception:
        reads = []  # Table may not exist yet
    conn.close()
    return jsonify(reads)


@app.route('/api/reads', methods=['POST'])
@login_required
def create_read():
    data = request.json
    title = data.get('title')
    url = data.get('url', '')
    author = data.get('author', '')
    notes = data.get('notes', '')
    added_by = session.get('email', 'Unknown')

    if not title:
        return jsonify({'error': 'Title is required'}), 400

    conn = get_db()
    if USE_CLOUD_SQL:
        cursor = execute_query(conn,
            'INSERT INTO reads (title, url, author, notes, added_by) VALUES (?, ?, ?, ?, ?) RETURNING id',
            (title, url, author, notes, added_by)
        )
        read_id = cursor.fetchone()[0]
    else:
        cursor = execute_query(conn,
            'INSERT INTO reads (title, url, author, notes, added_by) VALUES (?, ?, ?, ?, ?)',
            (title, url, author, notes, added_by)
        )
        read_id = cursor.lastrowid
    cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
    read = fetchone(cursor)
    conn.commit()
    conn.close()

    if read is None:
        return jsonify({'error': 'Read not found'}), 404
    return jsonify(task_to_dict(read)), 201


@app.route('/api/reads/<int:read_id>', methods=['PUT'])
@login_required
def update_read(read_id):
    data = request.json

    conn = get_db()
    cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
    read = fetchone(cursor)

    if not read:
        conn.close()
        return jsonify({'error': 'Read not found'}), 404

    title = data.get('title', read['title'])
    url = data.get('url', read['url'])
    author = data.get('author', read['author'])
    notes = data.get('notes', read['notes'])
    status = data.get('status', read['status'])

    execute_query(conn,
        'UPDATE reads SET title = ?, url = ?, author = ?, notes = ?, status = ?, updated_at = ? WHERE id = ?',
        (title, url, author, notes, status, datetime.now(), read_id)
    )
    conn.commit()

    cursor = execute_query(conn, 'SELECT * FROM reads WHERE id = ?', (read_id,))
    updated_read = fetchone(cursor)
    conn.close()

    if updated_read is None:
        return jsonify({'error': 'Read not found'}), 404
    return jsonify(task_to_dict(updated_read))


@app.route('/api/reads/<int:read_id>', methods=['DELETE'])
@login_required
def delete_read(read_id):
    conn = get_db()
    execute_query(conn, 'DELETE FROM reads WHERE id = ?', (read_id,))
    conn.commit()
    conn.close()
    return '', 204


# Debug endpoint - check reads table (remove after debugging)
@app.route('/api/debug/reads-status', methods=['GET'])
def debug_reads_status():
    conn = get_db()
    try:
        cursor = execute_query(conn, 'SELECT COUNT(*) as count FROM reads')
        row = cursor.fetchone()
        count = row[0] if row else 0

        cursor = execute_query(conn, 'SELECT id, title, status FROM reads ORDER BY id DESC LIMIT 5')
        reads = []
        for r in cursor.fetchall():
            reads.append({'id': r[0], 'title': r[1], 'status': r[2]})

        return jsonify({'count': count, 'recent': reads})
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        conn.close()


# Internal maintenance endpoint - requires secret token
@app.route('/api/internal/clear-all-chats/<token>', methods=['POST'])
def internal_clear_all_chats(token):
    """Clear all chat history - requires MAINTENANCE_TOKEN env var"""
    expected = os.environ.get('MAINTENANCE_TOKEN', '')
    if not expected or token != expected:
        return '', 404  # Pretend it doesn't exist
    conn = get_db()
    execute_query(conn, 'DELETE FROM chat_history')
    conn.commit()
    conn.close()
    return jsonify({'cleared': True})


# Get per-user chat history
# Note: user_email now stores the conversation owner (the user), not the sender role
# This allows filtering conversations per user while keeping role in 'role' column
@app.route('/api/chat/history', methods=['GET'])
@login_required
def get_chat_history():
    user_email = session.get('email')
    conn = get_db()
    cursor = execute_query(conn,
        'SELECT role, content, user_email, created_at FROM chat_history WHERE user_email = ? ORDER BY id ASC LIMIT 100',
        (user_email,)
    )
    messages = fetchall(cursor)
    conn.close()
    return jsonify([task_to_dict(m) for m in messages])


# Clear current user's chat history only
@app.route('/api/chat/clear', methods=['POST'])
@login_required
def clear_chat():
    user_email = session.get('email')
    conn = get_db()

    # Clear chat history (starts fresh conversation)
    execute_query(conn, 'DELETE FROM chat_history WHERE user_email = ?', (user_email,))

    conn.commit()
    conn.close()
    return jsonify({'success': True})




# Chat with Gemini AI using Chats API for conversation history
@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    data = request.json or {}
    user_message: str = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'Message required'}), 400

    user_email: str = session.get('email') or 'Unknown'
    conn = get_db()

    # Get current tasks (dynamic context - changes frequently)
    cursor = execute_query(conn, 'SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20')
    tasks = fetchall(cursor)
    tasks_context = "\n".join([
        f"- [#{t['id']}] [{t['status']}] {t['title']}"
        for t in tasks
    ]) if tasks else "No tasks yet."

    # Get current reads (graceful fallback if table doesn't exist yet)
    reads_context = ""
    try:
        cursor = execute_query(conn, 'SELECT * FROM reads ORDER BY created_at DESC LIMIT 20')
        reads = fetchall(cursor)
        reads_context = "\n".join([
            f"- [#{r['id']}] [{r['status']}] {r['title']}" + (f" ({r['url']})" if r['url'] else "")
            for r in reads
        ]) if reads else ""
    except Exception:
        pass  # Table may not exist yet on first deploy

    # Get user's chat history for conversation continuity
    cursor = execute_query(conn,
        'SELECT role, content FROM chat_history WHERE user_email = ? ORDER BY id ASC LIMIT 20',
        (user_email,))
    history_rows = fetchall(cursor)

    # Convert to Gemini Content format
    gemini_history = []
    for row in history_rows:
        role = "user" if row['role'] == 'user' else "model"
        gemini_history.append(types.Content(
            role=role,
            parts=[types.Part(text=row['content'])]
        ))

    try:
        # Build full system prompt with persona + current tasks + reads
        system_prompt = build_system_prompt(tasks_context, reads_context)

        # Create chat with history and tools
        gemini_chat = gemini_client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=get_task_tools(),
                temperature=0.7,
                max_output_tokens=2048,
            ),
            history=gemini_history if gemini_history else None,
        )

        # Send the new message and handle function calling loop
        response = gemini_chat.send_message(user_message)

        ai_response = ""
        actions_performed = []
        action_descriptions = []

        # Loop to handle chained function calls (max 5 iterations for safety)
        for _ in range(5):
            function_calls: list[dict[str, Any]] = []
            function_responses = []

            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if hasattr(part, 'text') and part.text:
                                ai_response += part.text
                            if hasattr(part, 'function_call') and part.function_call:
                                function_calls.append({
                                    "name": part.function_call.name,
                                    "args": dict(part.function_call.args) if part.function_call.args else {}
                                })

            # If no function calls, we're done
            if not function_calls:
                break

            # Execute function calls and collect responses
            for func_call in function_calls:
                func_name = func_call.get('name', '')

                # Handle ask_clarification specially
                if func_name == 'ask_clarification':
                    question = func_call.get('args', {}).get('question', '')
                    if question and not ai_response:
                        ai_response = question
                    actions_performed.append({'type': 'clarification'})
                    function_responses.append(types.Part.from_function_response(
                        name=func_name,
                        response={'status': 'asked'}
                    ))
                    continue

                result = execute_function_call(func_call, conn, user_email)
                if result.get('type') != 'error':
                    actions_performed.append(result)

                # Send function result back to model
                function_responses.append(types.Part.from_function_response(
                    name=func_name,
                    response=result
                ))

            # If we have function responses, send them back to get next response
            if function_responses:
                response = gemini_chat.send_message(function_responses)
            else:
                break

        # Build action descriptions for history
        for action in actions_performed:
            action_type = action.get('type')
            if action_type == 'added':
                action_descriptions.append(f"Added task: {action['task']['title']}")
            elif action_type == 'updated':
                action_descriptions.append(f"Updated task #{action['task']['id']}")
            elif action_type == 'deleted':
                action_descriptions.append(f"Deleted task: {action['task']['title']}")
            elif action_type == 'done':
                action_descriptions.append(f"Completed task: {action['task']['title']}")
            elif action_type == 'read_added':
                action_descriptions.append(f"Added to reading list: {action['read']['title']}")
            elif action_type == 'read_updated':
                action_descriptions.append(f"Updated read #{action['read']['id']}")
            elif action_type == 'read_deleted':
                action_descriptions.append(f"Removed from reading list: {action['read']['title']}")
            elif action_type == 'read_done':
                action_descriptions.append(f"Marked as read: {action['read']['title']}")
            elif action_type == 'arxiv_result':
                if 'error' not in action.get('result', {}):
                    action_descriptions.append(f"Found on arxiv: {action['result'].get('title', 'Unknown')}")

        # Build response for history - ALWAYS include action descriptions so AI remembers what it did
        history_response = ai_response.strip()
        if action_descriptions:
            action_summary = "[" + ", ".join(action_descriptions) + "]"
            if history_response:
                history_response = history_response + "\n" + action_summary
            else:
                history_response = action_summary

        # Save to local history (for display in UI and future context)
        execute_query(conn,
            'INSERT INTO chat_history (role, content, user_email) VALUES (?, ?, ?)',
            ('user', user_message, user_email))
        if history_response:
            execute_query(conn,
                'INSERT INTO chat_history (role, content, user_email) VALUES (?, ?, ?)',
                ('assistant', history_response, user_email))

        conn.commit()
        conn.close()

        return jsonify({
            "response": ai_response,
            "actions": actions_performed
        })

    except Exception as e:
        conn.close()
        return jsonify({"error": str(e)}), 500


# Initialize database tables (runs on import for gunicorn compatibility)
init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
