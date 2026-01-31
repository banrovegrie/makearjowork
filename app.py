import os
import re
import sqlite3
import secrets
import smtplib
import json
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from functools import wraps
from flask import Flask, request, jsonify, render_template, redirect, session, url_for, g
from werkzeug.security import generate_password_hash
import hashlib
import base64
from dotenv import load_dotenv

# Google GenAI imports for Gemini Chats API
from google import genai
from google.genai import types

# Google Calendar imports (optional - graceful fallback if not installed)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False

load_dotenv()

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


def build_system_prompt(tasks_context):
    """Build full system prompt with persona and current tasks"""
    persona = load_persona()
    persona_context = build_persona_context(persona)
    name = persona.get('identity', {}).get('name', 'Arjo')

    return f"""You ARE {name}. First person always.

## Who I Am
{persona_context}

## Current Tasks
{tasks_context}

## What I Can Do
- add_task: Accept new work
- update_task: Change title/description/status
- delete_task: Remove a task
- mark_task_done: Complete a task
- ask_clarification: Ask when something is unclear

## Rules
- Only act on the current user message
- Previous messages are context only - don't re-execute
- Be direct and concise (CLI interface)
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
                type="OBJECT",
                properties={
                    "title": types.Schema(type="STRING", description="Task title"),
                    "description": types.Schema(type="STRING", description="Optional details")
                },
                required=["title"]
            )
        ),
        types.FunctionDeclaration(
            name="update_task",
            description="Update a task's title, description, or status",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "id": types.Schema(type="INTEGER", description="Task ID"),
                    "title": types.Schema(type="STRING"),
                    "description": types.Schema(type="STRING"),
                    "status": types.Schema(type="STRING", enum=["pending", "in_progress", "done"])
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="delete_task",
            description="Delete a task permanently",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "id": types.Schema(type="INTEGER", description="Task ID")
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="mark_task_done",
            description="Mark a task as completed",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "id": types.Schema(type="INTEGER", description="Task ID")
                },
                required=["id"]
            )
        ),
        types.FunctionDeclaration(
            name="ask_clarification",
            description="Ask the user a clarifying question before proceeding. Use when request is ambiguous or you need more context. Don't overuse.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "question": types.Schema(type="STRING", description="The clarifying question to ask"),
                    "context": types.Schema(type="STRING", description="Brief context for why you're asking")
                },
                required=["question"]
            )
        )
    ])]



def task_to_dict(task):
    """Convert task row to dict, handling both SQLite and PostgreSQL rows"""
    return dict(task._data) if hasattr(task, '_data') else dict(task)

def execute_function_call(func_call, conn, user_email):
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
            return {'type': 'added', 'task': task_to_dict(fetchone(cursor))}

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
            return {'type': 'updated', 'task': task_to_dict(fetchone(cursor))}

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
            return {'type': 'done', 'task': task_to_dict(fetchone(cursor))}

        elif name == 'ask_clarification':
            # No DB action - clarification is in the AI's text response
            return {'type': 'clarification'}

        return {'type': 'error', 'message': f'Unknown function: {name}'}

    except Exception as e:
        return {'type': 'error', 'message': f'Function execution failed: {str(e)}'}


def get_calendar_events():
    """Fetch events from Google Calendar for next 5 days"""
    if not GOOGLE_CALENDAR_AVAILABLE:
        return []

    creds_b64 = os.environ.get('GOOGLE_CALENDAR_CREDENTIALS')
    if not creds_b64:
        return []

    try:
        creds_json = json.loads(base64.b64decode(creds_b64))
        creds = service_account.Credentials.from_service_account_info(
            creds_json, scopes=['https://www.googleapis.com/auth/calendar.readonly']
        )
        service = build('calendar', 'v3', credentials=creds)

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


class PgRowWrapper:
    """Wrapper to make psycopg2 rows dict-accessible like sqlite3.Row"""
    def __init__(self, row, columns):
        self._data = dict(zip(columns, row))

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()


def get_db():
    if USE_CLOUD_SQL:
        import psycopg2
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


def execute_query(conn, query, params=None):
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


def fetchone(cursor):
    """Fetch one row with proper dict-like access"""
    row = cursor.fetchone()
    if row is None:
        return None
    if USE_CLOUD_SQL:
        columns = [desc[0] for desc in cursor.description]
        return PgRowWrapper(row, columns)
    return row


def fetchall(cursor):
    """Fetch all rows with proper dict-like access"""
    rows = cursor.fetchall()
    if USE_CLOUD_SQL:
        columns = [desc[0] for desc in cursor.description]
        return [PgRowWrapper(row, columns) for row in rows]
    return rows


def init_db():
    conn = get_db()
    if USE_CLOUD_SQL:
        cursor = conn.cursor()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    email TEXT UNIQUE NOT NULL,
                    is_admin INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS magic_links (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL,
                    token TEXT UNIQUE NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    used INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    description TEXT,
                    assigned_by TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS chat_history (
                    id SERIAL PRIMARY KEY,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    user_email TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create index for per-user chat history queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_chat_history_user_email
                ON chat_history(user_email)
            ''')
            conn.commit()
        except Exception as e:
            # Tables already exist or concurrent init - safe to ignore
            conn.rollback()
        finally:
            conn.close()
    else:
        conn.executescript('''
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


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        conn = get_db()
        cursor = execute_query(conn, 'SELECT is_admin FROM users WHERE id = ?', (session['user_id'],))
        user = fetchone(cursor)
        conn.close()
        if not user or not user['is_admin']:
            return jsonify({'error': 'Admin access required'}), 403
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

    return jsonify(dict(task._data) if hasattr(task, '_data') else dict(task)), 201


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

    return jsonify(dict(updated_task._data) if hasattr(updated_task, '_data') else dict(updated_task))


@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
@login_required
def delete_task(task_id):
    conn = get_db()
    execute_query(conn, 'DELETE FROM tasks WHERE id = ?', (task_id,))
    conn.commit()
    conn.close()
    return '', 204


@app.route('/api/users/make-admin', methods=['POST'])
@admin_required
def make_admin():
    data = request.json
    email = data.get('email', '').lower().strip()

    conn = get_db()
    execute_query(conn, 'UPDATE users SET is_admin = 1 WHERE email = ?', (email,))
    conn.commit()
    conn.close()

    return jsonify({'success': True})


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
    return jsonify([dict(m._data) if hasattr(m, '_data') else dict(m) for m in messages])


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
    data = request.json
    user_message = data.get('message', '')

    if not user_message:
        return jsonify({'error': 'Message required'}), 400

    user_email = session.get('email', 'Unknown')
    conn = get_db()

    # Get current tasks (dynamic context - changes frequently)
    cursor = execute_query(conn, 'SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20')
    tasks = fetchall(cursor)
    tasks_context = "\n".join([
        f"- [#{t['id']}] [{t['status']}] {t['title']}"
        for t in tasks
    ]) if tasks else "No tasks yet."

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
        # Build full system prompt with persona + current tasks
        system_prompt = build_system_prompt(tasks_context)

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

        # Send the new message
        response = gemini_chat.send_message(user_message)

        # Extract text and function calls from response
        ai_response = ""
        function_calls = []

        for candidate in response.candidates:
            for part in candidate.content.parts:
                if hasattr(part, 'text') and part.text:
                    ai_response += part.text
                if hasattr(part, 'function_call') and part.function_call:
                    function_calls.append({
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args)
                    })

        # Execute all function calls
        actions_performed = []
        action_descriptions = []
        for func_call in function_calls:
            # Handle ask_clarification specially - use question as response
            if func_call.get('name') == 'ask_clarification':
                question = func_call.get('args', {}).get('question', '')
                if question and not ai_response:
                    ai_response = question
                actions_performed.append({'type': 'clarification'})
                continue

            result = execute_function_call(func_call, conn, user_email)
            if result.get('type') != 'error':
                actions_performed.append(result)
                # Build description for history
                if result.get('type') == 'added':
                    action_descriptions.append(f"Added task: {result['task']['title']}")
                elif result.get('type') == 'updated':
                    action_descriptions.append(f"Updated task #{result['task']['id']}")
                elif result.get('type') == 'deleted':
                    action_descriptions.append(f"Deleted task: {result['task']['title']}")
                elif result.get('type') == 'done':
                    action_descriptions.append(f"Completed task: {result['task']['title']}")

        # Build response for history - include action descriptions if no text
        history_response = ai_response.strip()
        if action_descriptions and not history_response:
            history_response = "[" + ", ".join(action_descriptions) + "]"

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


# CLI command to make first admin
@app.cli.command('make-admin')
def make_first_admin():
    import sys
    if len(sys.argv) < 2:
        print("Usage: flask make-admin <email>")
        return
    email = sys.argv[1].lower().strip()
    conn = get_db()
    if USE_CLOUD_SQL:
        execute_query(conn, 'INSERT INTO users (email, is_admin) VALUES (?, 1) ON CONFLICT (email) DO UPDATE SET is_admin = 1', (email,))
    else:
        execute_query(conn, 'INSERT OR IGNORE INTO users (email, is_admin) VALUES (?, 1)', (email,))
        execute_query(conn, 'UPDATE users SET is_admin = 1 WHERE email = ?', (email,))
    conn.commit()
    conn.close()
    print(f"Made {email} an admin")


# Initialize database tables (runs on import for gunicorn compatibility)
init_db()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
