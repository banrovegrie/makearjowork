"""End-to-end integration tests for Make Arjo Work."""

import pytest
import os

os.environ['DATABASE'] = ':memory:'

from app import app, init_db, get_db, execute_query, add_link


@pytest.fixture
def client():
    """Create test client with fresh database."""
    app.config['TESTING'] = True
    app.config['SECRET_KEY'] = 'test-secret'

    with app.test_client() as client:
        with app.app_context():
            conn = get_db()
            conn.executescript('''
                DROP TABLE IF EXISTS reads;
                DROP TABLE IF EXISTS tasks;
                DROP TABLE IF EXISTS users;
                DROP TABLE IF EXISTS chat_history;
                DROP TABLE IF EXISTS magic_links;
            ''')
            conn.commit()
            conn.close()
            init_db()
        yield client


@pytest.fixture
def auth_client(client):
    """Create authenticated test client."""
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['email'] = 'test@fydy.ai'

    conn = get_db()
    execute_query(conn, 'INSERT INTO users (id, email) VALUES (?, ?)', (1, 'test@fydy.ai'))
    conn.commit()
    conn.close()

    return client


class TestFullWorkflow:
    """Test complete user workflows end-to-end."""

    def test_tasks_workflow(self, auth_client):
        """Test complete task CRUD workflow."""
        # Create task
        resp = auth_client.post('/api/tasks',
            json={'title': 'Test task', 'description': 'Test description'},
            content_type='application/json')
        assert resp.status_code == 201
        task_id = resp.json['id']

        # Read task
        resp = auth_client.get('/api/tasks')
        assert resp.status_code == 200
        assert len(resp.json) >= 1
        assert any(t['id'] == task_id for t in resp.json)

        # Update task
        resp = auth_client.put(f'/api/tasks/{task_id}',
            json={'status': 'done'},
            content_type='application/json')
        assert resp.status_code == 200
        assert resp.json['status'] == 'done'

        # Delete task
        resp = auth_client.delete(f'/api/tasks/{task_id}')
        assert resp.status_code == 204

    def test_reads_workflow(self, auth_client):
        """Test complete reads CRUD workflow."""
        # Create read
        resp = auth_client.post('/api/reads',
            json={'title': 'Attention Is All You Need', 'url': 'https://arxiv.org/abs/1706.03762'},
            content_type='application/json')
        assert resp.status_code == 201
        read_id = resp.json['id']
        assert resp.json['status'] == 'unread'

        # Read all
        resp = auth_client.get('/api/reads')
        assert resp.status_code == 200
        assert len(resp.json) >= 1

        # Update status
        resp = auth_client.put(f'/api/reads/{read_id}',
            json={'status': 'reading'},
            content_type='application/json')
        assert resp.status_code == 200
        assert resp.json['status'] == 'reading'

        # Mark as read
        resp = auth_client.put(f'/api/reads/{read_id}',
            json={'status': 'read'},
            content_type='application/json')
        assert resp.status_code == 200
        assert resp.json['status'] == 'read'

        # Delete
        resp = auth_client.delete(f'/api/reads/{read_id}')
        assert resp.status_code == 204

    def test_chat_history_workflow(self, auth_client):
        """Test chat history persistence."""
        # Initial history should be accessible
        resp = auth_client.get('/api/chat/history')
        assert resp.status_code == 200

        # Clear should work
        resp = auth_client.post('/api/chat/clear')
        assert resp.status_code == 200

        resp = auth_client.get('/api/chat/history')
        assert resp.status_code == 200
        assert len(resp.json) == 0


class TestAddLinkIntegration:
    """Test add_link integration."""

    def test_add_link_paper(self):
        """Test add_link for a paper."""
        result = add_link('A Path to Autonomous Machine Intelligence Yann LeCun')

        assert 'url' in result
        assert 'title' in result

    def test_add_link_always_returns_url(self):
        """add_link always returns a URL (fallback to Google search)."""
        result = add_link('some obscure query that wont match anything')

        assert 'url' in result
        assert 'title' in result


class TestConcurrency:
    """Test concurrent request handling."""

    def test_multiple_rapid_requests(self, auth_client):
        """Test handling multiple rapid requests."""
        # Create multiple reads rapidly
        for i in range(5):
            resp = auth_client.post('/api/reads',
                json={'title': f'Paper {i}'},
                content_type='application/json')
            assert resp.status_code == 201

        # Verify all were created
        resp = auth_client.get('/api/reads')
        assert resp.status_code == 200
        assert len(resp.json) >= 5

    def test_multiple_rapid_tasks(self, auth_client):
        """Test handling multiple rapid task requests."""
        for i in range(5):
            resp = auth_client.post('/api/tasks',
                json={'title': f'Task {i}'},
                content_type='application/json')
            assert resp.status_code == 201

        resp = auth_client.get('/api/tasks')
        assert resp.status_code == 200
        assert len(resp.json) >= 5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_title_rejected(self, auth_client):
        """Empty titles should be rejected."""
        resp = auth_client.post('/api/reads',
            json={'title': ''},
            content_type='application/json')
        assert resp.status_code == 400

        resp = auth_client.post('/api/tasks',
            json={'title': ''},
            content_type='application/json')
        assert resp.status_code == 400

    def test_nonexistent_resource(self, auth_client):
        """Accessing nonexistent resources returns 404."""
        resp = auth_client.get('/api/reads/99999')
        # GET on specific read isn't implemented, so test PUT
        resp = auth_client.put('/api/reads/99999',
            json={'title': 'test'},
            content_type='application/json')
        assert resp.status_code == 404

        resp = auth_client.put('/api/tasks/99999',
            json={'title': 'test'},
            content_type='application/json')
        assert resp.status_code == 404

    def test_special_characters_in_title(self, auth_client):
        """Titles with special characters should work."""
        special_title = "Test with émojis 🎉 and <html> & 'quotes'"
        resp = auth_client.post('/api/reads',
            json={'title': special_title},
            content_type='application/json')
        assert resp.status_code == 201
        assert resp.json['title'] == special_title

    def test_very_long_title(self, auth_client):
        """Very long titles should work."""
        long_title = 'A' * 1000
        resp = auth_client.post('/api/reads',
            json={'title': long_title},
            content_type='application/json')
        assert resp.status_code == 201
        assert resp.json['title'] == long_title


class TestDashboard:
    """Test dashboard rendering."""

    def test_dashboard_loads(self, auth_client):
        """Dashboard should load for authenticated users."""
        resp = auth_client.get('/dashboard')
        assert resp.status_code == 200
        assert b'Make Arjo Work' in resp.data

    def test_dashboard_has_theme_toggle(self, auth_client):
        """Dashboard should have theme toggle."""
        resp = auth_client.get('/dashboard')
        assert b'theme-toggle' in resp.data

    def test_dashboard_has_panel_tabs(self, auth_client):
        """Dashboard should have Tasks/Reads tabs."""
        resp = auth_client.get('/dashboard')
        assert b'panel-tab' in resp.data
        assert b'data-panel="tasks"' in resp.data
        assert b'data-panel="reads"' in resp.data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
