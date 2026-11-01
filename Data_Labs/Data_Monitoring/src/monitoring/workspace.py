"""
Where monitoring results go: the Evidently workspace that the dashboard shows.

Inside docker-compose, EVIDENTLY_URL points at the dashboard service (http://ui:8000),
so every report is sent through its API and shows up on the dashboard immediately.
Without EVIDENTLY_URL (for example in the tests), reports go to a local folder instead.
"""
import os

from evidently.ui.workspace import RemoteWorkspace, Workspace

from monitoring.data import LAB_DIR

LOCAL_WORKSPACE_DIR = LAB_DIR / "workspace"


def open_workspace(url=None, path=LOCAL_WORKSPACE_DIR):
    """Connects to the dashboard service if a URL is given or set in EVIDENTLY_URL, else opens a local folder."""
    url = url or os.environ.get("EVIDENTLY_URL")
    if not url:
        return Workspace(str(path))
    try:
        return RemoteWorkspace(url)
    except OSError as error:  # what the connection errors of requests (used by Evidently) are
        raise RuntimeError(f"Can't reach the Evidently dashboard service at {url}. "
                           "Start it with: docker compose up -d") from error


def project_url(project):
    """The project's page in the browser, if the dashboard's address is set in DASHBOARD_URL."""
    base = os.environ.get("DASHBOARD_URL")
    return f"{base}/projects/{project.id}" if base else None


def get_or_create_project(workspace, name, description=""):
    """Returns the project with this name, creating it the first time, so reruns don't duplicate it."""
    existing = workspace.search_project(name)
    if existing:
        return existing[0]
    project = workspace.create_project(name, description=description)
    return project
