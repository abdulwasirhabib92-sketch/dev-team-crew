"""
Supabase integration tool for CrewAI agents.
Allows agents to query, insert, update, and delete records in Supabase.
"""
import os
import json
import requests
from typing import Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class SupabaseQueryTool(BaseTool):
    """Tool for querying (SELECT) data from Supabase tables."""
    name: str = "supabase_query"
    description: str = (
        "Query data from a Supabase table. "
        "Input should be a JSON string with 'table' (required), "
        "'columns' (optional, default '*'), 'limit' (optional, default 10), "
        "'filter_column' and 'filter_value' (optional for filtering). "
        "Example: {\"table\": \"users\", \"columns\": \"id,name,email\", \"limit\": 5}"
    )

    def _run(self, query: str) -> str:
        try:
            params = json.loads(query) if isinstance(query, str) else query
        except json.JSONDecodeError:
            return f"Error: Invalid JSON input. Expected JSON like: {{\"table\": \"users\", \"limit\": 5}}"

        table = params.get("table")
        if not table:
            return "Error: 'table' is required."

        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/{table}"
        headers = {
            "apikey": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_ROLE_KEY')}",
        }

        query_params = {
            "select": params.get("columns", "*"),
        }
        if params.get("limit"):
            query_params["limit"] = params["limit"]

        if params.get("filter_column") and params.get("filter_value"):
            query_params[params["filter_column"]] = f"eq.{params['filter_value']}"

        try:
            resp = requests.get(url, headers=headers, params=query_params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                return json.dumps(data, indent=2) if data else "No records found."
            else:
                return f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"Query failed: {str(e)}"


class SupabaseInsertTool(BaseTool):
    """Tool for inserting data into Supabase tables."""
    name: str = "supabase_insert"
    description: str = (
        "Insert data into a Supabase table. "
        "Input should be JSON with 'table' (required) and 'data' (required, "
        "a JSON object or array of objects to insert). "
        "Example: {\"table\": \"tasks\", \"data\": {\"title\": \"Build API\", \"status\": \"pending\"}}"
    )

    def _run(self, query: str) -> str:
        try:
            params = json.loads(query) if isinstance(query, str) else query
        except json.JSONDecodeError:
            return "Error: Invalid JSON input."

        table = params.get("table")
        data = params.get("data")
        if not table or not data:
            return "Error: 'table' and 'data' are required."

        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/{table}"
        headers = {
            "apikey": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_ROLE_KEY')}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        try:
            resp = requests.post(url, headers=headers, json=data, timeout=30)
            if resp.status_code in [200, 201]:
                return f"✅ Inserted successfully: {resp.json()}"
            else:
                return f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"Insert failed: {str(e)}"


class SupabaseUpdateTool(BaseTool):
    """Tool for updating data in Supabase tables."""
    name: str = "supabase_update"
    description: str = (
        "Update data in a Supabase table. "
        "Input should be JSON with 'table' (required), 'data' (required, "
        "fields to update), 'filter_column' and 'filter_value' (required, "
        "which records to update). "
        "Example: {\"table\": \"tasks\", \"data\": {\"status\": \"done\"}, "
        "\"filter_column\": \"id\", \"filter_value\": 1}"
    )

    def _run(self, query: str) -> str:
        try:
            params = json.loads(query) if isinstance(query, str) else query
        except json.JSONDecodeError:
            return "Error: Invalid JSON input."

        table = params.get("table")
        data = params.get("data")
        filter_col = params.get("filter_column")
        filter_val = params.get("filter_value")

        if not all([table, data, filter_col, filter_val]):
            return "Error: 'table', 'data', 'filter_column', and 'filter_value' are required."

        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/{table}"
        headers = {
            "apikey": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_ROLE_KEY')}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        params_url = {filter_col: f"eq.{filter_val}"}

        try:
            resp = requests.patch(url, headers=headers, params=params_url, json=data, timeout=30)
            if resp.status_code == 200:
                return f"✅ Updated successfully: {resp.json()}"
            else:
                return f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"Update failed: {str(e)}"


class SupabaseDeleteTool(BaseTool):
    """Tool for deleting data from Supabase tables."""
    name: str = "supabase_delete"
    description: str = (
        "Delete data from a Supabase table. "
        "Input should be JSON with 'table' (required), 'filter_column' "
        "and 'filter_value' (required, which records to delete). "
        "Example: {\"table\": \"tasks\", \"filter_column\": \"id\", \"filter_value\": 1}"
    )

    def _run(self, query: str) -> str:
        try:
            params = json.loads(query) if isinstance(query, str) else query
        except json.JSONDecodeError:
            return "Error: Invalid JSON input."

        table = params.get("table")
        filter_col = params.get("filter_column")
        filter_val = params.get("filter_value")

        if not all([table, filter_col, filter_val]):
            return "Error: 'table', 'filter_column', and 'filter_value' are required."

        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/{table}"
        headers = {
            "apikey": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_ROLE_KEY')}",
        }
        params_url = {filter_col: f"eq.{filter_val}"}

        try:
            resp = requests.delete(url, headers=headers, params=params_url, timeout=30)
            if resp.status_code in [200, 204]:
                return "✅ Deleted successfully."
            else:
                return f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"Delete failed: {str(e)}"


class SupabaseListTablesTool(BaseTool):
    """Tool for listing all tables in the Supabase database."""
    name: str = "supabase_list_tables"
    description: str = (
        "List all available tables in the Supabase database. "
        "No input required — just pass an empty string."
    )

    def _run(self, query: str = "") -> str:
        url = f"{os.getenv('SUPABASE_URL')}/rest/v1/"
        headers = {
            "apikey": os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
            "Authorization": f"Bearer {os.getenv('SUPABASE_SERVICE_ROLE_KEY')}",
        }
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            if resp.status_code == 200:
                paths = resp.json().get("paths", {})
                tables = list(paths.keys())
                return f"Available tables: {', '.join(tables)}" if tables else "No tables found."
            return f"Error {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"List tables failed: {str(e)}"


def get_supabase_tools():
    """Return all Supabase tools for agent use."""
    return [
        SupabaseQueryTool(),
        SupabaseInsertTool(),
        SupabaseUpdateTool(),
        SupabaseDeleteTool(),
        SupabaseListTablesTool(),
    ]


def is_supabase_configured():
    """Check if Supabase env vars are set."""
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_ROLE_KEY"))
