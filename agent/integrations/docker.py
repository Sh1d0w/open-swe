"""Docker sandbox backend implementation.

Provides a local Docker-based sandbox for development that offers better isolation
than the `local` backend while avoiding cloud dependencies.

WARNING: This runs containers on your local Docker daemon with host directory mounts.
Intended for local development only, not production use.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
    SandboxBackendProtocol,
    WriteResult,
)
from deepagents.backends.sandbox import BaseSandbox


def create_docker_sandbox(sandbox_id: str | None = None):
    """Create or connect to a Docker sandbox.

    This function creates a new ephemeral container for each agent run. Containers
    are named with pattern `open-swe-{timestamp}-{random}` for uniqueness and should
    be cleaned up explicitly via the provider's delete() method.

    Args:
        sandbox_id: Optional existing container ID to connect to. If None, creates
            a new container.

    Returns:
        DockerBackend instance implementing SandboxBackendProtocol.

    Environment variables:
        DOCKER_IMAGE: Container image (default: python:3-slim)
        DOCKER_CPU_LIMIT: CPU limit in cores (default: 2.0)
        DOCKER_MEMORY_LIMIT: Memory limit in bytes (default: 4294967296 = 4GB)
        DOCKER_WORK_DIR: Host work directory for mounts (default: /tmp/open-swe-work)
    """
    provider = DockerProvider()
    backend = provider.get_or_create(sandbox_id=sandbox_id)
    return backend


class SandboxProvider(ABC):
    """Interface for creating and deleting sandbox backends."""

    @abstractmethod
    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get an existing sandbox, or create one if needed."""
        raise NotImplementedError

    @abstractmethod
    def delete(
        self,
        *,
        sandbox_id: str,
        **kwargs: Any,
    ) -> None:
        """Delete a sandbox by id."""
        raise NotImplementedError


class DockerBackend(BaseSandbox):
    """Docker backend implementation conforming to SandboxBackendProtocol.

    This implementation extends BaseSandbox and overrides execute() for subprocess-based
    command execution with timeout handling, and file operations (write, download_files,
    upload_files) using the Docker SDK directly to avoid ARG_MAX issues.
    """

    def __init__(self, container: Any, work_dir: str) -> None:
        """Initialize Docker backend.

        Args:
            container: Docker container object from docker-py SDK.
            work_dir: Working directory inside the container.
        """
        self._container = container
        self._work_dir = work_dir
        self._default_timeout: int = 300  # 5 minute default

    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend (container ID)."""
        return self._container.short_id

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        """Execute a command in the Docker container.

        Uses subprocess with docker exec to run commands, implementing timeout
        at the Python level since docker-py SDK doesn't provide native timeout.

        Args:
            command: Full shell command string to execute.
            timeout: Maximum time in seconds to wait for completion.
                If None, uses default of 5 minutes.

        Returns:
            ExecuteResponse with combined output, exit code, and truncation flag.
        """
        effective_timeout = timeout if timeout is not None else self._default_timeout

        try:
            # Use docker exec via subprocess for better timeout control
            result = subprocess.run(
                f"docker exec {self._container.short_id} sh -c {command!r}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )

            # Combine stdout and stderr (matching other backends' approach)
            output = result.stdout or ""
            if result.stderr:
                output += "\n" + result.stderr if output else result.stderr

            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=False,
            )
        except subprocess.TimeoutExpired as e:
            # Kill the container process on timeout
            try:
                self._container.kill()
            except Exception:
                pass

            return ExecuteResponse(
                output=f"Command timed out after {effective_timeout} seconds",
                exit_code=-1,
                truncated=False,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Execution error: {e}",
                exit_code=-1,
                truncated=False,
            )

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write content to a file using Docker SDK's put_archive.

        This avoids ARG_MAX issues that occur when passing large content via shell commands.

        Args:
            file_path: Path inside the container where content should be written.
            content: String content to write.

        Returns:
            WriteResult with path or error message.
        """
        try:
            # Ensure parent directory exists
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                self._container.run(f"mkdir -p {parent_dir!r}", detach=True)

            # Create tar archive in memory
            tar_buffer = BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
                # Create file info
                data = content.encode("utf-8")
                tarinfo = tarfile.TarInfo(name=os.path.basename(file_path))
                tarinfo.size = len(data)
                tarinfo.mtime = int(time.time())

                # Add to archive
                tar.addfile(tarinfo, BytesIO(data))

            # Upload archive to container
            tar_buffer.seek(0)
            self._container.put_archive(parent_dir or "/", tar_buffer.read())

            return WriteResult(path=file_path, files_update=None)
        except Exception as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {e}")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download multiple files from the Docker container.

        Args:
            paths: List of file paths inside the container.

        Returns:
            List of FileDownloadResponse with path, content, and optional error.
        """
        responses: list[FileDownloadResponse] = []

        for path in paths:
            try:
                # Use get_archive to retrieve file
                bits, _ = self._container.get_archive(path)

                # Extract from tar archive
                content = None
                for chunk in bits:
                    with tarfile.open(fileobj=BytesIO(chunk), mode="r:gz") as tar:
                        members = tar.getmembers()
                        if members:
                            member = tar.extractfile(members[0])
                            if member:
                                content = member.read().decode("utf-8")
                                break

                responses.append(
                    FileDownloadResponse(path=path, content=content or "", error=None)
                )
            except Exception as e:
                responses.append(FileDownloadResponse(path=path, content="", error=str(e)))

        return responses

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload multiple files to the Docker container.

        Args:
            files: List of (path, content_bytes) tuples.

        Returns:
            List of FileUploadResponse with path and optional error.
        """
        responses: list[FileUploadResponse] = []

        for path, content in files:
            try:
                # Ensure parent directory exists
                parent_dir = os.path.dirname(path)
                if parent_dir:
                    self._container.run(f"mkdir -p {parent_dir!r}", detach=True)

                # Create tar archive
                tar_buffer = BytesIO()
                with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
                    tarinfo = tarfile.TarInfo(name=os.path.basename(path))
                    tarinfo.size = len(content)
                    tarinfo.mtime = int(time.time())
                    tar.addfile(tarinfo, BytesIO(content))

                # Upload to container
                tar_buffer.seek(0)
                self._container.put_archive(parent_dir or "/", tar_buffer.read())

                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))

        return responses

    def get_work_dir(self) -> str:
        """Return the working directory inside the container.

        Used by sandbox_paths.py to resolve repository paths.

        Returns:
            Absolute path to work directory inside container.
        """
        return self._work_dir


class DockerProvider(SandboxProvider):
    """Docker sandbox provider implementation.

    Manages Docker container lifecycle using the docker-py SDK. Creates ephemeral
    containers for each agent run with configurable resource limits and UID/GID mapping.
    """

    def __init__(self) -> None:
        """Initialize Docker provider."""
        try:
            import docker

            self._client = docker.from_env()
        except Exception as e:
            msg = f"Failed to connect to Docker daemon: {e}. Ensure Docker is running."
            raise RuntimeError(msg) from e

    def get_or_create(
        self,
        *,
        sandbox_id: str | None = None,
        **kwargs: Any,
    ) -> SandboxBackendProtocol:
        """Get existing container or create new one.

        Args:
            sandbox_id: Optional existing container ID to connect to.
            **kwargs: Ignored (for interface compatibility).

        Returns:
            DockerBackend instance.

        Raises:
            RuntimeError: If Docker daemon is unavailable or container creation fails.
        """
        if kwargs:
            msg = f"Received unsupported arguments: {list(kwargs.keys())}"
            raise TypeError(msg)

        # Connect to existing container if ID provided
        if sandbox_id:
            try:
                container = self._client.containers.get(sandbox_id)
                return DockerBackend(container, self._get_work_dir())
            except Exception as e:
                msg = f"Failed to connect to existing container '{sandbox_id}': {e}"
                raise RuntimeError(msg) from e

        # Create new container
        return self._create_container()

    def delete(self, *, sandbox_id: str, **kwargs: Any) -> None:
        """Delete a Docker container.

        Args:
            sandbox_id: Container ID to remove.
            **kwargs: Ignored (for interface compatibility).
        """
        if kwargs:
            msg = f"Received unsupported arguments: {list(kwargs.keys())}"
            raise TypeError(msg)

        try:
            container = self._client.containers.get(sandbox_id)
            container.stop(timeout=10)
            container.remove()
        except Exception as e:
            # Container may already be removed
            if "No such container" not in str(e):
                raise RuntimeError(f"Failed to delete container '{sandbox_id}': {e}") from e

    def _create_container(self) -> DockerBackend:
        """Create a new ephemeral Docker container.

        Returns:
            DockerBackend instance for the created container.
        """
        # Get configuration from environment
        image = os.getenv("DOCKER_IMAGE", "python:3-slim")
        cpu_limit = os.getenv("DOCKER_CPU_LIMIT", "2.0")
        memory_limit = int(os.getenv("DOCKER_MEMORY_LIMIT", "4294967296"))  # 4GB default
        host_work_dir = os.getenv("DOCKER_WORK_DIR", "/tmp/open-swe-work")

        # Generate unique container name
        timestamp = int(time.time())
        random_suffix = str(uuid.uuid4())[:8]
        container_name = f"open-swe-{timestamp}-{random_suffix}"

        # Get host UID/GID for file permission compatibility
        try:
            user_id = f"{os.getuid()}:{os.getgid()}"
        except Exception:
            user_id = "root"

        # Create host work directory if it doesn't exist
        os.makedirs(host_work_dir, exist_ok=True)

        # Container work directory (inside container)
        container_work_dir = "/work"

        try:
            # Run container in detached mode
            container = self._client.containers.run(
                image,
                command="tail -f /dev/null",  # Keep container running
                detach=True,
                name=container_name,
                user=user_id,  # Run as host user to avoid permission issues
                working_dir=container_work_dir,
                volumes={
                    host_work_dir: {"bind": container_work_dir, "mode": "rw"},
                },
                cpu_period=100000,
                cpu_quota=int(float(cpu_limit) * 100000),
                mem_limit=memory_limit,
                remove=False,  # Don't auto-remove on exit; we manage lifecycle explicitly
            )

            # Wait for container to be ready
            container.wait(condition="running", timeout=30)

            # Verify container is responsive
            max_retries = 10
            for i in range(max_retries):
                try:
                    result = container.exec_run("echo ready")
                    if result.exit_code == 0:
                        break
                except Exception:
                    pass
                time.sleep(1)
            else:
                container.stop(timeout=5)
                container.remove()
                msg = f"Container failed to become responsive within {max_retries} seconds"
                raise RuntimeError(msg)

            return DockerBackend(container, container_work_dir)

        except Exception as e:
            # Clean up on failure
            try:
                container.stop(timeout=5)
                container.remove()
            except Exception:
                pass
            msg = f"Failed to create Docker container: {e}"
            raise RuntimeError(msg) from e

    def _get_work_dir(self) -> str:
        """Get the default work directory path."""
        host_work_dir = os.getenv("DOCKER_WORK_DIR", "/tmp/open-swe-work")
        return "/work"  # Container-side mount point
