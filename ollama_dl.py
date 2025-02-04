"""
A command-line utility to download image layers (blobs) from a registry.

The program fetches a manifest for a given image and version, then downloads each layer
(blob) associated with the image using asynchronous HTTP requests. Progress is displayed
using a Rich progress bar. Partial downloads are supported via HTTP Range requests, and
retries are attempted upon network errors.
"""

import argparse
import asyncio
import dataclasses
import logging
import pathlib
import time
from typing import AsyncIterable
from urllib.parse import urljoin

import httpx
from rich.logging import RichHandler
from rich.progress import Progress, TaskID

# Constants for byte size conversions and download chunk size.
BYTES_IN_KILOBYTE = 1024
BYTES_IN_MEGABYTE = BYTES_IN_KILOBYTE**2
DOWNLOAD_READ_SIZE = BYTES_IN_MEGABYTE

# Set up logger for the module.
log = logging.getLogger("ollama-dl")

# Mapping from media types to output file naming templates.
media_type_to_file_template = {
    "application/vnd.ollama.image.license": "license-{shorthash}.txt",
    "application/vnd.ollama.image.model": "model-{shorthash}.gguf",
    "application/vnd.ollama.image.params": "params-{shorthash}.json",
    "application/vnd.ollama.image.system": "system-{shorthash}.txt",
    "application/vnd.ollama.image.template": "template-{shorthash}.txt",
}


def get_short_hash(layer: dict) -> str:
    """
    Extracts a short hash from a layer's digest.

    The layer's 'digest' field is expected to be of the form "sha256:<hexdigest>".
    This function returns the first 12 characters of the hexdigest.

    Args:
        layer (dict): A dictionary containing layer information, including a "digest" key.

    Returns:
        str: A 12-character short hash derived from the full digest.

    Raises:
        ValueError: If the digest does not start with "sha256:".
    """
    if not layer["digest"].startswith("sha256:"):
        raise ValueError(f"Unexpected digest: {layer['digest']}")
    # Partition the digest string and return the first 12 characters of the hex part.
    return layer["digest"].partition(":")[2][:12]


def format_size(size: int) -> str:
    """
    Formats a size in bytes into a human-readable string.

    Args:
        size (int): The size in bytes.

    Returns:
        str: A human-readable string (B, KB, or MB).
    """
    if size < BYTES_IN_KILOBYTE:
        return f"{size} B"
    if size < BYTES_IN_MEGABYTE:
        return f"{size // BYTES_IN_KILOBYTE} KB"
    return f"{size // BYTES_IN_MEGABYTE} MB"


@dataclasses.dataclass(frozen=True)
class DownloadJob:
    """
    Data class representing a download job for an image layer (blob).

    Attributes:
        layer (dict): The layer metadata from the manifest.
        dest_path (pathlib.Path): The destination file path where the blob will be saved.
        blob_url (str): The URL from which to download the blob.
        size (int): The expected size of the blob in bytes.
    """
    layer: dict
    dest_path: pathlib.Path
    blob_url: str
    size: int


async def _inner_download(
    client: httpx.AsyncClient,
    *,
    url: str,
    temp_path: pathlib.Path,
    size: int,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """
    Downloads a blob from a URL into a temporary file, updating progress as it downloads.

    For small blobs (less than 1 MB), the blob is downloaded in one go.
    For larger blobs, the download supports resuming by checking if a partial file exists.

    Args:
        client (httpx.AsyncClient): The HTTP client used to perform requests.
        url (str): The URL to download from.
        temp_path (pathlib.Path): The path to a temporary file where data is saved.
        size (int): The expected total size of the blob in bytes.
        progress (Progress): A Rich Progress object to update the download progress.
        task_id (TaskID): The task identifier for progress updates.

    Raises:
        ValueError: If the HTTP status code is not as expected.
        httpx.HTTPStatusError: For any non-success HTTP status.
    """
    # If the blob is small, perform a simple GET request.
    if size < BYTES_IN_MEGABYTE:
        resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        temp_path.write_bytes(resp.content)
        return

    # For larger blobs, attempt to resume the download if a partial file exists.
    if temp_path.is_file():
        start_offset = temp_path.stat().st_size
        headers = {"Range": f"bytes={start_offset}-"}
    else:
        start_offset = 0
        headers = {}

    # Stream the blob content.
    async with client.stream(
        "GET",
        url,
        headers=headers,
        follow_redirects=True,
    ) as resp:
        # Verify that the response status is either 206 (partial content) or 200 (full content)
        if resp.status_code != (206 if start_offset else 200):
            raise ValueError(f"Unexpected status code: {resp.status_code}")
        resp.raise_for_status()
        # Open the temporary file in append-binary mode.
        with temp_path.open("ab") as f:
            # Write chunks of data to the file and update progress.
            async for chunk in resp.aiter_bytes(DOWNLOAD_READ_SIZE):
                f.write(chunk)
                progress.update(task_id, completed=f.tell())


async def download_blob(
    client: httpx.AsyncClient,
    job: DownloadJob,
    *,
    progress: Progress,
    num_retries: int = 10,
) -> None:
    """
    Downloads a blob (image layer) with retry logic and progress tracking.

    This function creates any necessary parent directories, attempts to download the blob,
    and verifies that the downloaded file size matches the expected size. A temporary file is
    used during download to avoid incomplete downloads.

    Args:
        client (httpx.AsyncClient): The HTTP client used for making requests.
        job (DownloadJob): The download job containing blob information.
        progress (Progress): A Rich Progress object to display download progress.
        num_retries (int): Maximum number of download retry attempts (default is 10).

    Raises:
        httpx.TransportError: If all retries fail due to transport issues.
        RuntimeError: If the final downloaded file size does not match the expected size.
    """
    # Ensure the destination directory exists.
    job.dest_path.parent.mkdir(parents=True, exist_ok=True)
    task_desc = f"{job.dest_path} ({format_size(job.size)})"
    # Create a progress task for this download.
    task = progress.add_task(task_desc, total=job.size)
    # Use a temporary file name with a unique suffix (using current time).
    temp_path = job.dest_path.with_suffix(f".tmp-{time.time()}")
    try:
        # Retry loop for downloading the blob.
        for attempt in range(1, num_retries + 1):
            if attempt != 1:
                progress.update(
                    task,
                    description=f"{task_desc} (retry {attempt}/{num_retries})",
                )
            try:
                await _inner_download(
                    client,
                    url=job.blob_url,
                    temp_path=temp_path,
                    size=job.size,
                    progress=progress,
                    task_id=task,
                )
            except httpx.TransportError as exc:
                # Log the error and retry if attempts remain.
                log.warning(
                    "%s: Attempt %d/%d failed: %s",
                    job.blob_url,
                    attempt,
                    num_retries,
                    exc,
                )
                if attempt == num_retries:
                    raise
            else:
                # Exit loop if download succeeds.
                break

        # Verify that the downloaded file size is as expected.
        result_size = temp_path.stat().st_size
        if result_size != job.size:
            raise RuntimeError(
                f"Did not download expected size: {result_size} != {job.size}",
            )
        # Rename the temporary file to the final destination.
        temp_path.rename(job.dest_path)
        progress.update(task, completed=job.size)
    finally:
        # Cleanup: remove the temporary file if it still exists.
        if temp_path.is_file():
            temp_path.unlink()


async def get_download_jobs_for_image(
    *,
    client: httpx.AsyncClient,
    registry: str,
    dest_dir: str,
    name: str,
    version: str,
) -> AsyncIterable[DownloadJob]:
    """
    Retrieves download jobs for all recognized layers in an image manifest.

    The function fetches the image manifest from the registry, parses it, and yields
    DownloadJob objects for each layer whose media type is recognized.

    Args:
        client (httpx.AsyncClient): The HTTP client for making requests.
        registry (str): The base URL of the registry.
        dest_dir (str): The directory where blobs will be saved.
        name (str): The name of the image (repository).
        version (str): The image version (tag).

    Yields:
        DownloadJob: An object representing the download task for a layer.

    Raises:
        ValueError: If the manifest's media type is not as expected.
    """
    # Construct the URL for the image manifest.
    manifest_url = urljoin(registry, f"v2/{name}/manifests/{version}")
    resp = await client.get(manifest_url)
    resp.raise_for_status()
    manifest_data = resp.json()
    manifest_media_type = manifest_data["mediaType"]
    # Verify that the manifest media type is as expected.
    if manifest_media_type != "application/vnd.docker.distribution.manifest.v2+json":
        raise ValueError(
            f"Unexpected media type for manifest: {manifest_media_type}",
        )
    # Sort layers by size (smallest first) and yield jobs for known media types.
    for layer in sorted(manifest_data["layers"], key=lambda x: x["size"]):
        file_template = media_type_to_file_template.get(layer["mediaType"])
        if not file_template:
            log.warning(
                "Ignoring layer with unknown media type: %s",
                layer["mediaType"],
            )
            continue
        # Create a filename using the template and a short hash from the layer digest.
        filename = file_template.format(shorthash=get_short_hash(layer))
        dest_path = pathlib.Path(dest_dir) / filename
        yield DownloadJob(
            layer=layer,
            dest_path=dest_path,
            blob_url=urljoin(registry, f"v2/{name}/blobs/{layer['digest']}"),
            size=layer["size"],
        )


async def download(*, registry: str, name: str, version: str, dest_dir: str) -> None:
    """
    Coordinates the download of all layers for an image.

    A Rich progress bar is used to display download progress. For each download job,
    the function checks if the file already exists; if not, it schedules a download task.

    Args:
        registry (str): The registry base URL.
        name (str): The image name (repository).
        version (str): The image version (tag).
        dest_dir (str): The destination directory for the downloads.
    """
    # Create a progress context to display progress bars.
    with Progress() as progress:
        async with httpx.AsyncClient() as client:
            tasks = []
            # Retrieve download jobs by parsing the manifest.
            async for job in get_download_jobs_for_image(
                client=client,
                registry=registry,
                dest_dir=dest_dir,
                name=name,
                version=version,
            ):
                # Skip download if the destination file already exists.
                if job.dest_path.is_file():
                    log.info("Already have %s", job.dest_path)
                    continue
                tasks.append(download_blob(client, job, progress=progress))
            # Run all download tasks concurrently.
            if tasks:
                await asyncio.gather(*tasks)


def main() -> None:
    """
    The entry point for the command-line utility.

    This function parses command-line arguments, sets up logging, processes the image name
    and version, and triggers the asynchronous download process.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("name", help="Name of the image (repository[:tag])")
    ap.add_argument("--registry", default="https://registry.ollama.ai/",
                    help="URL of the registry to download from")
    ap.add_argument("-d", "--dest-dir", default=None,
                    help="Destination directory for downloads (defaults based on image name)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose logging")
    args = ap.parse_args()

    # Configure logging with RichHandler for better console output.
    logging.basicConfig(
        format="%(message)s",
        level=(logging.DEBUG if args.verbose else logging.INFO),
        handlers=[RichHandler(show_path=False)],
    )
    # Reduce verbosity of httpx logs unless verbose is enabled.
    if not args.verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)

    # Process the image name; prepend "library/" if no namespace is given.
    name = args.name
    if "/" not in name:
        name = f"library/{name}"

    # Set the destination directory based on input or a default derived from the image name.
    dest_dir = args.dest_dir
    if not dest_dir:
        dest_dir = name.replace("/", "-").replace(":", "-")
    log.info("Downloading to: %s", dest_dir)

    # If no tag is specified, default to "latest".
    if ":" not in name:
        name += ":latest"
    # Split the image name and version.
    name, _, version = name.rpartition(":")

    # Run the asynchronous download process.
    asyncio.run(
        download(
            registry=args.registry,
            name=name,
            dest_dir=dest_dir,
            version=version,
        ),
    )


if __name__ == "__main__":
    main()
