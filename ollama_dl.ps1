[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Name,
    
    [string]$Registry = "https://registry.ollama.ai/",
    
    [Alias("d")]
    [string]$DestDir
)

if ($ExecutionContext.SessionState.LanguageMode -eq 'ConstrainedLanguage') {
    Write-Host "Error: This script requires full language mode to run. Please run the script in an environment that supports FullLanguage mode." -ForegroundColor Red
    exit 1
}

# Try to load the System.Net.Http assembly.
try {
    Add-Type -AssemblyName System.Net.Http -ErrorAction Stop
}
catch {
    Write-Host "Error: The required .NET assembly 'System.Net.Http' could not be loaded. This script requires a full .NET environment. Please run it on PowerShell 7 or an environment that supports these assemblies." -ForegroundColor Red
    exit 1
}


<#
.SYNOPSIS
    Downloads image layers (“blobs”) from a Docker registry.

.DESCRIPTION
    This script fetches an image manifest from the specified registry URL and then downloads each
    recognized layer (blob) to a destination directory. For blobs smaller than 1 MB a simple GET request is used.
    For larger blobs, the script supports resuming partial downloads via HTTP Range requests.
    Download progress is shown using Write-Progress, and up to 10 retries are attempted in case of failures.

.PARAMETER Name
    The name of the image in the form "namespace/repository[:version]". If no namespace is provided,
    "library/" is prepended. If no version tag is provided, ":latest" is assumed.

.PARAMETER Registry
    The base URL of the registry. Default is "https://registry.ollama.ai/".

.PARAMETER DestDir
    The destination directory for downloaded files. If not provided, a default name based on the image name is used.

.EXAMPLE
    pwsh ./ollama_dl.ps1 tinyllama --Registry "https://registry.ollama.ai/" -Verbose
#>

#region Global Constants and Helper Functions

# Constants for size calculations
$BYTES_IN_KILOBYTE = 1024
$BYTES_IN_MEGABYTE = $BYTES_IN_KILOBYTE * $BYTES_IN_KILOBYTE
$DOWNLOAD_READ_SIZE = $BYTES_IN_MEGABYTE  # 1 MB block size

# Mapping from media types to file naming templates.
$MediaTypeToFileTemplate = @{
    "application/vnd.ollama.image.license"  = "license-{0}.txt"
    "application/vnd.ollama.image.model"    = "model-{0}.gguf"
    "application/vnd.ollama.image.params"   = "params-{0}.json"
    "application/vnd.ollama.image.system"    = "system-{0}.txt"
    "application/vnd.ollama.image.template"  = "template-{0}.txt"
}

function Get-ShortHash {
    <#
    .SYNOPSIS
        Extracts a short hash (first 12 characters) from a layer’s digest.
    .PARAMETER layer
        A PSObject representing a layer. Its 'digest' property must be of the form "sha256:<hexdigest>".
    .OUTPUTS
        A string containing the first 12 characters of the digest (after "sha256:").
    .THROWS
        If the digest does not start with "sha256:".
    #>
    param(
        [Parameter(Mandatory = $true)]
        $layer
    )
    if (-not $layer.digest.StartsWith("sha256:")) {
        throw "Unexpected digest: $($layer.digest)"
    }
    return $layer.digest.Substring(7, 12)
}

function Format-Size {
    <#
    .SYNOPSIS
        Formats a file size (in bytes) as a human-readable string.
    .PARAMETER size
        The size in bytes.
    .OUTPUTS
        A string such as "512 B", "12 KB", or "3 MB".
    #>
    param(
        [Parameter(Mandatory = $true)]
        [int]$size
    )
    if ($size -lt $BYTES_IN_KILOBYTE) {
        return "$size B"
    }
    elseif ($size -lt $BYTES_IN_MEGABYTE) {
        $kb = [math]::Floor($size / $BYTES_IN_KILOBYTE)
        return "$kb KB"
    }
    else {
        $mb = [math]::Floor($size / $BYTES_IN_MEGABYTE)
        return "$mb MB"
    }
}
#endregion

#region Download Job Functions

function Get-DownloadJobsForImage {
    <#
    .SYNOPSIS
        Retrieves download jobs for each recognized layer in the image manifest.
    .PARAMETER client
        A .NET HttpClient instance.
    .PARAMETER Registry
        The registry base URL.
    .PARAMETER DestDir
        The destination directory for blobs.
    .PARAMETER Name
        The image name (repository).
    .PARAMETER Version
        The image tag/version.
    .OUTPUTS
        An array of PSObjects representing download jobs. Each object has properties: layer, destPath, blobUrl, and size.
    .THROWS
        If the manifest media type is not as expected.
    #>
    param(
        [Parameter(Mandatory = $true)]
        [System.Net.Http.HttpClient]$client,
        [string]$Registry,
        [string]$DestDir,
        [string]$Name,
        [string]$Version
    )
    # Trim and ensure $Registry ends with a slash.
    $Registry = $Registry.Trim()
    if (-not $Registry.EndsWith("/")) { $Registry += "/" }

    # Build the manifest URL by concatenating strings.
    $manifestUrl = "$Registry" + "v2/$Name/manifests/$Version"
    Write-Host "Fetching manifest from $manifestUrl"
    
    $response = $client.GetAsync($manifestUrl).Result
    $response.EnsureSuccessStatusCode() | Out-Null
    $manifestJson = $response.Content.ReadAsStringAsync().Result

    # Attempt to convert the response to JSON.
    try {
        $manifest = $manifestJson | ConvertFrom-Json
    }
    catch {
        throw "Failed to convert manifest response to JSON. Response content:`n$manifestJson"
    }

    if ($manifest.mediaType -ne "application/vnd.docker.distribution.manifest.v2+json") {
        throw "Unexpected media type for manifest: $($manifest.mediaType)"
    }
    $jobs = @()
    # Sort layers by size (smallest first).
    $sortedLayers = $manifest.layers | Sort-Object size
    foreach ($layer in $sortedLayers) {
        if (-not $MediaTypeToFileTemplate.ContainsKey($layer.mediaType)) {
            Write-Warning "Ignoring layer with unknown media type: $($layer.mediaType)"
            continue
        }
        $shortHash = Get-ShortHash $layer
        $fileName = $MediaTypeToFileTemplate[$layer.mediaType] -f $shortHash
        $destPath = Join-Path $DestDir $fileName

        # Build the blob URL.
        $blobUrl = "$Registry" + "v2/$Name/blobs/$($layer.digest)"
        $job = [PSCustomObject]@{
            layer    = $layer
            destPath = $destPath
            blobUrl  = $blobUrl
            size     = $layer.size
        }
        $jobs += $job
    }
    return $jobs
}

function Get-Blob {
    <#
    .SYNOPSIS
        Downloads a blob (image layer) using retry logic and HTTP Range for resume support.
    .PARAMETER client
        A .NET HttpClient instance.
    .PARAMETER job
        A download job object with properties: blobUrl, destPath, and size.
    .PARAMETER numRetries
        Maximum number of retry attempts (default is 10).
    #>
    param(
        [Parameter(Mandatory = $true)]
        [System.Net.Http.HttpClient]$client,
        [Parameter(Mandatory = $true)]
        $job,
        [int]$numRetries = 10
    )
    Write-Host "Downloading $($job.destPath) ($(Format-Size $job.size))"
    # Ensure the destination directory exists.
    $destDir = Split-Path $job.destPath -Parent
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }

    # Create a temporary file name (appending a timestamp).
    $tempPath = "$($job.destPath).tmp-$([int][double]::Parse((Get-Date -UFormat %s)))"
    $blockSize = $DOWNLOAD_READ_SIZE
    $attempt = 0
    do {
        $attempt++
        if ($attempt -gt 1) {
            Write-Host "Retry attempt $attempt for $($job.blobUrl)"
        }
        try {
            if ($job.size -lt $BYTES_IN_MEGABYTE) {
                # For small files, perform a simple GET.
                $response = $client.GetAsync($job.blobUrl).Result
                $response.EnsureSuccessStatusCode() | Out-Null
                $data = $response.Content.ReadAsByteArrayAsync().Result
                [System.IO.File]::WriteAllBytes($tempPath, $data)
            }
            else {
                # For larger files, attempt to resume if a temporary file exists.
                $startOffset = 0
                if (Test-Path $tempPath) {
                    $startOffset = (Get-Item $tempPath).Length
                }
                $request = New-Object System.Net.Http.HttpRequestMessage 'GET', $job.blobUrl
                if ($startOffset -gt 0) {
                    # Set the Range header to resume download.
                    $rangeHeader = New-Object System.Net.Http.Headers.RangeHeaderValue($startOffset, $null)
                    $request.Headers.Range = $rangeHeader
                }
                $response = $client.SendAsync($request, [System.Net.Http.HttpCompletionOption]::ResponseHeadersRead).Result
                if (($startOffset -gt 0 -and $response.StatusCode -ne [System.Net.HttpStatusCode]::PartialContent) -or 
                    ($startOffset -eq 0 -and $response.StatusCode -ne [System.Net.HttpStatusCode]::OK)) {
                    throw "Unexpected status code: $($response.StatusCode)"
                }
                $response.EnsureSuccessStatusCode() | Out-Null
                $stream = $response.Content.ReadAsStreamAsync().Result
                $fileStream = [System.IO.File]::Open($tempPath, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write)
                try {
                    $buffer = New-Object byte[] $blockSize
                    $totalRead = $startOffset
                    while (($read = $stream.Read($buffer, 0, $buffer.Length)) -gt 0) {
                        $fileStream.Write($buffer, 0, $read)
                        $totalRead += $read
                        $percent = [math]::Round(($totalRead / $job.size * 100), 2)
                        Write-Progress -Activity "Downloading" -Status "$percent% complete" -PercentComplete $percent
                    }
                }
                finally {
                    $fileStream.Close()
                    $stream.Close()
                }
            }
            # Verify the downloaded size.
            $downloadedSize = (Get-Item $tempPath).Length
            if ($downloadedSize -ne $job.size) {
                throw "Did not download expected size: $downloadedSize != $($job.size)"
            }
            Move-Item -Path $tempPath -Destination $job.destPath -Force
            Write-Progress -Activity "Downloading" -Completed
            Write-Host "Downloaded $($job.destPath)"
            return
        }
        catch {
            Write-Warning "$($job.blobUrl): Attempt $attempt/$numRetries failed: $_"
            if ($attempt -ge $numRetries) {
                throw "Failed to download $($job.blobUrl) after $numRetries attempts."
            }
        }
        finally {
            if (Test-Path $tempPath) {
                Remove-Item $tempPath -Force -ErrorAction SilentlyContinue
            }
        }
    } while ($attempt -lt $numRetries)
}

function Get-Image {
    <#
    .SYNOPSIS
        Coordinates the download of all image layers (blobs) as specified in the manifest.
    .PARAMETER Registry
        The registry base URL.
    .PARAMETER Name
        The image name (repository).
    .PARAMETER Version
        The image tag/version.
    .PARAMETER DestDir
        The destination directory for downloads.
    #>
    param(
        [string]$Registry,
        [string]$Name,
        [string]$Version,
        [string]$DestDir
    )
    # Create a .NET HttpClient instance.
    $clientHandler = New-Object System.Net.Http.HttpClientHandler
    $client = New-Object System.Net.Http.HttpClient($clientHandler)
    
    $jobs = Get-DownloadJobsForImage -client $client -Registry $Registry -DestDir $DestDir -Name $Name -Version $Version
    foreach ($job in $jobs) {
        if (Test-Path $job.destPath) {
            Write-Host "Already have $($job.destPath)"
            continue
        }
        Get-Blob -client $client -job $job
    }
    $client.Dispose()
}
#endregion

#region Main Script Execution

# If no namespace is provided, prepend "library/".
if ($Name -notmatch "/") {
    $Name = "library/$Name"
}

# Set the destination directory if not provided.
if (-not $DestDir) {
    # Use the image name (with "/" and ":" replaced) as the default directory.
    $DestDir = $Name -replace "/", "-" -replace ":", "-"
}
Write-Host "Downloading to: $DestDir"

# Append default tag if missing.
if ($Name -notmatch ":") {
    $Name = "${Name}:latest"
}

# Split the image name and version.
$parts = $Name.Split(":", 2)
$repoName = $parts[0]
$version = $parts[1]

Get-Image -Registry $Registry -Name $repoName -Version $version -DestDir $DestDir

#endregion
