# Copy ResearchFlow into a new project directory (same codebase, fresh git history).
# Usage:
#   .\scripts\new-from-researchflow.ps1 -Destination "..\My-New-Project"
#   .\scripts\new-from-researchflow.ps1 -Destination "..\My-New-Project" -PackageName "myflow"

param(
    [Parameter(Mandatory = $true)]
    [string] $Destination,

    [string] $PackageName = "researchflow",

    [string] $Source = ""
)

$ErrorActionPreference = "Stop"

if (-not $Source) {
    $Source = Split-Path $PSScriptRoot -Parent
}

$destPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Destination)
if (Test-Path $destPath) {
    throw "Destination already exists: $destPath"
}

$excludeDirs = @(".git", ".venv", ".chroma", ".checkpoints", "__pycache__", ".pytest_cache", ".ruff_cache", "researchflow.egg-info")
$excludeFiles = @(".env", "live_run_output.json", "live_run_output2.json", "live_run_output.txt", "live_run_stderr.txt", "live_run_stderr2.txt")

New-Item -ItemType Directory -Path $destPath -Force | Out-Null

Get-ChildItem -LiteralPath $Source -Force | ForEach-Object {
    if ($_.PSIsContainer) {
        if ($excludeDirs -contains $_.Name) { return }
        if ($_.Name -eq "Data") {
            robocopy $_.FullName (Join-Path $destPath "Data") /E /XD runs /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
            if ($LASTEXITCODE -ge 8) { throw "robocopy failed for Data" }
            return
        }
        robocopy $_.FullName (Join-Path $destPath $_.Name) /E /XD $excludeDirs /NFL /NDL /NJH /NJS /nc /ns /np | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy failed for $($_.Name)" }
    } else {
        if ($excludeFiles -contains $_.Name) { return }
        Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $destPath $_.Name)
    }
}

$pkg = $PackageName.Trim().ToLower() -replace '[^a-z0-9_]', '_'
if ($pkg -ne "researchflow") {
    $oldDir = Join-Path $destPath "src\researchflow"
    $newDir = Join-Path $destPath "src\$pkg"
    if (-not (Test-Path $oldDir)) { throw "Expected package dir not found: $oldDir" }
    Rename-Item -LiteralPath $oldDir -NewName $pkg

    $textExtensions = @(".py", ".toml", ".md", ".example", ".json", ".txt")
    Get-ChildItem -Path $destPath -Recurse -File | Where-Object {
        $textExtensions -contains $_.Extension -and $_.FullName -notmatch '\\\.git\\'
    } | ForEach-Object {
        $content = Get-Content -LiteralPath $_.FullName -Raw -Encoding UTF8
        $updated = $content -replace 'researchflow', $pkg -replace 'researchflow-api', "$pkg-api"
        if ($updated -ne $content) {
            Set-Content -LiteralPath $_.FullName -Value $updated -Encoding UTF8 -NoNewline
        }
    }
}

Write-Host "Created project at: $destPath"
Write-Host "Package name: $pkg"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  cd `"$destPath`""
Write-Host "  git init"
Write-Host "  python -m venv .venv"
Write-Host "  .venv\Scripts\Activate.ps1"
Write-Host "  pip install -e `".[dev]`""
Write-Host "  copy .env.example .env"
