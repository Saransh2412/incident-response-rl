param(
    [ValidateSet("test", "smoke", "api", "validate", "baseline")]
    [string]$Mode = "test",

    [string]$BindHost = "0.0.0.0",

    [int]$Port = 7860
)

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "Virtual environment not found at .venv. Create it first with: py -3.12 -m venv .venv"
    exit 1
}

switch ($Mode) {
    "test" {
        & $python -m pytest
        exit $LASTEXITCODE
    }
    "smoke" {
        & $python -m incident_response_rl.smoke
        exit $LASTEXITCODE
    }
    "api" {
        & $python -m uvicorn server.app:app --host $BindHost --port $Port
        exit $LASTEXITCODE
    }
    "validate" {
        & $python -m openenv.cli validate .
        exit $LASTEXITCODE
    }
    "baseline" {
        & $python inference.py
        exit $LASTEXITCODE
    }
}
