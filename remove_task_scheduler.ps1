# =============================================================================
# remove_task_scheduler.ps1 - Windows Task Scheduler Clean Removal
# =============================================================================
# Cleanly unregisters all automated AI-Stock-Trader tasks from Task Scheduler
# and cleans up any shortcuts in the Windows Startup folder.
# =============================================================================

param (
    [switch]$Quiet
)

$TaskNames = @(
    "AI-Stock-Trader\AI-Stock-Trader-Daily",
    "AI-Stock-Trader\AI-Stock-Trader-Morning-HealthCheck",
    "AI-Stock-Trader\AI-Stock-Trader-HealthCheck",
    "AI-Stock-Trader\AI-Stock-Trader-Startup",
    "AI-Stock-Trader\AI-Stock-Trader-Dashboard",
    "AI-Stock-Trader\AI-Stock-Trader-DailyStatus"
)

Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host " AI STOCK TRADER - TASK SCHEDULER UNINSTALLER" -ForegroundColor Cyan
Write-Host "=============================================================================" -ForegroundColor Cyan

$RemovedCount = 0

foreach ($t in $TaskNames) {
    # Check if task exists in Task Scheduler
    & schtasks.exe /query /tn "$t" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        & schtasks.exe /delete /tn "$t" /f 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "  [REMOVED Task Scheduler] $t" -ForegroundColor Yellow
            $RemovedCount++
        } else {
            Write-Host "  [!] Could not remove $t from Task Scheduler" -ForegroundColor Red
        }
    } else {
        if (-not $Quiet) {
            Write-Host "  [NOT FOUND Task Scheduler] $t" -ForegroundColor Gray
        }
    }
}

# Clean up Windows Startup shortcuts if present
$StartupDir = [Environment]::GetFolderPath('Startup')
$StartupLinks = @(
    "AI-Stock-Trader-Startup.lnk",
    "AI-Stock-Trader-Dashboard.lnk"
)
foreach ($link in $StartupLinks) {
    $linkPath = Join-Path $StartupDir $link
    if (Test-Path $linkPath) {
        Remove-Item -Path $linkPath -Force -ErrorAction SilentlyContinue
        Write-Host "  [REMOVED Windows Startup] $link" -ForegroundColor Yellow
        $RemovedCount++
    }
}

Write-Host "`n=============================================================================" -ForegroundColor Cyan
Write-Host " UNINSTALL COMPLETE: Cleaned up $RemovedCount automated items." -ForegroundColor Green
Write-Host "=============================================================================" -ForegroundColor Cyan
