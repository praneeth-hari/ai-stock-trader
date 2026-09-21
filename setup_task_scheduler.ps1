# =============================================================================
# setup_task_scheduler.ps1 - Windows Task Scheduler Automated Setup
# =============================================================================
# Configures automated tasks under the "AI-Stock-Trader" folder:
#   1. AI-Stock-Trader-Daily:
#      Runs run_trader.bat every Mon-Fri at 2:00 AM IST (after US market close).
#   2. AI-Stock-Trader-Morning-HealthCheck:
#      Runs run_health_check.bat every Mon-Fri at 9:00 AM IST (morning check).
#   3. AI-Stock-Trader-Startup:
#      Runs run_health_check.bat immediately when the computer starts up.
#   4. AI-Stock-Trader-Dashboard:
#      Runs start_dashboard.bat on startup (with 60-second delay).
#   5. AI-Stock-Trader-DailyStatus:
#      Runs run_daily_status.bat every Mon-Fri at 3:00 AM IST (nightly digest).
#   6. AI-Stock-Trader-Monthly-Retrain:
#      Runs run_retrain.bat on the first Saturday of each month at 10:00 AM IST.
#   7. AI-Stock-Trader-Weekly-Summary:
#      Runs run_weekly_summary.bat every Sunday at 9:00 AM IST (weekly digest).
# =============================================================================

param (
    [switch]$Elevate
)

# Resolve project root from this script location
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$LogsDir = Join-Path $ScriptDir "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir -Force | Out-Null
}

$IsAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if ($Elevate -and (-not $IsAdmin)) {
    Write-Host "[INFO] Requesting Administrator elevation..." -ForegroundColor Cyan
    Start-Process powershell.exe -Verb RunAs -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$($MyInvocation.MyCommand.Path)`""
    exit
}

Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host " AI STOCK TRADER - TASK SCHEDULER SETUP" -ForegroundColor Cyan
Write-Host " Project Directory : $ScriptDir" -ForegroundColor Gray
$AdminStatus = if ($IsAdmin) { 'YES (Highest Privileges enabled)' } else { 'NO (Standard User - Windows Startup Fallback Active)' }
Write-Host " Administrator     : $AdminStatus" -ForegroundColor Gray
Write-Host "=============================================================================" -ForegroundColor Cyan

# Batch file targets (absolute paths)
$TraderBat    = Join-Path $ScriptDir "run_trader.bat"
$HealthBat    = Join-Path $ScriptDir "run_health_check.bat"
$DashboardBat = Join-Path $ScriptDir "start_dashboard.bat"
$StatusBat    = Join-Path $ScriptDir "run_daily_status.bat"
$RetrainBat   = Join-Path $ScriptDir "run_retrain.bat"
$WeeklyBat    = Join-Path $ScriptDir "run_weekly_summary.bat"

# Verify all runner scripts exist
$RequiredFiles = @($TraderBat, $HealthBat, $DashboardBat, $StatusBat, $RetrainBat, $WeeklyBat)
foreach ($file in $RequiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Error "Required script missing: $file"
        exit 1
    }
}

# Define scheduled tasks
$Tasks = @(
    @{
        Name        = 'AI-Stock-Trader\AI-Stock-Trader-Daily'
        Description = 'Automated daily trading pipeline [Mon-Fri at 2:00 AM IST (after US market close)]'
        Schedule    = 'weekly /d MON,TUE,WED,THU,FRI /st 02:00'
        Target      = $TraderBat
    },
    @{
        Name        = 'AI-Stock-Trader\AI-Stock-Trader-Morning-HealthCheck'
        Description = 'Morning health inspection [Mon-Fri at 9:00 AM IST]'
        Schedule    = 'weekly /d MON,TUE,WED,THU,FRI /st 09:00'
        Target      = $HealthBat
    },
    @{
        Name        = 'AI-Stock-Trader\AI-Stock-Trader-Startup'
        Description = 'Startup health check on computer boot or logon'
        Schedule    = if ($IsAdmin) { 'onstart' } else { 'onlogon' }
        Target      = $HealthBat
    },
    @{
        Name        = 'AI-Stock-Trader\AI-Stock-Trader-Dashboard'
        Description = 'Auto-starts Streamlit dashboard in background [delayed 60s]'
        Schedule    = if ($IsAdmin) { 'onlogon /delay 0001:00' } else { 'onlogon' }
        Target      = $DashboardBat
    },
    @{
        Name        = 'AI-Stock-Trader\AI-Stock-Trader-DailyStatus'
        Description = 'Nightly portfolio status report [Mon-Fri at 3:00 AM IST]'
        Schedule    = 'weekly /d MON,TUE,WED,THU,FRI /st 03:00'
        Target      = $StatusBat
    },
    @{
        Name        = 'AI-Stock-Trader\AI-Stock-Trader-Monthly-Retrain'
        Description = 'Monthly model retraining and evaluation [First Saturday of each month at 10:00 AM IST]'
        Schedule    = 'monthly /mo FIRST /d SAT /st 10:00'
        Target      = $RetrainBat
    },
    @{
        Name        = 'AI-Stock-Trader\AI-Stock-Trader-Weekly-Summary'
        Description = 'Weekly portfolio summary digest via Telegram [Every Sunday at 9:00 AM IST]'
        Schedule    = 'weekly /d SUN /st 09:00'
        Target      = $WeeklyBat
    }
)

Write-Host ""
Write-Host "Registering automated tasks..." -ForegroundColor Yellow

$RegisteredCount = 0
$StartupDir = [Environment]::GetFolderPath('Startup')

foreach ($t in $Tasks) {
    $tName = $t.Name
    $tTarget = $t.Target
    $tSched = $t.Schedule

    # Cleanly remove existing task if it already exists
    & schtasks.exe /delete /tn "$tName" /f 2>$null | Out-Null

    # Parse schedule parts
    $schedParts = $tSched -split " "
    $scType = $schedParts[0]

    # Build argument array
    $cmdArgs = @("/create", "/tn", "$tName", "/tr", "`"$tTarget`"", "/sc", $scType)
    for ($i = 1; $i -lt $schedParts.Count; $i++) {
        $cmdArgs += $schedParts[$i]
    }
    if ($IsAdmin) {
        $cmdArgs += @("/rl", "HIGHEST")
    }
    $cmdArgs += "/f"

    $output = & schtasks.exe @cmdArgs 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [OK] Registered in Task Scheduler: $tName" -ForegroundColor Green
        $RegisteredCount++
    } else {
        # If standard user cannot register ONSTART/ONLOGON, create shortcut in user Startup folder
        if ($tName -like "*Startup*" -or $tName -like "*Dashboard*") {
            try {
                $baseName = Split-Path -Leaf $tName
                $shortcutPath = Join-Path $StartupDir "$baseName.lnk"
                $wshShell = New-Object -ComObject WScript.Shell
                $shortcut = $wshShell.CreateShortcut($shortcutPath)
                $shortcut.TargetPath = $tTarget
                $shortcut.WorkingDirectory = $ScriptDir
                $shortcut.Description = $t.Description
                $shortcut.Save()
                Write-Host "  [OK] Registered in Windows Startup: $baseName.lnk (Runs on login)" -ForegroundColor Green
                $RegisteredCount++
            } catch {
                Write-Host "  [FAILED] Could not register $tName : $output" -ForegroundColor Red
            }
        } else {
            Write-Host "  [FAILED] Could not register $tName : $output" -ForegroundColor Red
        }
    }
}

Write-Host ""
Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host " SETUP COMPLETE - AUTOMATION SUMMARY" -ForegroundColor Cyan
Write-Host "=============================================================================" -ForegroundColor Cyan
Write-Host "Active Tasks ($RegisteredCount of $($Tasks.Count) configured successfully):" -ForegroundColor White
Write-Host ""

Write-Host "1. AI-Stock-Trader-Daily" -ForegroundColor Yellow
Write-Host "   Schedule : Monday-Friday at 2:00 AM IST (after US market close)" -ForegroundColor Gray
Write-Host "   Action   : $TraderBat" -ForegroundColor Gray
Write-Host "   Purpose  : Runs daily pipeline after market close with error catching and log cleanup" -ForegroundColor Gray
Write-Host ""

Write-Host "2. AI-Stock-Trader-Morning-HealthCheck" -ForegroundColor Yellow
Write-Host "   Schedule : Monday-Friday at 9:00 AM IST" -ForegroundColor Gray
Write-Host "   Action   : $HealthBat" -ForegroundColor Gray
Write-Host "   Purpose  : Morning health inspection to catch any previous night failures" -ForegroundColor Gray
Write-Host ""

Write-Host "3. AI-Stock-Trader-Startup" -ForegroundColor Yellow
Write-Host "   Schedule : On computer boot or user logon" -ForegroundColor Gray
Write-Host "   Action   : $HealthBat" -ForegroundColor Gray
Write-Host "   Purpose  : Immediately checks for missed runs while computer was off" -ForegroundColor Gray
Write-Host ""

Write-Host "4. AI-Stock-Trader-Dashboard" -ForegroundColor Yellow
Write-Host "   Schedule : On user logon (delayed 60 seconds)" -ForegroundColor Gray
Write-Host "   Action   : $DashboardBat" -ForegroundColor Gray
Write-Host "   Purpose  : Ensures Streamlit UI is running in background at http://localhost:8501" -ForegroundColor Gray
Write-Host ""

Write-Host "5. AI-Stock-Trader-DailyStatus" -ForegroundColor Yellow
Write-Host "   Schedule : Monday-Friday at 3:00 AM IST" -ForegroundColor Gray
Write-Host "   Action   : $StatusBat" -ForegroundColor Gray
Write-Host "   Purpose  : Generates plain-English portfolio digest and alerts to console and logs" -ForegroundColor Gray
Write-Host ""

Write-Host "6. AI-Stock-Trader-Monthly-Retrain" -ForegroundColor Yellow
Write-Host "   Schedule : First Saturday of each month at 10:00 AM IST" -ForegroundColor Gray
Write-Host "   Action   : $RetrainBat" -ForegroundColor Gray
Write-Host "   Purpose  : Monthly model retraining and evaluation" -ForegroundColor Gray
Write-Host ""

Write-Host "7. AI-Stock-Trader-Weekly-Summary" -ForegroundColor Yellow
Write-Host "   Schedule : Every Sunday at 9:00 AM IST" -ForegroundColor Gray
Write-Host "   Action   : $WeeklyBat" -ForegroundColor Gray
Write-Host "   Purpose  : Weekly portfolio summary digest via Telegram" -ForegroundColor Gray
Write-Host "=============================================================================" -ForegroundColor Cyan

if (-not $IsAdmin) {
    Write-Host ""
    Write-Host "[TIP] To register Startup & Dashboard tasks with SYSTEM-level highest privileges," -ForegroundColor Cyan
    Write-Host "      run PowerShell as Administrator or execute:" -ForegroundColor Cyan
    Write-Host "      powershell -ExecutionPolicy Bypass -File .\setup_task_scheduler.ps1 -Elevate" -ForegroundColor Gray
}

Write-Host ""
Write-Host "To view tasks in Windows GUI: Press Win+R, type 'taskschd.msc' and expand 'AI-Stock-Trader'." -ForegroundColor White
