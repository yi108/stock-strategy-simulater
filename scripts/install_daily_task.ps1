# 設定 Windows 工作排程器：每週二到週六（美股週一到週五收盤後）的早上自動跑訊號。
# 用法（在專案資料夾的 PowerShell 裡執行）：
#   powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -Capital 750 -Time 06:30
#   powershell -ExecutionPolicy Bypass -File scripts\install_daily_task.ps1 -Alt none   # 轉弱時全放國庫券
param(
    [string]$Time = "06:00",
    [string]$Ticker = "QQQ",
    [string]$Alt = "GLD:0.5",
    [double]$Capital = 750,
    [switch]$AlwaysOpen,
    [string]$TaskName = "StockStrategySignal"
)
$ErrorActionPreference = "Stop"

$Repo = Split-Path -Parent $PSScriptRoot
$Script = Join-Path $PSScriptRoot "daily_signal.ps1"

# 找到真正的 python.exe 路徑（排程執行時不一定讀得到你的 PATH）
$Python = (& python -c "import sys; print(sys.executable)").Trim()
if (-not $Python -or -not (Test-Path $Python)) {
    throw "找不到 python，請先確認在 PowerShell 裡打 python 可以執行。"
}
# 用不會寫入錯誤輸出的方式檢查，避免 PowerShell 5.1 在 Stop 模式下中止
& $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('yfinance') else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Warning "這個 Python 沒有安裝 yfinance，請先執行：$Python -m pip install yfinance"
}

if (-not $Alt) { $Alt = "none" }
$argLine = "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Script`" " +
           "-Python `"$Python`" -Ticker $Ticker -Capital $Capital -Alt $Alt"
if ($AlwaysOpen) { $argLine += " -AlwaysOpen" }

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $argLine -WorkingDirectory $Repo
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday, Wednesday, Thursday, Friday, Saturday -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "股票策略模擬器：每日 120 日動量訊號（$Ticker）" -Force | Out-Null

Write-Host "已建立排程「$TaskName」：每週二到週六 $Time 執行。"
Write-Host "電腦當時關機或睡眠的話，開機後會補跑一次。"
Write-Host "現在先試跑一次（最多等 90 秒）……"
$latest = Join-Path $Repo "results\signal_latest.txt"
Remove-Item $latest -ErrorAction SilentlyContinue
Start-ScheduledTask -TaskName $TaskName
for ($i = 0; $i -lt 90 -and -not (Test-Path $latest); $i++) { Start-Sleep -Seconds 1 }
if (Test-Path $latest) {
    Write-Host "試跑完成，結果如下（也存在 $latest）："
    Get-Content $latest -Encoding UTF8
} else {
    Write-Warning "還沒看到結果檔，可能還在下載報價；稍後打開 $latest 確認。"
}
