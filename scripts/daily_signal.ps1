# 每日訊號：由 Windows 工作排程器呼叫（用 install_daily_task.ps1 設定），也可以手動執行。
# 把訊號結果存到 results\signals\<日期>.txt 與 results\signal_latest.txt；
# 訊號翻轉或執行失敗時跳出通知並打開結果檔。
param(
    [string]$Python = "python",
    [string]$Ticker = "QQQ",
    [string]$Alt = "GLD:0.5",
    [double]$Capital = 750,
    [switch]$AlwaysOpen
)

$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
# Python 固定輸出 UTF-8，PowerShell 也用 UTF-8 解讀，避免中文在 Big5 編碼下變成亂碼
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$LogDir = Join-Path $Repo "results\signals"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogFile = Join-Path $LogDir ((Get-Date -Format "yyyy-MM-dd") + ".txt")
$Latest = Join-Path $Repo "results\signal_latest.txt"

$cliArgs = @("-m", "simulator.cli", "signal", "-t", $Ticker, "--rule", "mom",
             "--capital", "$Capital", "--weights", "1.0")
# Alt 為 none 或空白：轉弱時全部放國庫券
if ($Alt -and $Alt -ne "none") { $cliArgs += @("--alt", $Alt) }

$output = (& $Python @cliArgs 2>&1 | ForEach-Object { "$_" }) -join "`r`n"
$exitCode = $LASTEXITCODE

$header = "=== 執行時間 " + (Get-Date -Format "yyyy-MM-dd HH:mm") + " ===`r`n"
Set-Content -Path $LogFile -Value ($header + $output) -Encoding UTF8
Copy-Item -Path $LogFile -Destination $Latest -Force

function Show-Notice([string]$Title, [string]$Text) {
    try {
        Add-Type -AssemblyName System.Windows.Forms
        Add-Type -AssemblyName System.Drawing
        $icon = New-Object System.Windows.Forms.NotifyIcon
        $icon.Icon = [System.Drawing.SystemIcons]::Information
        $icon.BalloonTipTitle = $Title
        $icon.BalloonTipText = $Text
        $icon.Visible = $true
        $icon.ShowBalloonTip(15000)
        Start-Sleep -Seconds 16
        $icon.Dispose()
    } catch { }
}

$signalLine = ($output -split "`r?`n" | Where-Object { $_ -match "^訊號" } | Select-Object -First 1)
if ($signalLine) { $signalLine = $signalLine.Trim() }

if ($exitCode -ne 0) {
    Start-Process notepad.exe $Latest
    Show-Notice "訊號執行失敗" "請打開 results\signal_latest.txt 查看錯誤訊息"
} elseif ($output -match "剛翻轉") {
    Start-Process notepad.exe $Latest
    Show-Notice "$Ticker 訊號翻轉了！" "$signalLine（詳見 results\signal_latest.txt）"
} elseif ($AlwaysOpen) {
    Start-Process notepad.exe $Latest
}
