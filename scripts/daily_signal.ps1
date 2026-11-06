# 每日訊號：由 Windows 工作排程器呼叫（用 install_daily_task.ps1 設定），也可以手動執行。
# 部位一：$Ticker 的 120 日動量（轉弱時改持有 $Alt）；
# 部位二（選用，LevCapital > 0 時）：$LevTicker 的 200 日均線濾網（跌破時轉國庫券）。
# 把訊號結果存到 results\signals\<日期>.txt 與 results\signal_latest.txt；
# 任一訊號翻轉或執行失敗時跳出通知並打開結果檔。
param(
    [string]$Python = "python",
    [string]$Ticker = "QQQ",
    [string]$Alt = "GLD:0.5",
    [double]$Capital = 750,
    [string]$LevTicker = "QLD",
    [double]$LevCapital = 0,
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

function Invoke-Signal([string[]]$CliArgs) {
    $text = (& $Python @CliArgs 2>&1 | ForEach-Object { "$_" }) -join "`r`n"
    return @{ Text = $text; Code = $LASTEXITCODE }
}

$momArgs = @("-m", "simulator.cli", "signal", "-t", $Ticker, "--rule", "mom",
             "--capital", "$Capital", "--weights", "1.0")
# Alt 為 none 或空白：轉弱時全部放國庫券
if ($Alt -and $Alt -ne "none") { $momArgs += @("--alt", $Alt) }
$results = @([pscustomobject]@{ Name = "$Ticker 動量部位"; Run = (Invoke-Signal $momArgs) })

if ($LevCapital -gt 0) {
    $levArgs = @("-m", "simulator.cli", "signal", "-t", $LevTicker, "--rule", "sma",
                 "--capital", "$LevCapital", "--weights", "1.0")
    $results += [pscustomobject]@{ Name = "$LevTicker 均線部位"; Run = (Invoke-Signal $levArgs) }
}

$output = ($results | ForEach-Object { "##### " + $_.Name + " #####`r`n" + $_.Run.Text }) -join "`r`n`r`n"
$failed = @($results | Where-Object { $_.Run.Code -ne 0 } | ForEach-Object { $_.Name })
$flipped = @($results | Where-Object { $_.Run.Text -match "剛翻轉" } | ForEach-Object { $_.Name })

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

if ($failed.Count -gt 0) {
    Start-Process notepad.exe $Latest
    Show-Notice "訊號執行失敗" (($failed -join "、") + "：請打開 results\signal_latest.txt 查看錯誤訊息")
} elseif ($flipped.Count -gt 0) {
    Start-Process notepad.exe $Latest
    Show-Notice "訊號翻轉了！" (($flipped -join "、") + " 的訊號翻轉，詳見 results\signal_latest.txt")
} elseif ($AlwaysOpen) {
    Start-Process notepad.exe $Latest
}
