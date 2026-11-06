# 移除每日訊號排程
param([string]$TaskName = "StockStrategySignal")
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Write-Host "已移除排程「$TaskName」。"
