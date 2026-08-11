# Run in Windows PowerShell (Admin optional).
# Adds PROCESS-NAME ssh rules + allow-lan for Clash/Mihomo, then tells you to reload.

$ErrorActionPreference = "Stop"

$candidates = @(
  "$env:USERPROFILE\.config\clash\config.yaml",
  "$env:USERPROFILE\.config\clash\profiles\*.yaml",
  "$env:USERPROFILE\.config\mihomo\config.yaml",
  "$env:USERPROFILE\.config\mihomo\profiles\*.yaml",
  "$env:USERPROFILE\.config\clash-verge\clash-verge.yaml",
  "$env:USERPROFILE\.config\clash-verge\profiles\*.yaml",
  "$env:USERPROFILE\.config\clash-verge-rev\config.yaml",
  "$env:APPDATA\io.github.clash-verge-rev.clash-verge-rev\config.yaml",
  "$env:APPDATA\clash\config.yaml",
  "$env:USERPROFILE\Documents\clash\config.yaml"
)

$files = @()
foreach ($p in $candidates) {
  $files += @(Get-Item -Path $p -ErrorAction SilentlyContinue)
}
$files = $files | Where-Object { $_ -and $_.Length -gt 0 } | Sort-Object FullName -Unique

if (-not $files) {
  Write-Host "ERROR: no Clash config.yaml found. Open your Clash client -> open config folder, then set `$cfg manually."
  exit 1
}

Write-Host "Found configs:"
$files | ForEach-Object { Write-Host " - $($_.FullName)" }

# Prefer the largest/most recently written profile-like yaml under .config
$cfg = $files | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Write-Host "`nWill edit: $($cfg.FullName)"

$bak = "$($cfg.FullName).bak_$(Get-Date -Format yyyyMMdd_HHmmss)"
Copy-Item $cfg.FullName $bak
Write-Host "Backup: $bak"

$text = Get-Content -Raw -Path $cfg.FullName

# allow-lan: true
if ($text -match '(?m)^allow-lan:\s*') {
  $text = [regex]::Replace($text, '(?m)^allow-lan:\s*.*$', 'allow-lan: true')
} else {
  $text = "allow-lan: true`r`n" + $text
}

$sshRules = @(
  "  - PROCESS-NAME,ssh.exe,PROXY",
  "  - PROCESS-NAME,sshd.exe,PROXY",
  "  - PROCESS-NAME,OpenSSH,PROXY"
) -join "`r`n"

if ($text -match '(?m)^rules:\s*$') {
  if ($text -notmatch 'PROCESS-NAME,ssh\.exe,PROXY') {
    $text = [regex]::Replace(
      $text,
      '(?m)^rules:\s*$',
      "rules:`r`n$sshRules"
    )
    Write-Host "Inserted PROCESS-NAME ssh rules under rules:"
  } else {
    Write-Host "ssh PROCESS-NAME rules already present."
  }
} else {
  Write-Host "WARN: no top-level 'rules:' found. Appending a rules block at end."
  $text = $text.TrimEnd() + "`r`n`r`nrules:`r`n$sshRules`r`n"
}

Set-Content -Path $cfg.FullName -Value $text -Encoding UTF8
Write-Host "`nDone. Next:"
Write-Host "1) Clash UI: Reload / 重启内核"
Write-Host "2) Confirm: curl.exe -x http://127.0.0.1:7890 -I https://www.google.com"
Write-Host "3) Cursor: disconnect lxb-portal and reconnect"
Write-Host "4) Remote Linux: curl -x http://127.0.0.1:7890 -I --connect-timeout 8 https://www.google.com"
