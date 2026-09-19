param(
    [string]$OutDir,
    [string]$Url = "https://hf-mirror.com/BAAI/bge-large-zh-v1.5/resolve/main/pytorch_model.bin"
)

if (-not $OutDir) {
    $projectRoot = Split-Path -Parent $PSScriptRoot
    $OutDir = Join-Path $projectRoot "models\bge-large-zh-v1.5"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$out = Join-Path $OutDir "pytorch_model.bin"

$headers = curl.exe -sIL --max-time 30 $Url
$lengths = @(
    $headers |
    Select-String -Pattern "^content-length:\s*(\d+)" |
    ForEach-Object { [int64]$_.Matches[0].Groups[1].Value }
)
$expected = $lengths | Select-Object -Last 1
Write-Output "out=$out"
Write-Output "expected=$expected"

$attempt = 0
while ($true) {
    $have = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
    if ($expected -gt 0 -and $have -ge $expected) { break }
    $attempt++
    Write-Output ("attempt={0} have={1:N1}MB" -f $attempt, ($have / 1MB))
    curl.exe -L -C - --retry 3 --retry-delay 3 --retry-all-errors -o $out $Url
    Start-Sleep -Seconds 2
}

Write-Output ("DONE size={0}" -f (Get-Item $out).Length)
