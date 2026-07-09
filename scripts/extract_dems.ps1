# Extract all .dem files from HLTV .rar/.zip archives organised in
# <event>/<stage>/ folders, renaming each to <event>__<stage>__<match>.dem
# Run from anywhere (absolute paths); requires 7-Zip or WinRAR.
$src  = "D:\TRASH\Ultra\dem files zipped"
$dest = "D:\TRASH\Ultra\raw_dems"
$tmp  = "D:\TRASH\Ultra\_extract_tmp"

$sevenZip = @("$env:ProgramFiles\7-Zip\7z.exe",
              "${env:ProgramFiles(x86)}\7-Zip\7z.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
$unrar    = @("$env:ProgramFiles\WinRAR\UnRAR.exe",
              "${env:ProgramFiles(x86)}\WinRAR\UnRAR.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $sevenZip -and -not $unrar) {
    throw "Neither 7-Zip nor WinRAR found. Install one, e.g.:  winget install 7zip.7zip"
}
New-Item -ItemType Directory -Force -Path $dest | Out-Null
function Slug([string]$s) { ($s.ToLower() -replace "[^a-z0-9]+", "-").Trim("-") }

$rars = Get-ChildItem -Path $src -Recurse -Include *.rar, *.zip, *.7z -File
Write-Host "Found $($rars.Count) archive(s)"
foreach ($rar in $rars) {
    $rel   = $rar.DirectoryName.Substring($src.Length).Trim('\')
    $parts = $rel -split '\\'
    $eventSlug = if ($parts[0]) { Slug $parts[0] } else { "misc" }
    $stageSlug = if ($parts.Count -gt 1) { Slug ($parts[1..($parts.Count-1)] -join ' ') } else { "main" }
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    Write-Host "extracting: $($rar.FullName)"
    if ($sevenZip) { & $sevenZip x $rar.FullName "-o$tmp" -y | Out-Null }
    else           { & $unrar x -y $rar.FullName "$tmp\"    | Out-Null }
    if ($LASTEXITCODE -ne 0) { Write-Warning "  extract FAILED, skipping"; continue }
    $dems = Get-ChildItem -Path $tmp -Recurse -Filter *.dem -File
    if ($dems.Count -eq 0) { Write-Warning "  no .dem inside $($rar.Name)" }
    foreach ($dem in $dems) {
        $demSlug = Slug ([IO.Path]::GetFileNameWithoutExtension($dem.Name))
        $base    = "${eventSlug}__${stageSlug}__${demSlug}"
        $target  = Join-Path $dest "$base.dem"
        $k = 2
        while (Test-Path $target) { $target = Join-Path $dest "$base-$k.dem"; $k++ }
        Move-Item $dem.FullName $target
        Write-Host ("  -> " + (Split-Path $target -Leaf))
    }
}
Remove-Item $tmp -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Done. $((Get-ChildItem $dest -Filter *.dem).Count) .dem file(s) in $dest"
