# Set the root directory — you can change this to wherever you want to start
$rootDir = "."

# Recursively find all .hpp files
Get-ChildItem -Path $rootDir -Recurse -Filter *.hpp| ForEach-Object {
    $oldPath = $_.FullName
    $newPath = [System.IO.Path]::ChangeExtension($oldPath, ".hpp")

    # Only rename if target doesn't already exist
    if (-not (Test-Path -Path $newPath)) {
        Rename-Item -Path $oldPath -NewName ($_.BaseName + ".hpp")
        Write-Host "Renamed: $oldPath -> $newPath"
    } else {
        Write-Host "Skipped (target exists): $newPath"
    }
}
