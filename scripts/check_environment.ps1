$ErrorActionPreference = 'Continue'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Write-Output "InfantPose project: $ProjectRoot"
Write-Output "--- Windows ---"
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsBuildNumber
Write-Output "--- NVIDIA ---"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi } else { Write-Warning 'nvidia-smi not found; install/update NVIDIA Windows driver.' }
Write-Output "--- WSL ---"
wsl.exe --status
wsl.exe --list --verbose
Write-Output "--- Python ---"
py.exe -0p
Write-Output "--- Git ---"
$Git = Get-Command git -ErrorAction SilentlyContinue
if ($Git) { git --version; git -C $ProjectRoot status --short --branch }
else { Write-Warning 'Git is not on PATH. Visual Studio Git may be available at the path printed in MIGRATION_REPORT.md.' }
Write-Output "--- Local Python packages ---"
py.exe -3.9 -c "import sys,importlib.util as u; print(sys.version); print({n:bool(u.find_spec(n)) for n in ['torch','tensorflow','mediapipe','cv2','jupyter','onnxruntime']})"
