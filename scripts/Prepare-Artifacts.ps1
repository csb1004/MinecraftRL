#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$Spec,
    [Parameter(Mandatory)][string]$OutputDirectory,
    [string]$PreviousManifest
)
$ErrorActionPreference='Stop'
$root=Split-Path $PSScriptRoot -Parent
$arguments=@((Join-Path $root 'src/minecraftrl/artifact_release.py'),'prepare','--root',$root,'--spec',$Spec,'--output',$OutputDirectory)
if ($PreviousManifest) { $arguments+=@('--previous',$PreviousManifest) }
& python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Artifact preparation failed. Original source files are unchanged.' }
