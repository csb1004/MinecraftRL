[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$ImagePath,[Parameter(Mandatory=$true)][string]$OutputPath,[string]$Language='en-US')
$ErrorActionPreference='Stop'
if (Test-Path -LiteralPath $OutputPath) { throw 'OCR output already exists.' }
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$null=[Windows.Storage.StorageFile,Windows.Storage,ContentType=WindowsRuntime]
$null=[Windows.Storage.Streams.IRandomAccessStream,Windows.Storage.Streams,ContentType=WindowsRuntime]
$null=[Windows.Graphics.Imaging.BitmapDecoder,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$null=[Windows.Graphics.Imaging.SoftwareBitmap,Windows.Graphics.Imaging,ContentType=WindowsRuntime]
$null=[Windows.Media.Ocr.OcrEngine,Windows.Foundation,ContentType=WindowsRuntime]
$null=[Windows.Media.Ocr.OcrResult,Windows.Foundation,ContentType=WindowsRuntime]
$null=[Windows.Globalization.Language,Windows.Globalization,ContentType=WindowsRuntime]
$asTask=[System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.IsGenericMethod -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' } | Select-Object -First 1
function Await-Result($Operation,[Type]$ResultType) {
    $task=$asTask.MakeGenericMethod($ResultType).Invoke($null,@($Operation))
    $task.Wait()
    $task.Result
}
$absolute=(Resolve-Path -LiteralPath $ImagePath).Path
$file=Await-Result ([Windows.Storage.StorageFile]::GetFileFromPathAsync($absolute)) ([Windows.Storage.StorageFile])
$stream=Await-Result ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
try {
    $decoder=Await-Result ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap=Await-Result ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    try {
        $engine=[Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage((New-Object Windows.Globalization.Language($Language)))
        if ($null -eq $engine) { throw "Requested OCR language is not installed: $Language" }
        $result=Await-Result ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
        $record=[ordered]@{source='f3_ocr'; engine='Windows.Media.Ocr'; language=$engine.RecognizerLanguage.LanguageTag; image_sha256=(Get-FileHash -LiteralPath $absolute -Algorithm SHA256).Hash.ToLowerInvariant(); lines=@($result.Lines | ForEach-Object {$_.Text}); text=(@($result.Lines | ForEach-Object {$_.Text}) -join "`n")}
        [System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($OutputPath),($record | ConvertTo-Json -Depth 5),(New-Object System.Text.UTF8Encoding($false)))
    } finally { $bitmap.Dispose() }
} finally { $stream.Dispose() }
