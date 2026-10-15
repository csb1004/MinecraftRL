from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(shutil.which('pwsh'), 'PowerShell unavailable on this executor')
class ArtifactPowerShellTests(unittest.TestCase):
    def test_helper_contracts_without_network_or_credentials(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            script = Path(temporary)/'test.ps1'
            script.write_text(r'''
param([string]$Helpers,[string]$Target)
$ErrorActionPreference='Stop'
. $Helpers
$headers=Get-ArtifactHeaders
if ($headers.ContainsKey('Authorization')) { throw 'Public access acquired credentials' }
$sha='a'*64
$asset=[pscustomobject]@{state='uploaded';size=123;digest="sha256:$sha"}
Assert-ArtifactAsset -Asset $asset -Sha256 $sha -Bytes 123
$rejected=$false
try { Assert-ArtifactAsset -Asset $asset -Sha256 ('b'*64) -Bytes 123 } catch { $rejected=$true }
if (-not $rejected) { throw 'Wrong server digest accepted' }
$cache=Get-ArtifactCache -Root $Target
if (-not (Test-Path -LiteralPath $cache -PathType Container)) { throw 'Cache missing' }
$metadata=@($asset)
if ($metadata.Count -ne 1 -or $metadata[0].digest -notmatch '^sha256:([0-9a-f]{64})$') { throw 'Digest parse failed' }
if ($matches[1] -cne $sha) { throw 'Digest extraction mismatch' }
$script:calls=0
function Invoke-RestMethod {
    param($Uri,$Headers)
    $script:calls++
    if ($Uri -like '*/git/ref/tags/*') { return @{object=@{type='tag';sha='tag-object'}} }
    if ($Uri -like '*/git/tags/*') { return @{object=@{type='commit';sha=('c'*40)}} }
    if ($Uri -like '*page=1') { return @(@{tag_name='run-1';id=1;draft=$true}) }
    throw 'Unexpected request'
}
$release=Get-ArtifactRelease -Repository 'owner/repo' -Tag 'run-1' -Headers $headers
if (-not $release.draft) { throw 'Draft resumption lookup failed' }
if ((Get-ArtifactTagCommit -Repository 'owner/repo' -Tag 'run-1' -Headers $headers) -cne ('c'*40)) { throw 'Annotated tag resolution failed' }
if ($script:calls -ne 3) { throw 'Unexpected network call count' }
Write-Output 'contracts passed; no real network or credential access'
''', encoding='utf-8')
            result = subprocess.run(['pwsh','-NoProfile','-File',str(script),
                '-Helpers',str(root/'scripts/GitHub-ArtifactHelpers.ps1'),'-Target',temporary],
                capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)


if __name__ == '__main__': unittest.main()
