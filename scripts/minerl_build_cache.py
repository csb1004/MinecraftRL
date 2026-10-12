"""Generate a local cache guard around a fully successful upstream MCP build."""

def patch_setup(source):
    old = '    prep_mcp()\n'
    if source.count(old) != 1:
        raise RuntimeError('Unexpected MineRL prep_mcp call site')
    return source.replace(old, '''    import hashlib
    artifact = pathlib.Path(__file__).parent / 'minerl/MCP-Reborn/build/libs/mcprec-6.13.jar'
    receipt = pathlib.Path(__file__).parent / '.minecraftrl-mcp-built.json'
    digest = lambda: hashlib.sha256(artifact.read_bytes()).hexdigest()
    if receipt.exists():
        if not artifact.is_file() or digest() != json.loads(receipt.read_text())['sha256']:
            raise RuntimeError('Built Minecraft artifact changed; refusing reuse')
        print('Reusing verified Minecraft build artifact')
    else:
        prep_mcp()
        import zipfile
        with zipfile.ZipFile(artifact) as jar:
            if jar.testzip() is not None or 'net/minecraft/client/main/Main.class' not in jar.namelist():
                raise RuntimeError('Incomplete Minecraft build artifact')
        receipt.write_text(json.dumps({'sha256': digest()}))
''')
