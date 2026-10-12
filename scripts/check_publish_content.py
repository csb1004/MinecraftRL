"""Inspect exactly the indexed files without printing possible secret values."""
import hashlib,json,re,subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode('utf-8').split('\0')
names=[n for n in names if n]
if not names:raise SystemExit('Index is empty')
patterns=[r'\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}',r'\bgh[pousr]_[A-Za-z0-9]{30,}',
    r'\bgithub_pat_[A-Za-z0-9_]{30,}',r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    r'\bAKIA[0-9A-Z]{16}\b']
forbidden={'.runtime','.cache','external','weights','data','logs','.venv'}
bad=[];files=[]
for name in names:
    p=Path(name)
    if (any(part in forbidden for part in p.parts) or
        (p.parts[0]=='results' and name!='results/.gitkeep') or
        (p.name.startswith('.env') and p.name!='.env.example') or
        p.suffix.lower() in {'.pt','.pth','.ckpt','.weights','.model','.zip','.pkl','.mp4','.pem','.key'}):
        bad.append(dict(file=name,reason='excluded asset or secret filename'))
    content=subprocess.check_output(['git','show',':'+name],cwd=root)
    if len(content)>5*1024*1024:bad.append(dict(file=name,reason='file exceeds publication size cap'))
    if b'\0' not in content:
        text=content.decode('utf-8',errors='replace')
        for number,line in enumerate(text.splitlines(),1):
            if any(re.search(pattern,line) for pattern in patterns):bad.append(dict(file=name,line=number,reason='potential credential'))
    files.append(dict(path=name,bytes=len(content),sha256=hashlib.sha256(content).hexdigest()))
out=root/'results/blind-recovery-20261012/publication-scan.json'
out.write_text(json.dumps(dict(passed=not bad,file_count=len(files),total_bytes=sum(f['bytes'] for f in files),
    findings=bad,files=files,scope='Indexed paths, file sizes and common credential patterns; manual review also required'),indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(passed=not bad,file_count=len(files),total_bytes=sum(f['bytes'] for f in files),findings=bad)))
raise SystemExit(1 if bad else 0)
