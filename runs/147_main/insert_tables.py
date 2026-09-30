"""make_tables.py 출력으로 결과 문서의 TABLES 구간을 갈아 끼운다."""
import subprocess, re
t = subprocess.run(['python3', '147_main/make_tables.py'], capture_output=True, text=True, check=True).stdout
p = '../docs/PROMPTCAL_LOWBIT_RESULTS_2026-09-30.md'
s = open(p).read()
s = re.sub(r'<!-- TABLES:BEGIN -->.*<!-- TABLES:END -->', '<!-- TABLES:BEGIN -->\n' + t.strip('\n') + '\n<!-- TABLES:END -->', s, flags=re.S)
open(p, 'w').write(s)
print('ok', t.count('\n'))
