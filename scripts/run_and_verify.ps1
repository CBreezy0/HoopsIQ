$py = "C:\Users\chris\OneDrive\NCAAB_Rankings\.venv\Scripts\python.exe"

& $py C:\Users\chris\OneDrive\NCAAB_Rankings\ncaab_ranker.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py C:\Users\chris\OneDrive\NCAAB_Rankings\ncaab_ranker.py --verify
exit $LASTEXITCODE