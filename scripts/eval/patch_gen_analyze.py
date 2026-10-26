import sys
p = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-tmp/analysis/prep_gen.py'
s = open(p).read()
old = "        g, t = graded(f'{root}/{setname}'), graded(testdir)\n"
new = ("        tds = sorted(d for pat in testdir.split(',') for d in glob.glob(pat) if os.path.isdir(d))\n"
       "        assert tds, f'no test run dirs match {testdir}'\n"
       "        tl = [graded(d) for d in tds]\n"
       "        g, t = graded(f'{root}/{setname}'), {q: float(np.mean([x[q] for x in tl])) for q in tl[0]}   # original = mean over test seeds\n")
assert s.count(old) == 1, 'patch target not found'
s = s.replace(old, new).replace("print(f'\\n==== {setname}: accuracy % [95% CI] and paired change vs the same model on the original test question (pp)')",
  "print(f'\\n==== {setname}: accuracy % [95% CI]; orig = same model on the original test question, averaged over its test seeds; change = acc - orig (pp)')")
compile(s, p, 'exec'); open(p, 'w').write(s); print('prep_gen.py analyze: original accuracy now averaged over all matching test seeds')
