import duckdb, re
c = duckdb.connect()
sql = open('oppgaver.sql').read()
for part in re.split(r'(?m)^\.print \'(.*)\'\n', sql):
    pass
parts = re.split(r"(?m)^\.print '(.*)'\n", sql)
c.execute(parts[0])
for title, q in zip(parts[1::2], parts[2::2]):
    print(f"\n{title}")
    c.sql(q).show(max_rows=15)
