##############################################################
# COMMIT FILE FOR LOCAL TO BIGQUERY
# File name: commit_bq.py

from google.cloud import bigquery
from google.oauth2 import service_account
import time

# GET OPTIONS
import sys, getopt  # for getting option for cmd

try:
    opts, args = getopt.getopt(sys.argv[1:], 's:d',
                               ['source=', 'destination='])

except getopt.GetoptError as err:
    print("ERROR: Getoption gets error... please check!\n {}", err)
    sys.exit(1)

for opt, arg in opts:
    if opt in ('--source'):
        source = str(arg)

    if opt in ('--destination'):
        dest = str(arg)

    if opt in ('--empty'):
        empty = str(arg)
        if empty.lower() == 'true':
            empty = True
        if empty.lower() == 'false':
            empty = False


class Commit(object):
    def __init__(self, source, dest, empty=True):
        # default variables
        self.source = source
        self.dest = dest
        self.empty = empty

        # setup client  
        self.client = bigquery.Client()
		
    def main(self):
        # upload to BigQuery
        print("Dataset:", self.dest.split('.')[0])
        print("Table:", self.dest.split('.')[1])
        table_ref = self.client.dataset(self.dest.split('.')[0]).table(self.dest.split('.')[1])
        job_config = bigquery.LoadJobConfig()
        job_config.source_format = bigquery.SourceFormat.CSV
        job_config.skip_leading_rows = 1  # ignore the header
        job_config.autodetect = True
        t1 = time.time()

        if self.empty:
            # DROP ---------
            sql_drop = ('DROP TABLE IF EXISTS ' + self.dest)
            print("[INFO] SQL sql_drop: \n {}\n".format(sql_drop))
            sandbox_tb_drop = self.client.query(sql_drop)
            rows = sandbox_tb_drop.result()  # waiting complete
            time.sleep(10)
            print("Drop total rows", rows )

        with open(self.source, "rb") as source_file:
            job = self.client.load_table_from_file(
                source_file, table_ref, job_config=job_config
            )

        # job is async operation so we have to wait for it to finish
        job.result()
        print("IMPORT SUCCESSFUL", time.time() - t1)


if __name__ == "__main__":
    Commit(source=source, dest=dest, empty=True).main()
