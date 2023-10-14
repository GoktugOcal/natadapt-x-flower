import psutil
from datetime import datetime

filename = "cpu_logs_test-1.txt"

file = open(filename, "w")
no_cores = psutil.cpu_count()
header = "datetime;"+";".join([str(i) for i in range(1,no_cores+1)])
file.write(header)
file.close()

file = open(filename, "a")
while True:
    print("hop")
    per_cpu = psutil.cpu_percent(percpu=True, interval=3)
    now = datetime.now()
    file.write(f"\n{now.strftime('%m/%d/%Y %H:%M:%S')};"+";".join([str(usage) for usage in per_cpu]))
    #for idx, usage in enumerate(per_cpu):
        #print(f"CORE {idx+1}: {usage}%")