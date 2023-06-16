import concurrent.futures
import os
from functools import wraps

def parallelize(func):
    @wraps(func)
    def wrapper(lst):
        number_of_threads_multiple = 2 # You can change this multiple according to you requirement
        number_of_workers = int(os.cpu_count() * number_of_threads_multiple)
        if len(lst) < number_of_workers:
            number_of_workers = len(lst)

        if number_of_workers:
            if number_of_workers == 1:
                result = [func(lst[0])]
            else:
                result = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_workers) as executer:
                    bag = {executer.submit(func, i): i for i in lst}
                    for future in concurrent.futures.as_completed(bag):
                        result.append(future.result())
        else:
            result = []
        return result
    return wrapper
