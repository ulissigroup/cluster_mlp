import functools

files_list = ["deap_ga.py", "fillPool.py", "mutations.py", "utils.py"]

def call_dask(client):
    for i in range(len(files_list)):
        fname = files_list[i]
        with open(fname, "rb") as f:
            data = f.read()
    
        def _worker_upload(dask_worker, *, data, fname):
            dask_worker.loop.add_callback(
                callback=dask_worker.upload_file,
                comm=None,  # not used
                filename=fname,
                data=data,
                load=True,
            )
    
        client.register_worker_callbacks(
            setup=functools.partial(
                _worker_upload,
                data=data,
                fname=fname,
            )
        )
