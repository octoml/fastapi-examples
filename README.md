### Example cURL and Apache Bench commands

curl -v -F model_name=yolov5 -F image_file=@../images/zidane.jpg http://localhost:8000/predict

ab -n 1 -v 4 -T "multipart/form-data; boundary=123456789" -p ./requests/zidane.ab.request http://localhost:8000/predict
ab -n 1000 -c 8 -T "multipart/form-data; boundary=123456789" -p ./requests/zidane.ab.request http://localhost:8000/predict


### Single Replica Tests
MODEL_INTRAOP_THREADS=0 uvicorn servers.yolov5:app --log-level critical

ab -n 1000 -c 1 -T "multipart/form-data; boundary=123456789" -p ./requests/zidane.ab.request -e benchmarks/c6i4xl.r1.t0.c1.csv http://localhost:8000/predict > benchmarks/c6i4xl.r1.t0.c1.txt

ab -n 1000 -c 4 -T "multipart/form-data; boundary=123456789" -p ./requests/zidane.ab.request -e benchmarks/c6i4xl.r1.t0.c4.csv http://localhost:8000/predict > benchmarks/c6i4xl.r1.t0.c4.txt

ab -n 1000 -c 16 -T "multipart/form-data; boundary=123456789" -p ./requests/zidane.ab.request -e benchmarks/c6i4xl.r1.t0.c16.csv http://localhost:8000/predict > benchmarks/c6i4xl.r1.t0.c16.txt

MODEL_INTRAOP_THREADS=4 uvicorn servers.yolov5:app --log-level critical

ab -n 1000 -c 16 -T "multipart/form-data; boundary=123456789" -p ./requests/zidane.ab.request -e benchmarks/c6i4xl.r1.t4.c16.csv http://localhost:8000/predict > benchmarks/c6i4xl.r1.t4.c16.txt

### Multi Replica Tests

MODEL_INTRAOP_THREADS=0 gunicorn servers.yolov5:app -w 16 -k uvicorn.workers.UvicornWorker

ab -n 1000 -c 16 -T "multipart/form-data; boundary=123456789" -p ./requests/zidane.ab.request -e benchmarks/c6i4xl.r16.t0.c16.csv http://localhost:8000/predict > benchmarks/c6i4xl.r16.t0.c16.txt

MODEL_INTRAOP_THREADS=1 gunicorn servers.yolov5:app -w 16 -k uvicorn.workers.UvicornWorker

ab -n 1000 -c 16 -T "multipart/form-data; boundary=123456789" -p ./requests/zidane.ab.request -e benchmarks/c6i4xl.r16.t1.c16.csv http://localhost:8000/predict > benchmarks/c6i4xl.r16.t1.c16.txt