# Example cURL and Apache Bench commands

## Curl
curl -v -F image=@./requests/zidane.jpg http://localhost:8000/predict/image
curl -v -F tensor=@./requests/zidane.npy http://localhost:8000/predict/tensor

## Apache Bench
We use the "create_ab_multipart_request.sh" script to generate payloads for Apache Bench.

For images
./create_ab_multipart_request.sh 1234567890 image zidane.jpg
ab -n 1 -v 4 -T "multipart/form-data; boundary=1234567890" -p ./zidane.jpg.request http://localhost:8000/predict/image
ab -n 1000 -c 8 -T "multipart/form-data; boundary=1234567890" -p ./requests/zidane.jpg.request http://localhost:8000/predict/image

For Tensors
./create_ab_multipart_request.sh 1234567890 tensor zidane.ndarray
ab -n 1 -v 4 -T "multipart/form-data; boundary=1234567890" -p ./zidane.ndarray.request http://localhost:8000/predict/tensor
ab -n 1000 -c 8 -T "multipart/form-data; boundary=1234567890" -p ./requests/zidane.ndarray.request http://localhost:8000/predict/tensor

# Single Replica Tests
MODEL_INTRAOP_THREADS=0 uvicorn servers.yolov5:app --log-level critical
ab -n 1000 -c 1 -T "multipart/form-data; boundary=1234567890" -p ./requests/zidane.jpg.request -e benchmarks/image/c6i4xl.r1.t0.c1.csv http://localhost:8000/predict/image > benchmarks/image/c6i4xl.r1.t0.c1.txt
ab -n 1000 -c 4 -T "multipart/form-data; boundary=1234567890" -p ./requests/zidane.jpg.request -e benchmarks/image/c6i4xl.r1.t0.c4.csv http://localhost:8000/predict/image > benchmarks/image/c6i4xl.r1.t0.c4.txt
ab -n 1000 -c 16 -T "multipart/form-data; boundary=1234567890" -p ./requests/zidane.jpg.request -e benchmarks/image/c6i4xl.r1.t0.c16.csv http://localhost:8000/predict/image > benchmarks/image/c6i4xl.r1.t0.c16.txt

MODEL_INTRAOP_THREADS=4 uvicorn servers.yolov5:app --log-level critical
ab -n 1000 -c 16 -T "multipart/form-data; boundary=1234567890" -p ./requests/zidane.jpg.request -e benchmarks/image/c6i4xl.r1.t4.c16.csv http://localhost:8000/predict/image > benchmarks/image/c6i4xl.r1.t4.c16.txt

# Multi Replica Tests

MODEL_INTRAOP_THREADS=0 gunicorn servers.yolov5:app -w 16 -k uvicorn.workers.UvicornWorker
ab -n 1000 -c 16 -T "multipart/form-data; boundary=1234567890" -p ./requests/zidane.jpg.request -e benchmarks/image/c6i4xl.r16.t0.c16.csv http://localhost:8000/predict/image > benchmarks/image/c6i4xl.r16.t0.c16.txt

MODEL_INTRAOP_THREADS=1 gunicorn servers.yolov5:app -w 16 -k uvicorn.workers.UvicornWorker
ab -n 1000 -c 16 -T "multipart/form-data; boundary=1234567890" -p ./requests/zidane.jpg.request -e benchmarks/image/c6i4xl.r16.t1.c16.csv http://localhost:8000/predict/image > benchmarks/image/c6i4xl.r16.t1.c16.txt