# Deploying ML Models with Kubernetes

*patcharanat p.*

```bash
# install python dependency
pip install -r requirements.txt

uv init

uv add fastapi uvicorn onnxruntime keras-image-helper numpy

# OR just
uv sync

# run powershell as administrator
choco install kind

# download model
wget https://github.com/DataTalksClub/machine-learning-zoomcamp/releases/download/dl-models/clothing_classifier_mobilenet_v2_latest.onnx -O clothing-model.onnx

# run fastapi locally to test
uv run uvicorn app:app --host 0.0.0.0 --port 8080 --reload

# test application
curl http://localhost:8080/health

# run service within a docker
docker build -t clothing-classifier:v1 .

docker run -it --rm -p 8080:8080 clothing-classier:v1

# k8s
cd k8s

# start k8s cluster
kind create cluster --name mlzoomcamp

# verify k8s cluster
kubectl cluster-info
kubectl get nodes

# Kind clusters run in Docker, so they can't access images from your local Docker daemon by default. We need to load the image into Kind.
kind load docker-image clothing-classifier:v1 --name mlzoomcamp

# without `kind load` step, pod can be spin-up, but will not be available

kubectl apply -f deployment.yaml
# kubectl delete -f deployment.yaml

# verify deployments
kubectl get pods
kubectl get deployments

# add Load Balancer Service
kubectl apply -f service.yaml

# verify service
kubectl get services

# forward port to local machine
kubectl port-forward service/clothing-classifier 30080:8080

# verify Load Balancer
curl localhost:30080/health

# apply metrics server to monitor cpu usage
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml

# patch metrics server to work without TLS (HTTPS)
kubectl patch -n kube-system deployment metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'

# apply HPA
kubectl apply -f hpa.yaml

# load testing to test HPA
uv run python load_test.py

# monitor scaling and resource usage during load testing
kubectl get hpa

# clean up
kubectl delete -f deployment.yaml
kubectl delete -f service.yaml
kubectl delete -f hpa.yaml

kubectl delete all -l app=clothing-classifier

kind delete cluster --name mlzoomcamp
```

**Additional Information**
- Where to store ML model
    1. Put the model in Dockerfile making it easier to rollback and scaling up.
    2. Download the model from S3, for example, when prediction service start to minimize docker image size, but it also requires downloading the model for every pod when we talk about scaling (slower).
- For k8s, we need
    - Nodes -> e.g. EC2, 1 machine
    - Pods -> 1 application container(s)
    - Deployments -> use to specify specification of Pods to be created
    - Services -> Load Balancer
    - HPA -> Horizontal Pod Autoscaler (help K8s to scaling up horizontally)
- We can't know which port is available for us to send the request, so that's why Load balancer can help us in this as k8s service.
- Metrics Server is required for HPA in order to monitor usage resource to trigger scaling.