kubectl apply -f download-dw-config.yaml 

argo submit download_dynamic_world.yaml -n argo -p years="2025" -p months="1,2,3"
