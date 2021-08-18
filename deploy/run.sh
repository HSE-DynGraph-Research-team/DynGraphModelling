docker build . --build-arg SSH_PRIVATE_KEY="$(cat repo-key)" -t 'graph'
docker volume create log_vol
docker container prune -f
docker run -it --gpus all --name training --mount source=log_vol,target=/app/DynGraph-modelling/logs graph
tar czf /resuls.tar.gz /var/lib/docker/volumes/log_vol/_data