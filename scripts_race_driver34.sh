#!/bin/bash
# attend que le train de driver2 soit fini puis lance driver3 et driver4 à la suite
cd "/Users/mathiasguignet/Documents/Epitech/Master/Crash and learn" || exit 1

# driver2 a démarré
while [ ! -f /tmp/train_driver2_oa.log ]; do sleep 5; done
# plus aucun container -> driver2 fini
while docker ps --filter "name=crashandlearn" --format "{{.Names}}" | grep -q .; do sleep 5; done

echo "driver2 done, launching driver3" >> /tmp/chain_driver34.log

docker compose run --rm -e PYTHONUNBUFFERED=1 sim python -u train_champion.py \
  --resume /app/logs_driver_oa/driver3_init.zip \
  --opponent-model /app/submission/driver1/model.onnx \
  --timesteps 500000 --eval-freq 125000 --n-envs 4 --gamma 0.999 --ent-coef 0.005 \
  --wall-proximity-margin 0.4 --reward-wall-proximity-scale 1.0 \
  --logdir /app/logs_driver_oa/driver3 --out /app/driver3_oa_model.onnx \
  > /tmp/train_driver3_oa.log 2>&1

echo "driver3 done, launching driver4" >> /tmp/chain_driver34.log

docker compose run --rm -e PYTHONUNBUFFERED=1 sim python -u train_champion.py \
  --resume /app/logs_driver_oa/driver4_init.zip \
  --opponent-model /app/submission/driver1/model.onnx \
  --timesteps 500000 --eval-freq 125000 --n-envs 4 --gamma 0.999 --ent-coef 0.005 \
  --reward-steering-jerk-scale 0.04 \
  --logdir /app/logs_driver_oa/driver4 --out /app/driver4_oa_model.onnx \
  > /tmp/train_driver4_oa.log 2>&1

echo "driver4 done, chain complete" >> /tmp/chain_driver34.log
