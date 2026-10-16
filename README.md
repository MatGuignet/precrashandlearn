# Crash & Learn — Grand Prix

Entraînement d'agents RL pour piloter un simulateur de course inspiré de F1TENTH
(moteur `f110_gym`). Le projet est structuré en 4 parties :

1. **Solo Driving Licence** — un agent PPO apprend à boucler seul sur 5 circuits.
2. **Grand Crash Auto** — diagnostic : pourquoi un agent entraîné seul ne tient pas
   un circuit partagé avec d'autres voitures.
3. **Multi-agent Champion** — un agent entraîné face à des adversaires (politique
   figée), via transfert de poids depuis le modèle solo.
4. **Final Grand Prix** — 4 pilotes différenciés (personnalités distinctes), rendus
   également "opponent-aware" par la même technique de transfert.

Le détail des choix de conception, bugs trouvés/corrigés et résultats est dans
[REPORT.md](REPORT.md).

## Prérequis

- Docker Desktop (le daemon doit tourner — `docker ps` ne doit pas renvoyer d'erreur)
- Construire l'image une première fois :
  ```bash
  docker compose build
  ```

## Lancer le projet

```bash
# Smoke test rapide (demo.py)
docker compose run --rm sim

# Entraîner le modèle solo (Part 1)
docker compose run --rm sim python train.py --timesteps 3000000

# Entraîner le champion multi-agent (Part 3), depuis un modèle solo transféré
docker compose run --rm sim python transfer_champion_init.py \
    --solo-model logs_weather/best_model.zip \
    --opponent-model submission/driver1/model.onnx \
    --out logs_champion/transferred_init
docker compose run --rm sim python train_champion.py \
    --resume logs_champion/transferred_init.zip \
    --opponent-model submission/driver1/model.onnx \
    --timesteps 2000000

# Suivre une courbe d'entraînement (TensorBoard)
docker compose up tensorboard   # http://localhost:6006
```

## Tester le projet

| # | Quoi | Commande |
|---|---|---|
| 1 | Tests du moteur fourni (contrat + intégration) | `docker compose run --rm sim python -m pytest test_contract.py test_integration.py -v` |
| 2 | Un seul agent, comme en tournoi | `docker compose run --rm sim python sim_recorder.py --submissions submission/driver1 --num-cars 1 --steps 6000 --laps 2` |
| 3 | Les 4 pilotes ensemble, stats agrégées sur les 5 circuits (Part 2 "Grand Crash Auto") | `docker compose run --rm sim python race_drivers.py --minutes 10` |
| 4 | N'importe quelle autre combinaison, stats agrégées (ex. champion + 3 pilotes) | `docker compose run --rm sim python race_drivers.py --minutes 10 --drivers champion,driver1,driver2,driver3` |
| 5 | Enregistrer UNE course multi-voitures précise, pour la rejouer ensuite | `docker compose run --rm sim python sim_recorder.py --drivers driver1,driver2,driver3,driver4 --steps 6000 --laps 2` |
| 6 | Visualiser la course enregistrée (animée, interactive) | voir section ci-dessous |

`race_drivers.py` (#3/#4) sert à comparer des pilotes sur beaucoup d'épisodes (pas
de replay visuel). `sim_recorder.py --drivers ...` (#5) sert à capturer **une**
course précise en `.npz` pour la rejouer avec `viz_replay.py`. Les deux acceptent
le même flag `--drivers` (liste de sous-dossiers de `submission/`, dans l'ordre,
max 4 voitures — limite du moteur) et chargent explicitement les noms demandés,
contrairement à `sim_recorder.py --submissions` seul, qui trie les sous-dossiers de
`submission/` par ordre alphabétique et prend les N premiers — avec `--num-cars 4`
sur `submission/`, ça charge `champion, driver1, driver2, driver3` et **ignore
silencieusement `driver4`**.

## Visualiser une course enregistrée (macOS)

`viz_replay.py` ouvre une fenêtre interactive (play/pause, zoom, avance pas à pas)
via X11 — macOS ne l'a pas nativement, il faut XQuartz :

```bash
# Une fois : installer XQuartz, puis dans XQuartz > Settings > Security,
# cocher "Allow connections from network clients", et relancer XQuartz.
brew install --cask xquartz

# À chaque session :
/opt/X11/bin/xhost + 127.0.0.1
docker compose run --rm replay python viz_replay.py --npz /app/recordings/episode_0001.npz --follow
```

Sans XQuartz / sans `DISPLAY` valide, `viz_replay.py` se rabat automatiquement sur
un mode dégradé : il sauvegarde juste **une image statique du départ**
(`..._snapshot.png`), pas la course animée.

## Structure du projet

### Moteur fourni (ne pas modifier)

- `env_simulation.py` — moteur physique (`f110_gym`), interface bas niveau
  (`reset`, `get_obs`, `apply_action`, `set_map`, `get_step_info`...)
- `gym/`, `maps/` — dépendances du moteur et fichiers de circuits
- `test_contract.py`, `test_integration.py` — tests officiels du moteur
- `agent_loader.py` — charge et valide les `agent.py` de chaque équipe (harnais de
  tournoi)
- `sim_recorder.py` / `viz_replay.py` — enregistrement et replay fournis
- `demo.py`, `setup.py`, `_harness.py` — scaffolding initial

### Code écrit pour ce projet

- `features.py` — prétraitement pur numpy partagé entre entraînement et inférence
  (normalisation LiDAR, encodage des adversaires, décodage des actions). Seule
  source de vérité des transformations, pour ne jamais désynchroniser train/serve.
- `env.py` — `RacingEnv`, wrapper Gymnasium solo (Part 1) : reward, terminaison,
  rotation sur les 5 circuits d'entraînement.
- `env_champion.py` — `ChampionRacingEnv(RacingEnv)` : ajoute les adversaires
  (pilotés par un modèle ONNX figé), observation enrichie à 123 dims.
- `train.py` / `train_champion.py` — boucles PPO (stable-baselines3), avec un
  `BestModelCallback` qui protège contre la non-monotonie de PPO sur longs
  entraînements, export ONNX à la fin.
- `pretrain_bc.py` / `pretrain_bc_champion.py` — warm-start par behavior cloning
  (heuristique Follow-The-Gap).
- `transfer_champion_init.py` — transplante les poids d'un modèle solo dans un
  réseau "opponent-aware" (technique clé des Parties 3 et 4).
- `race_drivers.py` — fait courir les 4 pilotes ensemble sur les 5 circuits et sort
  des statistiques agrégées (finish/DNF/progress/collisions).

### Soumission (ce qui compte pour l'évaluation)

- `submission/driver1..4/` et `submission/champion/` — chacun contient `agent.py` +
  `features.py` + `model.onnx`, strictement isolés (n'importent que numpy et
  onnxruntime, jamais torch/gymnasium) car chargés tous ensemble dans un seul
  process au moment du tournoi.

### Documentation

- `REPORT.md` — rapport complet (Parties 1 à 4) : choix de conception, bugs
  trouvés/corrigés, résultats de généralisation et de course.
